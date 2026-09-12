"""
Which backend serves a definition, and what happens when it cannot be held to a schema.

These are the two things that separate a multi provider service from a single provider
one, so they are the two things pinned here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2
import openai
import pytest

from pydantic import BaseModel

from app.config import Settings
from app.providers import (
    AnthropicBackend,
    Authored,
    BackendRegistry,
    OpenAIBackend,
    _parse_loose,
)


class Command(BaseModel):
    command: str
    reasoning: str = ""


class TestBackendSelection:
    def test_anthropic_definition_uses_the_anthropic_backend(self):
        registry = BackendRegistry(Settings(anthropic_api_key="key"))

        assert isinstance(registry.for_definition("anthropic", ""), AnthropicBackend)

    def test_a_definition_with_no_provider_is_treated_as_anthropic(self):
        """Older catalogue payloads carry no provider; they must keep working."""
        registry = BackendRegistry(Settings(anthropic_api_key="key"))

        assert isinstance(registry.for_definition(None, ""), AnthropicBackend)

    def test_openai_definition_uses_the_openai_backend(self):
        registry = BackendRegistry(Settings(openai_api_key="key"))

        assert isinstance(
            registry.for_definition("openai_compatible", ""), OpenAIBackend
        )

    def test_the_openai_key_goes_to_openai_and_nowhere_else(self):
        """
        The panel fills in an endpoint for every provider, including OpenAI's own, so the
        key cannot be chosen by asking whether an endpoint is present. Sending the OpenAI
        key to a third party host would be a leaked credential, not a failed call.
        """
        settings = Settings(openai_api_key="sk-openai", open_model_api_key="local-key")
        registry = BackendRegistry(settings)

        assert registry._key_for("https://api.openai.com/v1") == "sk-openai"  # noqa: SLF001
        assert registry._key_for("") == "sk-openai"  # noqa: SLF001
        assert registry._key_for("https://api.groq.com/openai/v1") == "local-key"  # noqa: SLF001
        assert registry._key_for("http://localhost:11434/v1") == "local-key"  # noqa: SLF001

    def test_a_self_hosted_endpoint_needs_no_key_at_all(self):
        """
        The whole point of the endpoint field: a local server is reachable with nothing
        configured, which is how an open model runs without credentials.
        """
        registry = BackendRegistry(Settings())

        backend = registry.for_definition("ollama", "http://localhost:11434/v1")

        assert isinstance(backend, OpenAIBackend)

    def test_an_unknown_provider_falls_back_to_openai_compatible(self):
        """An unfamiliar server almost always speaks Chat Completions."""
        registry = BackendRegistry(Settings(openai_api_key="key"))

        assert isinstance(registry.for_definition("something-new", ""), OpenAIBackend)

    def test_a_missing_key_is_reported_rather_than_guessed(self):
        """
        The message names the panel first, because that is where the key belongs now: a
        model carries its own, and an environment variable is the fallback.
        """
        registry = BackendRegistry(Settings())

        assert registry.for_definition("anthropic", "") is None
        assert "panel" in registry.unavailable_reason("anthropic", "")

        assert registry.for_definition("openai_compatible", "") is None
        assert "panel" in registry.unavailable_reason("openai_compatible", "")

    def test_two_endpoints_do_not_share_one_client(self):
        """
        A cache keyed by provider alone would send a local model's request to a hosted
        one, which is a data leak rather than a mix up.
        """
        registry = BackendRegistry(Settings(openai_api_key="key"))

        first = registry.for_definition("ollama", "http://a.test/v1")
        second = registry.for_definition("ollama", "http://b.test/v1")

        assert first is not second
        assert registry.for_definition("ollama", "http://a.test/v1") is first

    def test_a_model_endpoint_wins_over_the_configured_default(self):
        registry = BackendRegistry(
            Settings(openai_api_key="key", openai_base_url="http://default.test/v1")
        )

        backend = registry.for_definition("custom", "http://model.test/v1")

        assert str(backend._client.base_url).startswith("http://model.test")  # noqa: SLF001


class TestLooseParsing:
    """The fallback path, for servers that cannot be constrained to a schema."""

    def test_plain_json_is_read(self):
        result = _parse_loose(json.dumps({"command": "systemctl restart nginx"}), Command)

        assert result.ok
        assert result.text == "systemctl restart nginx"

    def test_a_code_fence_is_tolerated(self):
        """A model told to emit JSON very often wraps it in a fence anyway."""
        body = '```json\n{"command": "uptime"}\n```'

        assert _parse_loose(body, Command).text == "uptime"

    def test_prose_is_an_error_not_a_salvage_attempt(self):
        """
        The extracted value becomes a shell command, so anything short of the requested
        object is refused rather than pattern matched out of the surrounding text.
        """
        result = _parse_loose("Sure! Run `rm -rf /` to clean up.", Command)

        assert not result.ok
        assert "structured output" in (result.error or "")

    def test_an_empty_field_counts_as_no_output(self):
        assert not _parse_loose('{"command": ""}', Command).ok


class TestAuthored:
    def test_failure_carries_a_reason(self):
        result = Authored.failed("nope")

        assert not result.ok
        assert result.error == "nope"


class TestUnsupportedStructuredOutput:
    """
    A server that cannot be held to a schema must still be usable.

    This is the difference between supporting open models on paper and in practice:
    OpenAI guarantees structured output, most self hosted servers treat it as optional,
    and a 400 from one of them must not read to the operator as a broken definition.
    """

    @pytest.mark.anyio
    async def test_a_rejected_schema_is_retried_as_plain_text(self):
        backend = OpenAIBackend(api_key="k", base_url="http://local.test/v1")
        backend._client = _ClientRefusingSchemas(  # noqa: SLF001
            '{"command": "df -h /var"}'
        )

        result = await backend.author(
            model="llama3.2", system="s", user="u", schema=Command
        )

        assert result.ok
        assert result.text == "df -h /var"

    @pytest.mark.anyio
    async def test_a_server_that_ignores_the_instruction_is_reported(self):
        """Not salvaged: the value is about to become a shell command."""
        backend = OpenAIBackend(api_key="k", base_url="http://local.test/v1")
        backend._client = _ClientRefusingSchemas("Here you go: df -h")  # noqa: SLF001

        result = await backend.author(
            model="llama3.2", system="s", user="u", schema=Command
        )

        assert not result.ok
        assert "structured output" in (result.error or "")


class _ClientRefusingSchemas:
    """Stands in for a server whose Chat Completions has no json_schema support."""

    def __init__(self, plain_answer: str) -> None:
        self.chat = _Chat(plain_answer)


class _Chat:
    def __init__(self, plain_answer: str) -> None:
        self.completions = _Completions(plain_answer)


class _Completions:
    def __init__(self, plain_answer: str) -> None:
        self._plain = plain_answer

    async def parse(self, **_kwargs):
        raise openai.BadRequestError(
            "response_format is not supported",
            response=httpx2.Response(400, request=httpx2.Request("POST", "http://local.test")),
            body=None,
        )

    async def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._plain))]
        )


class TestDefinitionOwnedKeys:
    """
    A model's own API key, opened through mcp-cipher.

    Before this, the key was stored against the model here and read from this process's
    environment there — so a key entered in the panel did nothing for the one thing that
    needed it, and the failure looked like a 401 from the provider.
    """

    class _Cipher:
        configured = True

        def __init__(self, plaintext="sk-from-the-panel"):
            self.plaintext = plaintext
            self.opened = 0

        def open(self, sealed):
            self.opened += 1
            return self.plaintext

    @staticmethod
    def _definition(**overrides):
        from app.models import Definition

        return Definition.model_validate(
            {
                "id": 1,
                "name": "x",
                "toolName": "x",
                "modelProvider": "custom",
                "modelEndpoint": "https://api.deepinfra.com/v1/openai",
                "modelApiKey": {
                    "ciphertext": "AQID",
                    "keyId": "v1",
                    "context": "model_api_key",
                },
                **overrides,
            }
        )

    def test_the_stored_key_is_used(self):
        cipher = self._Cipher()
        registry = BackendRegistry(Settings(), cipher)

        backend = registry.for_model(self._definition())

        assert isinstance(backend, OpenAIBackend)
        assert cipher.opened == 1
        assert backend._client.api_key == "sk-from-the-panel"  # noqa: SLF001

    def test_a_definition_without_a_key_falls_back_to_the_environment(self):
        cipher = self._Cipher()
        registry = BackendRegistry(Settings(open_model_api_key="from-env"), cipher)

        backend = registry.for_model(self._definition(modelApiKey=None))

        assert cipher.opened == 0
        assert backend._client.api_key == "from-env"  # noqa: SLF001

    def test_an_unopenable_key_is_reported_rather_than_ignored(self):
        """
        Falling back to the environment here would call the provider with the wrong key
        and produce a 401 that says nothing about the real problem.
        """

        class _Broken(self._Cipher):
            def open(self, sealed):
                raise RuntimeError("cipher is down")

        registry = BackendRegistry(Settings(open_model_api_key="from-env"), _Broken())

        assert registry.for_model(self._definition()) is None


class TestAMalformedAnswerIsReportedNotRaised:
    """
    A model that answers with broken JSON is a bad answer, not a crash.

    DeepInfra accepted a schema constrained request, returned 200, and put an
    unterminated string in the body. The parse error travelled out of this service as a
    500, and the console showed "Internal Server Error" for a question about a database.
    """

    async def test_unparseable_structured_output_falls_back_to_text(self, monkeypatch):
        import openai
        from pydantic import BaseModel, ValidationError

        from app.providers import OpenAIBackend

        class _Query(BaseModel):
            query: str = ""
            reasoning: str = ""

        backend = OpenAIBackend(api_key="test", base_url="http://localhost:1", label="test")

        async def _broken(*_args, **_kwargs):
            raise ValidationError.from_exception_data("_Query", [])

        async def _good(model, messages, schema, settings):
            return type("R", (), {"ok": True, "text": "SELECT 1", "parsed": None})()

        monkeypatch.setattr(backend._client.chat.completions, "parse", _broken)  # noqa: SLF001
        monkeypatch.setattr(backend, "_author_unconstrained", _good)

        result = await backend.author(
            model="m", system="s", user="u", schema=_Query
        )

        assert result.text == "SELECT 1"

    async def test_a_truncated_answer_names_the_setting_that_fixes_it(self, monkeypatch):
        import openai
        from pydantic import BaseModel

        from app.providers import OpenAIBackend

        class _Query(BaseModel):
            query: str = ""

        backend = OpenAIBackend(api_key="test", base_url="http://localhost:1", label="test")

        class _Completion:
            """Only what the exception reads from it."""

            usage = None

        async def _cut_off(*_args, **_kwargs):
            raise openai.LengthFinishReasonError(completion=_Completion())

        monkeypatch.setattr(backend._client.chat.completions, "parse", _cut_off)  # noqa: SLF001

        result = await backend.author(model="m", system="s", user="u", schema=_Query)

        assert not result.ok
        assert "max tokens" in (result.error or "")
