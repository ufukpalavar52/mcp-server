"""
Model backends.

A definition names a provider; this module turns that name into something that can be
asked to write one command or one query. The planner never learns which SDK answered —
it hands over a system prompt, a user prompt and a schema, and gets back text or a
reason it did not.

Three backends cover the field, because the field has consolidated on two wire
protocols:

* :class:`AnthropicBackend` speaks the Messages API.
* :class:`OpenAIBackend` speaks Chat Completions, which is also what Ollama, vLLM, LM
  Studio, llama.cpp, Together, Groq, Fireworks and OpenRouter serve. Running an open
  model is therefore a matter of pointing ``base_url`` at it, not of writing a fourth
  backend for each host.

What separates a hosted OpenAI model from a self hosted open one is not the protocol but
what the server implements: structured output is guaranteed by OpenAI and optional
everywhere else. :class:`OpenAIBackend` handles that difference itself rather than making
the caller care, which is what lets one class serve both.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import anthropic
import openai
from pydantic import BaseModel, ValidationError

from app.config import Settings

logger = logging.getLogger(__name__)

#: Providers the gateway can name. Mirrors its ``model_provider`` enum.
PROVIDERS = frozenset(
    {"anthropic", "openai_compatible", "azure", "vertex", "bedrock", "ollama", "custom"}
)

#: Model used when a definition names none. Only reached if a definition lost its model.
FALLBACK_MODEL = "claude-opus-5"

_MAX_OUTPUT_TOKENS = 2048

#: Hosts that are OpenAI itself rather than something wearing its API.
_OPENAI_HOSTS = ("api.openai.com",)

#: Wrapper a model puts around output when it was not constrained to a schema.
_FENCE = re.compile(r"^\s*```(?:[a-zA-Z]*)\s*\n(?P<body>.*?)\n?\s*```\s*$", re.S)


@dataclass(frozen=True)
class Authored:
    """
    What a backend produced.

    Failure is a value rather than an exception because every outcome ends up in the same
    place — a rejection reason on the plan — and the planner should not have to know which
    SDK's exception hierarchy to catch to get there.
    """

    #: The single command or query, for callers that asked for one.
    text: str | None = None

    #: The whole structured answer. Set whenever the model answered at all, which is what
    #: lets a caller use a schema of its own shape — the prompt router asks for a tool
    #: name, not a command, and reading `text` would tell it there was no usable output.
    parsed: BaseModel | None = None

    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None or self.parsed is not None

    @staticmethod
    def failed(reason: str) -> "Authored":
        return Authored(error=reason)


@dataclass(frozen=True)
class ModelOptions:
    """
    What the operator asked for, as far as a request can express it.

    Separate from the definition so the router can call a backend without one, and so a
    provider that rejects a setting can ignore it in one place. ``None`` means the field
    was not set: the provider's own default is used rather than a number chosen here.
    """

    temperature: float | None = None
    max_tokens: int | None = None

    @staticmethod
    def of(definition: Any) -> "ModelOptions":
        params = getattr(definition, "model_params", None)

        if params is None:
            return ModelOptions()
        return ModelOptions(temperature=params.temperature, max_tokens=params.max_tokens)

    def tokens(self) -> int:
        """The output ceiling, defaulting to this service's own when unset."""
        return self.max_tokens or _MAX_OUTPUT_TOKENS

    def sampling(self) -> dict[str, Any]:
        """Sampling arguments, present only when the operator set one."""
        return {} if self.temperature is None else {"temperature": self.temperature}


class ModelBackend(Protocol):
    """Writes one command or query, or explains why it did not."""

    async def author(
        self, *, model: str, system: str, user: str, schema: type[BaseModel],
        options: ModelOptions | None = None,
    ) -> Authored: ...


class AnthropicBackend:
    """Claude, through the Messages API."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def author(
        self, *, model: str, system: str, user: str, schema: type[BaseModel],
        options: ModelOptions | None = None,
    ) -> Authored:
        settings = options or ModelOptions()

        try:
            response = await self._client.messages.parse(
                model=model,
                # No temperature: recent Claude models reject sampling parameters, and the
                # panel refuses to set one for this provider. Depth is `effort` here.
                max_tokens=settings.tokens(),
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except anthropic.APIStatusError as exc:
            return Authored.failed(f"Model request failed ({exc.status_code})")
        except anthropic.APIConnectionError:
            return Authored.failed("The model endpoint is unreachable")

        if response.stop_reason == "refusal":
            return Authored.failed("The model declined to produce a command")

        return _read(response.parsed_output)


class OpenAIBackend:
    """
    Chat Completions: OpenAI itself, and everything that copies its API.

    One class for both because the request is identical; only the base URL and the key
    differ. The self hosted case is the reason for the fallback below.
    """

    def __init__(self, *, api_key: str, base_url: str = "", label: str = "openai") -> None:
        self._label = label
        self._client = openai.AsyncOpenAI(
            # A local server accepts any key and many reject an empty one outright, so a
            # placeholder is sent rather than nothing.
            api_key=api_key or "not-needed",
            base_url=base_url or None,
        )

    async def author(
        self, *, model: str, system: str, user: str, schema: type[BaseModel],
        options: ModelOptions | None = None,
    ) -> Authored:
        settings = options or ModelOptions()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            completion = await self._client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=schema,
                max_completion_tokens=settings.tokens(),
                **settings.sampling(),
            )
        except openai.BadRequestError as exc:
            # Structured output is an OpenAI guarantee and an optional extra elsewhere.
            # A 400 here usually means the server does not implement it, so the request is
            # made again in a form every Chat Completions server understands rather than
            # reporting a failure the operator cannot act on.
            logger.info(
                "%s rejected a schema constrained request for %s (%s); retrying as text",
                self._label, model, exc,
            )
            return await self._author_unconstrained(model, messages, schema, settings)
        except openai.LengthFinishReasonError:
            # The answer was cut off mid-sentence. Said plainly, because the setting that
            # fixes it is one the operator can change.
            return Authored.failed(
                "The model's answer was cut off; raise the model's max tokens"
            )
        except ValidationError:
            # A 200 carrying JSON that does not parse. Accepting a schema and honouring it
            # are different things, and several servers advertise the first: this one
            # answered with an unterminated string, and the exception travelled all the way
            # out as a 500 that said "Internal Server Error" to the person who asked a
            # question about their database.
            logger.warning(
                "%s returned malformed structured output for %s; retrying as text",
                self._label, model,
            )
            return await self._author_unconstrained(model, messages, schema, settings)
        except openai.APIStatusError as exc:
            return Authored.failed(f"Model request failed ({exc.status_code})")
        except openai.APIConnectionError:
            return Authored.failed(f"The {self._label} endpoint is unreachable")

        message = completion.choices[0].message
        if message.refusal:
            return Authored.failed("The model declined to produce a command")

        return _read(message.parsed)

    async def _author_unconstrained(
        self, model: str, messages: list[dict[str, str]], schema: type[BaseModel],
        settings: ModelOptions,
    ) -> Authored:
        """
        Ask again without a schema, and read what comes back.

        The schema is described in the prompt instead of enforced by the server. That is
        strictly weaker, which is why the result is validated here and a model that
        ignores the instruction is reported rather than guessed at. It is also why the
        guardrails matter more on this path than on any other: nothing upstream
        constrained the shape of what arrived.
        """
        instructed = [
            *messages,
            {
                "role": "system",
                "content": (
                    "Reply with a single JSON object and nothing else, matching this "
                    f"schema: {json.dumps(schema.model_json_schema())}"
                ),
            },
        ]

        try:
            completion = await self._client.chat.completions.create(
                model=model,
                messages=instructed,
                max_completion_tokens=settings.tokens(),
                **settings.sampling(),
            )
        except openai.APIStatusError as exc:
            return Authored.failed(f"Model request failed ({exc.status_code})")
        except openai.APIConnectionError:
            return Authored.failed(f"The {self._label} endpoint is unreachable")

        content = completion.choices[0].message.content
        if not content:
            return Authored.failed("The model returned an empty response")

        return _parse_loose(content, schema)


def _parse_loose(content: str, schema: type[BaseModel]) -> Authored:
    """
    Read a schema instance out of unconstrained model output.

    Tolerates a code fence, because a model told to emit JSON very often emits it inside
    one. Nothing beyond that is tolerated: a response that is not the requested object is
    an error, not something to salvage by pattern matching, since the value being
    extracted is about to become a shell command.
    """
    body = content.strip()
    if fenced := _FENCE.match(body):
        body = fenced.group("body").strip()

    try:
        return _read(schema.model_validate_json(body))
    except ValidationError:
        logger.warning("Model output did not match %s", schema.__name__)
        return Authored.failed(
            "The model did not answer in the required format; this endpoint may not "
            "support structured output"
        )


def _read(parsed: BaseModel | None) -> Authored:
    """
    Turns a parsed answer into a result.

    ``text`` is filled only for the schemas that carry a single command or query, because
    that is what the planner wants and hunting for it at every call site would be worse.
    ``parsed`` is always kept: a caller whose schema is a different shape has somewhere to
    read from, instead of being told there was no usable output.
    """
    if parsed is None:
        return Authored.failed("The model produced no usable output")

    # A schema that declares a command or a query is one whose answer *is* that string, so
    # an empty one is no answer. A schema that declares neither is a different question
    # entirely — the prompt router asks for a tool name — and is handed back whole.
    declared = [field for field in ("command", "query") if hasattr(parsed, field)]

    if declared:
        for field in declared:
            if value := getattr(parsed, field, None):
                return Authored(text=value, parsed=parsed)
        return Authored.failed("The model produced no usable output")

    return Authored(parsed=parsed)


class BackendRegistry:
    """
    Picks a backend for a definition, and remembers the ones it built.

    Keyed by provider *and* endpoint: two definitions may both be
    ``openai_compatible`` while pointing at different servers, and sharing one client
    between them would send a local model's request to a hosted one.
    """

    def __init__(self, settings: Settings, cipher: Any = None) -> None:
        self._settings = settings
        self._cipher = cipher
        self._cache: dict[tuple[str, str], ModelBackend] = {}

    def for_model(self, definition: Any) -> ModelBackend | None:
        """
        The backend for a definition, using its own stored API key when it has one.

        Preferred over :meth:`for_definition` because the key belongs to the model, not to
        this process. Falling back to the environment keeps a deployment working that has
        no cipher, or a definition whose model carries no key.
        """
        sealed = getattr(definition, "model_api_key", None)

        if sealed is not None and self._cipher is not None and self._cipher.configured:
            try:
                key = self._cipher.open(sealed)
            except Exception as exc:  # noqa: BLE001 - reported as an unusable backend
                logger.error("Could not open the model's API key: %s", exc)
                return None

            # Not cached by (provider, endpoint) like the others: two definitions may share
            # an endpoint and use different keys, and reusing one client would send the
            # wrong credential.
            return self._build_openai(
                definition.model_provider or "anthropic",
                definition.model_endpoint,
                key,
            )

        return self.for_definition(definition.model_provider, definition.model_endpoint)

    def _build_openai(self, provider: str, endpoint: str, key: str) -> ModelBackend:
        if provider.lower() == "anthropic":
            return AnthropicBackend(key)

        return OpenAIBackend(
            api_key=key,
            base_url=endpoint or self._settings.openai_base_url,
            label=provider,
        )

    def for_definition(self, provider: str | None, endpoint: str) -> ModelBackend | None:
        """
        The backend for this definition, or ``None`` when it cannot be served.

        ``None`` means a missing key or an unreachable configuration, never an unknown
        provider name: an unrecognised provider is treated as OpenAI compatible, because
        that is what an unfamiliar server almost always turns out to speak.
        """
        name = (provider or "anthropic").lower()
        key = (name, endpoint or "")

        if key not in self._cache:
            backend = self._build(name, endpoint)
            if backend is None:
                return None
            self._cache[key] = backend

        return self._cache[key]

    def unavailable_reason(self, provider: str | None, endpoint: str) -> str:
        """Why :meth:`for_definition` returned nothing, phrased for a plan's problems."""
        name = (provider or "anthropic").lower()

        if name == "anthropic":
            return (
                "This model has no stored API key and no ANTHROPIC_API_KEY is set. Enter "
                "the key on the model in the panel, or configure one here"
            )
        return (
            "This model has no stored API key and none is configured here. Enter the key "
            "on the model in the panel"
        )

    def _build(self, provider: str, endpoint: str) -> ModelBackend | None:
        if provider == "anthropic":
            if not self._settings.has_anthropic_key:
                return None
            return AnthropicBackend(self._settings.anthropic_api_key)

        # Everything else speaks Chat Completions. An endpoint on the model wins over the
        # configured default: it is the more specific statement of where this model lives.
        base_url = endpoint or self._settings.openai_base_url
        api_key = self._key_for(base_url)

        # A local server needs no key, so an endpoint alone is enough to proceed. With
        # neither an endpoint nor a key there is nowhere to send the request.
        if not api_key and not base_url:
            return None

        return OpenAIBackend(api_key=api_key, base_url=base_url, label=provider)

    def _key_for(self, base_url: str) -> str:
        """
        The credential for this destination.

        Chosen by where the request is going, not by which provider was selected. The two
        can disagree: a model marked ``openai_compatible`` may point at Groq, Together or
        a laptop, and the panel fills in an endpoint for every provider it offers. Keying
        off the provider name would have sent the OpenAI key to whichever host the
        endpoint named — a leaked credential rather than a failed call.

        No fallback between the two, in either direction, for the same reason.
        """
        if not base_url or _is_openai_host(base_url):
            return self._settings.openai_api_key
        return self._settings.open_model_api_key


def _is_openai_host(base_url: str) -> bool:
    """Whether a base URL addresses OpenAI, as opposed to an API shaped like theirs."""
    host = urlsplit(base_url).hostname or ""
    return host.lower() in _OPENAI_HOSTS
