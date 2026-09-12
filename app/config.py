"""Runtime configuration, read from the environment and from mcp-config."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.remote_config import OVERRIDES, Server, fetch


class Settings(BaseSettings):
    """
    Settings for the MCP server.

    Short by design. This service holds no database credentials and no gateway
    credentials, because it calls neither: the gateway pushes what it needs to know.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str = Field(
        default="",
        description=(
            "Key for definitions whose model provider is Anthropic. Model calls are this "
            "service's only outbound dependency, and they exist because writing a command "
            "is the job. Interim: keys belong in the secret vault once that service exists."
        ),
    )

    openai_api_key: str = Field(
        default="",
        description=(
            "Key for OpenAI, and for any OpenAI compatible endpoint that wants one. "
            "A self hosted server usually does not, which is why an endpoint can be used "
            "without a key at all."
        ),
    )

    openai_base_url: str = Field(
        default="",
        description=(
            "Default endpoint for OpenAI compatible providers, used when the model does "
            "not name one. Empty means api.openai.com. Point it at Ollama, vLLM, LM "
            "Studio or a gateway to run open models with no other change."
        ),
    )

    open_model_api_key: str = Field(
        default="",
        description=(
            "Key presented to a self hosted or third party open model endpoint, when the "
            "model names its own endpoint. Kept apart from OPENAI_API_KEY so a key meant "
            "for a local server is never sent to OpenAI, or the other way round."
        ),
    )

    #: Read as MCP_CIPHER_ADDRESS, which is what mcp-action and mcp-config already call it.
    #: The plain name is accepted too, so neither spelling is a trap.
    cipher_address: str = Field(
        default="",
        validation_alias=AliasChoices("MCP_CIPHER_ADDRESS", "CIPHER_ADDRESS"),
        description=(
            "Where mcp-cipher listens. Without it a definition's own API key cannot be "
            "opened and only the keys below are usable."
        ),
    )
    cipher_token: str = Field(
        default="",
        validation_alias=AliasChoices("MCP_CIPHER_TOKEN", "CIPHER_TOKEN"),
    )

    router_model: str = Field(
        default="",
        description=(
            "Model that picks a tool for a prompt. Distinct from the models on the "
            "definitions: nothing is chosen yet when this runs, so it cannot come from one."
        ),
    )
    router_provider: str = Field(default="anthropic")
    router_endpoint: str = Field(default="")

    queue_host: str = Field(default="", description="RabbitMQ host. Empty disables dispatch.")
    queue_port: int = 5672
    queue_user: str = "guest"
    queue_password: str = ""
    queue_vhost: str = "/"

    #: Where executable jobs are published for mcp-action.
    queue_name: str = "mcp.actions"

    #: Shared secret the gateway presents when publishing or requesting an execution.
    publisher_token: str = Field(
        default="",
        description="When set, callers must send it as X-MCP-Token.",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    #: Where a copy of the log is written, alongside the console.
    #:
    #: A shared directory rather than a file path, so every service in the stack names its
    #: own file inside one place and the log shipper has one directory to watch.
    #:
    #: Set it to nothing to turn the file off. Nothing is lost either way — the console
    #: keeps everything, which is what an IDE shows.
    log_dir: str = Field(
        default="~/mcp-logs",
        validation_alias=AliasChoices("MCP_LOG_DIR", "log_dir"),
    )

    config_server_url: str = Field(
        default="",
        validation_alias=AliasChoices("CONFIG_SERVER_URL", "config_server_url"),
        description=(
            "Where mcp-config lives. Empty leaves this service on its local settings, "
            "which is a supported way to run it rather than a degraded one."
        ),
    )
    config_user: str = Field(
        default="", validation_alias=AliasChoices("CONFIG_USER", "config_user")
    )
    config_password: str = Field(
        default="", validation_alias=AliasChoices("CONFIG_PASSWORD", "config_password")
    )
    config_profile: str = Field(
        default="default", validation_alias=AliasChoices("CONFIG_PROFILE", "config_profile")
    )

    #: Why nothing came from mcp-config, when nothing did. Empty means it was read.
    #:
    #: Not read from the environment: it describes what happened at start up rather than
    #: something anyone configures. Reported by /health so an operator can see that a
    #: service is running on its local fallback before a call fails because of it.
    remote_status: str = Field(default="", exclude=True)

    #: Path the MCP streamable HTTP transport is mounted at.
    mcp_path: str = "/mcp"

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_cipher(self) -> bool:
        return bool(self.cipher_address)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def configured_providers(self) -> list[str]:
        """
        Providers this deployment can actually reach.

        Reported by /health so an operator sees the gap before a call fails: a definition
        can name a provider this service has no key for, and that is a configuration
        problem rather than a fault in the definition.
        """
        available: list[str] = []
        if self.has_anthropic_key:
            available.append("anthropic")
        if self.has_openai_key or self.openai_base_url:
            available.append("openai_compatible")
        # A local server usually needs no key at all, so it counts as reachable whenever
        # an endpoint is configured — or when the definition carries its own.
        available.append("self_hosted")
        return available

    @property
    def queue_url(self) -> str:
        """
        AMQP connection string. Never logged: it carries the broker password.
        """
        from urllib.parse import quote

        return (
            f"amqp://{quote(self.queue_user)}:{quote(self.queue_password)}"
            f"@{self.queue_host}:{self.queue_port}/{quote(self.queue_vhost, safe='')}"
        )

    @property
    def has_queue(self) -> bool:
        return bool(self.queue_host)

    @property
    def requires_token(self) -> bool:
        return bool(self.publisher_token)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Settings are read once; the process must be restarted to pick up changes.

    Local first, then whatever mcp-config holds for the fields this machine did not set.
    That order is the one every other service in this stack applies: central configuration
    overrides the code's defaults, and the machine in front of you overrides both.

    Passed to the constructor rather than assigned afterwards, so the served values are
    validated and coerced like any other — a port arrives as a string and has to become an
    integer somewhere, and it should not be here.
    """
    local = Settings()

    # Read from the local settings rather than from os.environ: the address and credentials
    # are ordinary configuration and belong in .env like everything else, and a fetch that
    # only looked at exported variables would find nothing there.
    served, status = fetch(
        Server(
            url=local.config_server_url,
            user=local.config_user,
            password=local.config_password,
            profile=local.config_profile,
        ),
        _locally_set(),
    )

    settings = Settings(**served) if served else local
    settings.remote_status = status
    return settings


def _locally_set() -> set[str]:
    """
    Which of the overriding variables this machine sets.

    The environment and ``.env`` both count. A value written into ``.env`` is a deliberate
    statement about this host — the same kind of statement as exporting it — and treating
    it as a default waiting to be replaced centrally would silently ignore what somebody
    wrote in front of them.
    """
    names = set(OVERRIDES.values())
    found = {name for name in names if os.environ.get(name)}

    env_file = Path(Settings.model_config.get("env_file", ".env"))
    if not env_file.is_file():
        return found

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, _, value = line.partition("=")
        if name.strip() in names and value.strip():
            found.add(name.strip())

    return found
