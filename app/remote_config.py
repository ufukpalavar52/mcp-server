"""
Reads the settings mcp-config holds for this service.

Not a Spring client, so the two things a Spring client gets for free happen here: fetching
``/{application}/{profile}``, and resolving the ``${NAME:default}`` placeholders the config
server deliberately leaves alone. mcp-action and mcp-cipher do the same in Go; this is the
third copy of a small idea, and the alternative was a Spring service in the middle of a
Python one.

What is read remotely is which model routes a prompt and how to reach it. That is a
deployment fact, not a property of any definition: routing runs before a tool is chosen, so
there is no definition to take a model or a key from. Keeping it central means changing the
router is an edit in one place rather than an edit on every host that runs this service.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ENV_CONFIG_USER = "CONFIG_USER"
ENV_CONFIG_PASSWORD = "CONFIG_PASSWORD"

APPLICATION = "mcp-server"
PREFIX = "mcp-server."

TIMEOUT_SECONDS = 5.0

#: The allow list, served field to settings field.
#:
#: Anything else served under the prefix is ignored, so adding a key centrally cannot
#: change how this process behaves until someone here decides that it should.
SETTABLE: dict[str, str] = {
    "router-provider": "router_provider",
    "router-endpoint": "router_endpoint",
    "router-model": "router_model",
    "router-api-key": "open_model_api_key",
    "cipher-address": "cipher_address",
    "cipher-token": "cipher_token",
    "queue-host": "queue_host",
    "queue-port": "queue_port",
    "queue-user": "queue_user",
    "queue-password": "queue_password",
    "queue-vhost": "queue_vhost",
}

#: The environment variable that overrides each served field.
#:
#: A value set on this machine is never replaced by a central one — the precedence a Spring
#: client applies: central configuration overrides the code's defaults, and the machine in
#: front of you overrides both.
OVERRIDES: dict[str, str] = {
    "router-provider": "ROUTER_PROVIDER",
    "router-endpoint": "ROUTER_ENDPOINT",
    "router-model": "ROUTER_MODEL",
    "router-api-key": "OPEN_MODEL_API_KEY",
    "cipher-address": "MCP_CIPHER_ADDRESS",
    "cipher-token": "MCP_CIPHER_TOKEN",
    "queue-host": "QUEUE_HOST",
    "queue-port": "QUEUE_PORT",
    "queue-user": "QUEUE_USER",
    "queue-password": "QUEUE_PASSWORD",
    "queue-vhost": "QUEUE_VHOST",
}

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z0-9_.-]+)(?::([^}]*))?}")


@dataclass(frozen=True)
class Server:
    """Where mcp-config is, and who this service says it is."""

    url: str = ""
    user: str = ""
    password: str = ""
    profile: str = "default"


def fetch(server: Server, local: set[str]) -> tuple[dict[str, str], str]:
    """
    The settings the config server holds, and why there are none when there are none.

    ``local`` names the variables this machine already set; a field they cover is left out
    of the result rather than fetched and discarded, so the caller cannot apply it by
    accident.

    Being unable to reach the config server is not an error. This service has to keep
    starting when that one is down — the alternative is an outage there stopping every
    prompt everywhere.
    """
    base = server.url.rstrip("/")
    if not base:
        return {}, "no config server address is set"

    endpoint = f"{base}/{APPLICATION}/{server.profile or 'default'}"

    # mcp-config refuses an anonymous caller: it hands out the broker password, the cipher
    # token and the router's API key.
    auth = (server.user, server.password) if server.user else None

    try:
        response = httpx.get(endpoint, auth=auth, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        return {}, f"{endpoint} is unreachable: {exc}"

    if response.status_code == 401:
        # Named, because the fix is a pair of variables rather than anything about this
        # service: "answered 401" sends the reader looking at the config server instead.
        return {}, (
            f"{endpoint} refused the credentials; "
            f"set {ENV_CONFIG_USER} and {ENV_CONFIG_PASSWORD}"
        )

    if response.status_code != 200:
        return {}, f"{endpoint} answered {response.status_code}"

    try:
        served = _flatten(response.json())
    except ValueError as exc:
        return {}, f"{endpoint} returned an unusable body: {exc}"

    values: dict[str, str] = {}

    for name, value in served.items():
        if not name.startswith(PREFIX):
            continue

        field = name[len(PREFIX):]
        target = SETTABLE.get(field)

        if target is None or OVERRIDES.get(field, "") in local:
            continue

        resolved = _resolve(value, served)
        if resolved:
            values[target] = resolved

    return values, ""


def _flatten(document: Any) -> dict[str, Any]:
    """
    Collapses the property sources into one map.

    Earlier sources win, which is the precedence a Spring client applies: the service's own
    file overrides the shared one.
    """
    sources = document.get("propertySources") if isinstance(document, dict) else None
    if not isinstance(sources, list):
        raise ValueError("no propertySources in the response")

    merged: dict[str, Any] = {}
    for source in reversed(sources):
        values = source.get("source") if isinstance(source, dict) else None
        if isinstance(values, dict):
            merged.update(values)

    return merged


def _resolve(value: Any, served: dict[str, Any]) -> str:
    """
    Substitutes ``${NAME:default}``, in the order a Spring client would.

    The environment first, then the properties the config server itself served, then the
    inline default. That middle step is not optional: the config server cannot substitute a
    placeholder into a file it serves, so a value it wants to supply arrives as a separate
    property — it sends ``MCP_CIPHER_TOKEN`` alongside ``mcp-server.cipher-token=${MCP_CIPHER_TOKEN:}``.
    Looking only at the environment resolved that to the empty default and quietly started
    the service with authentication switched off.
    """
    if not isinstance(value, str):
        return str(value)

    def substitute(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2) or ""

        found = os.environ.get(name, "")
        if found:
            return found

        served_value = served.get(name)
        if isinstance(served_value, str) and served_value:
            return served_value

        return fallback

    return _PLACEHOLDER.sub(substitute, value)
