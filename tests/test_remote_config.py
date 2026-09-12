"""
What this service takes from mcp-config, and what it refuses to take.

Central configuration is only useful if the precedence is exactly what everyone assumes:
the config server overrides the code's defaults, and the machine in front of you overrides
both. Each test here pins one edge of that.
"""

from __future__ import annotations

import httpx
import respx

from app.remote_config import APPLICATION, Server, fetch

ENDPOINT = f"http://config:8888/{APPLICATION}/default"

SERVER = Server(url="http://config:8888", user="mcp", password="secret")


def document(*sources: tuple[str, dict[str, object]]) -> dict[str, object]:
    """An environment document in the shape Spring Cloud Config serves."""
    return {
        "name": APPLICATION,
        "propertySources": [
            {"name": name, "source": source} for name, source in sources
        ],
    }


@respx.mock
def test_it_reads_what_the_config_server_holds() -> None:
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json=document(
                (
                    "file [config-repo/mcp-server.yml]",
                    {
                        "mcp-server.router-provider": "openai_compatible",
                        "mcp-server.router-model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                    },
                )
            ),
        )
    )

    served, status = fetch(SERVER, set())

    assert status == ""
    assert served["router_provider"] == "openai_compatible"
    assert served["router_model"] == "Qwen/Qwen3-235B-A22B-Instruct-2507"


@respx.mock
def test_a_value_set_on_this_machine_is_not_replaced() -> None:
    """
    The precedence everyone assumes, and the reason a local experiment is possible.

    Left out of the result rather than fetched and discarded: a caller cannot apply by
    accident something that was never returned.
    """
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json=document(
                (
                    "file [config-repo/mcp-server.yml]",
                    {
                        "mcp-server.router-model": "central-model",
                        "mcp-server.router-provider": "openai_compatible",
                    },
                )
            ),
        )
    )

    served, _ = fetch(SERVER, {"ROUTER_MODEL"})

    assert "router_model" not in served
    assert served["router_provider"] == "openai_compatible"


@respx.mock
def test_a_placeholder_is_resolved_from_what_the_server_itself_served() -> None:
    """
    The step that is easy to leave out, and silent when it is missing.

    The config server does not substitute placeholders into the files it serves, so a value
    it wants to supply arrives as a separate property. Resolving only against this process's
    environment left the router with an empty key and a 401 from the model provider three
    steps later.
    """
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json=document(
                ("overrides", {"ROUTER_API_KEY": "from-the-config-server"}),
                (
                    "file [config-repo/mcp-server.yml]",
                    {"mcp-server.router-api-key": "${ROUTER_API_KEY:}"},
                ),
            ),
        )
    )

    served, _ = fetch(SERVER, set())

    assert served["open_model_api_key"] == "from-the-config-server"


@respx.mock
def test_a_key_nobody_here_asked_for_is_ignored() -> None:
    # The allow list: adding a key centrally cannot change how this process behaves until
    # someone in this repository decides that it should.
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json=document(
                (
                    "file [config-repo/mcp-server.yml]",
                    {
                        "mcp-server.router-model": "a-model",
                        "mcp-server.publisher-token": "surprise",
                    },
                )
            ),
        )
    )

    served, _ = fetch(SERVER, set())

    assert served == {"router_model": "a-model"}


@respx.mock
def test_the_services_own_file_wins_over_the_shared_one() -> None:
    # The precedence a Spring client applies. Earlier sources win.
    respx.get(ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json=document(
                ("file [config-repo/mcp-server.yml]", {"mcp-server.router-model": "mine"}),
                ("file [config-repo/application.yml]", {"mcp-server.router-model": "shared"}),
            ),
        )
    )

    served, _ = fetch(SERVER, set())

    assert served["router_model"] == "mine"


@respx.mock
def test_a_refusal_names_the_variables_that_fix_it() -> None:
    # "Answered 401" sends the reader to look at the config server. The fix is a pair of
    # variables on this side, so the message says which ones.
    respx.get(ENDPOINT).mock(return_value=httpx.Response(401))

    served, status = fetch(SERVER, set())

    assert served == {}
    assert "CONFIG_USER" in status and "CONFIG_PASSWORD" in status


@respx.mock
def test_an_unreachable_config_server_is_a_reason_rather_than_a_failure() -> None:
    """
    This service has to keep starting when that one is down.

    The alternative is an outage in the config server stopping every prompt everywhere,
    which is a worse failure than running on local settings and saying so.
    """
    respx.get(ENDPOINT).mock(side_effect=httpx.ConnectError("no route to host"))

    served, status = fetch(SERVER, set())

    assert served == {}
    assert "unreachable" in status


def test_no_address_is_not_an_attempt() -> None:
    # Running without a config server is a supported way to run this, not a degraded one.
    served, status = fetch(Server(), set())

    assert served == {}
    assert status == "no config server address is set"
