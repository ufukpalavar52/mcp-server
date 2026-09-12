"""Shared fixtures. Nothing here talks to a network, and nothing writes a log file."""

from __future__ import annotations

import os

# Before anything imports the application, because `app.main` builds one at module level —
# `app = create_app()` is what lets uvicorn say `app.main:app`, and it runs on import,
# which is during collection and long before any fixture.
#
# A fixture was the obvious place for this and it did not work: the file was already open
# by the time the first one ran. File logging is on by default, which is what makes it work
# for somebody who has configured nothing; under test it meant every run leaving something
# behind in the home directory of the machine it ran on.
os.environ["MCP_LOG_DIR"] = ""

import pytest

from app.config import Settings
from app.models import Action, ActionConfig, Definition, DefinitionInput


@pytest.fixture
def settings() -> Settings:
    """No Anthropic key, so the dynamic path is exercised as unavailable."""
    return Settings(anthropic_api_key="", publisher_token="")


@pytest.fixture
def ssh_definition() -> Definition:
    """A definition with one static SSH action against a host group."""
    return Definition(
        id=1,
        name="Apache fleet",
        toolName="apache_fleet",
        toolDescription="Restarts apache across the web fleet.",
        modelId=1,
        modelIdentifier="claude-opus-5",
        systemPrompt="You are a systems assistant.",
        inputs=[
            DefinitionInput(
                key="service", label="Service", type="select",
                required=True, defaultValue="apache2", options=["apache2", "nginx"],
            ),
            DefinitionInput(key="ssh_key", label="SSH key", type="password", required=False),
        ],
        actions=[
            Action(
                id=10, kind="ssh", name="Restart", position=0,
                hostGroupId=1, hostGroupName="web-servers", resolvedTargetCount=3,
                config=ActionConfig(
                    targetMode="group", strategy="rolling", batchSize=2,
                    port=22, user="deploy", auth="key",
                    privateKeyInputKey="ssh_key",
                    commandMode="static",
                    command="systemctl restart {{service}}",
                    sudo=True,
                ),
            )
        ],
    )


@pytest.fixture
def dynamic_db_definition() -> Definition:
    """A definition whose single database action is in dynamic mode."""
    return Definition(
        id=2,
        name="Ad hoc report",
        toolName="ad_hoc_report",
        toolDescription="Answers questions about tool usage.",
        modelIdentifier="claude-opus-5",
        systemPrompt="You are a reporting assistant.",
        inputs=[DefinitionInput(key="since", label="Since", type="date", required=True)],
        actions=[
            Action(
                id=20, kind="db", name="Query", position=0,
                config=ActionConfig(
                    engine="postgres", host="db.internal", database="metrics",
                    queryMode="dynamic",
                    schemaHint="tool_calls(tool, created_at)",
                    allowedOperations=["select"], maxRows=100, readOnly=True,
                ),
            )
        ],
    )
