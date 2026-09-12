"""
Turns a decided plan into work an executor can carry out.

Two documents come out of one decision, and they are deliberately different. The
:class:`~app.planner.Plan` is written to be read: its targets say
``<2 host(s) in web-tier>`` and its commands have secrets replaced with bullets. A
:class:`Job` is written to be run: real host names, the command as it will actually be
typed, and credentials still sealed.

The unmasked command is built here rather than being carried on the plan. A plan is
serialised straight into an HTTP response and shown in a browser; a field holding the
real command would leak every secret substituted into it the first time anyone looked at
a definition. Keeping the two apart means that cannot happen by forgetting.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models import Action, Definition
from app.planner import Plan
from app.templating import build_values, render, render_body, render_url


class IncompleteJob(Exception):
    """A job could not be built because an action had nothing to run."""


class Job(BaseModel):
    """One dispatched plan, matching mcp-action's contract."""

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    run_id: str = Field(alias="runId")
    tool_name: str = Field(alias="toolName")
    definition_id: int = Field(alias="definitionId")
    actor: str = ""

    #: The acting user's id, for the executor's log lines.
    #:
    #: Beside `actor` rather than instead of it: that one is a label kept with the run and
    #: read by a person in the console. This one goes only into logs, and it is an id
    #: because a log store with no encryption is the wrong place for an address.
    actor_id: str = Field(default="", alias="actorId")

    actions: list[dict[str, Any]] = Field(default_factory=list)
    timeout_seconds: int = Field(default=0, alias="timeoutSeconds")

    #: A real timestamp, not an empty string. The executor parses this into a time and
    #: rejects the whole job when it cannot — which is right, and which an empty default
    #: triggered on every message until it was caught.
    dispatched_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), alias="dispatchedAt"
    )


def build_job(
    definition: Definition,
    arguments: dict[str, Any],
    actor: str = "",
    plan: Plan | None = None,
    actor_id: str = "",
) -> tuple[Job, dict[int, str]]:
    """
    Builds the job and the per action run ids it will report against.

    Returns both because the caller has to tell the requester which id belongs to which
    action before any result exists — otherwise the first progress message arrives against
    an identifier nobody has seen.

    The plan is needed for dynamic actions and only for those. A static action's command
    lives on the definition and is rebuilt here; a dynamic one's was written by the model
    during planning and exists nowhere else, so leaving the plan out sends the executor
    whatever the definition happens to hold — which for a dynamic action is nothing.
    """
    values = build_values(definition.inputs, arguments)
    authored = {
        item.action_id: item.authored
        for item in (plan.actions if plan else [])
        if item.authored
    }

    run_id = str(uuid.uuid4())
    action_run_ids: dict[int, str] = {}
    actions: list[dict[str, Any]] = []

    for action in sorted(definition.actions, key=lambda item: item.position):
        # The plan decides what runs, not the definition. A definition's actions are what
        # the tool *can* do; the plan is what this request asked for, and building from
        # the definition made choosing between them pointless — the ones set aside were
        # dispatched anyway.
        if _set_aside(plan, action.id):
            continue

        action_run_id = str(uuid.uuid4())
        action_run_ids[action.id] = action_run_id
        actions.append(_action(action, values, action_run_id, authored.get(action.id, "")))

    job = Job(
        run_id=run_id,
        tool_name=definition.tool_name,
        definition_id=definition.id,
        actor=actor,
        actor_id=actor_id,
        actions=actions,
    )
    return job, action_run_ids


def _set_aside(plan: Plan | None, action_id: int) -> bool:
    """
    Whether the plan considered this action and left it out.

    A missing plan means every action runs, which is what a direct tools/call does and
    what every single-action definition has always done.
    """
    if plan is None:
        return False

    for planned in plan.actions:
        if planned.action_id == action_id:
            return planned.skipped

    # In the definition but not in the plan at all. Nothing decided to run it.
    return True


def _action(
    action: Action, values: dict[str, str], action_run_id: str, authored: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "actionRunId": action_run_id,
        "actionId": action.id,
        "name": action.name,
        "kind": action.kind,
    }

    if action.kind == "ssh":
        payload["ssh"] = _ssh(action, values, _template(action, action.config.command, authored))
    elif action.kind == "db":
        payload["db"] = _db(action, values, _template(action, action.config.query, authored))
    else:
        payload["rest"] = _rest(action, values)

    return payload


def _template(action: Action, stored: str | None, authored: str) -> str:
    """
    Chooses between what the operator saved and what the model wrote.

    Refuses rather than falling through to the empty stored template. Publishing an empty
    command means the executor refuses it a second later, with a job on the queue and a
    failed run recorded for a plan that was perfectly good — which is what happened, and
    read as the plan being wrong rather than the job being built without one.
    """
    if not action.is_dynamic:
        return stored or ""

    if not authored:
        raise IncompleteJob(
            f"Action {action.id} ({action.name}) is dynamic, but no authored "
            f"{'command' if action.kind == 'ssh' else 'query'} came with the plan"
        )

    return authored


def _ssh(action: Action, values: dict[str, str], command: str) -> dict[str, Any]:
    config = action.config

    return {
        # Unmasked, unlike the plan's copy. The guardrails are applied again by the
        # executor, on the machine that is about to run it.
        "command": render(command, values),
        "hosts": [render(host, values) for host in action.hosts],
        "port": config.port or 22,
        "user": config.user or "root",
        "workingDir": config.working_dir or "",
        "strategy": config.strategy or "sequential",
        "concurrency": config.concurrency or 0,
        "batchSize": config.batch_size or 0,
        "stopOnError": bool(config.stop_on_error),
        "allowedCommands": list(config.allowed_commands),
        "blockedPatterns": list(config.blocked_patterns),
        # Whether a model wrote this, so the executor can apply the gate that only makes
        # sense for one that did: an operator's own template may contain a pipe on
        # purpose, and a model's may not.
        "authoredByModel": config.command_mode == "dynamic",
        # Applied by the executor rather than written into the command here. The operator's
        # allowlist is written against the command itself — a prefix of "systemctl" does
        # not match "sudo systemctl" — so the guardrails must see the command the action
        # authorised, and the escalation is added at the point of running it.
        "sudo": bool(config.sudo),
        # Streaming is opt-in and bounded. The window travels with the flag because one is
        # meaningless without the other: a followed command with no end holds an executor
        # slot until the job times out.
        "follow": bool(config.follow),
        "followIdleSeconds": _follow_idle(config),
        "followMaxSeconds": FOLLOW_MAX_SECONDS if config.follow else 0,
        "hostKeys": dict(action.host_keys),
        **_credentials(action, ("privateKey", "passphrase", "password")),
    }


#: How long a followed command may stay quiet before it is stopped, when the action does
#: not say, and the most it may.
#:
#: A minute of silence is a log that has stopped saying anything; the cap on the idle
#: window is there because it is only half a bound — every line printed starts it again.
FOLLOW_IDLE_DEFAULT_SECONDS = 60
FOLLOW_IDLE_MAX_SECONDS = 300

#: The ceiling, whatever the log does.
#:
#: An idle window is not a bound on its own: a log with a line every second resets it
#: forever, and the command holds an executor slot for as long as the machine keeps
#: talking. This is what says "and no longer than", and it is not the operator's to raise
#: from a definition — it is a property of the executor's capacity.
FOLLOW_MAX_SECONDS = 600


def _follow_idle(config: ActionConfig) -> int:
    """The seconds of silence that end a followed command, defaulted and capped."""
    if not config.follow:
        return 0

    requested = config.follow_idle_seconds or FOLLOW_IDLE_DEFAULT_SECONDS
    return max(1, min(requested, FOLLOW_IDLE_MAX_SECONDS))


def _db(action: Action, values: dict[str, str], query: str) -> dict[str, Any]:
    config = action.config

    return {
        "engine": config.engine or "postgres",
        "host": render(config.host, values),
        "port": config.port or 5432,
        "database": config.database or "",
        "user": config.user or "",
        "query": render(query, values),
        "readOnly": bool(config.read_only),
        "maxRows": config.max_rows or 0,
        "allowedOperations": list(config.allowed_operations),
        **_credentials(action, ("password",)),
    }


def _rest(action: Action, values: dict[str, str]) -> dict[str, Any]:
    config = action.config

    return {
        "method": config.method or "GET",
        # The same URL the plan showed, dropped parameters and all. Rendering it twice by
        # two rules would mean approving one request and sending another.
        "url": render_url(config.url, values),
        "headers": {
            header["key"]: render(header.get("value"), values)
            for header in config.headers
            if header.get("key")
        },
        # The same body the plan showed, dropped fields and all.
        "body": render_body(config.body, values) if config.method != "GET" else "",
        "timeoutMs": config.timeout_ms or 0,
    }


def _credentials(action: Action, roles: tuple[str, ...]) -> dict[str, Any]:
    """
    Copies the sealed credentials this kind of action can use.

    Filtered by role rather than passed wholesale: an SSH action has no business carrying a
    database password to an executor, even a sealed one, and narrowing what travels is
    cheaper than reasoning about what happens if it does.
    """
    import base64

    sealed: dict[str, Any] = {}

    for role in roles:
        credential = action.credentials.get(role)
        if credential is None:
            continue

        sealed[role] = {
            # base64 because the executor reads this as JSON, where []byte is expected in
            # exactly that form; sending raw bytes would arrive as mojibake.
            "ciphertext": base64.b64encode(credential.ciphertext).decode(),
            "keyId": credential.key_id,
            "context": credential.context,
        }

    return sealed
