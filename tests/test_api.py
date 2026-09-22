"""
The REST surface the gateway talks to.

Uses the real application, so route wiring, dependency injection and the token check are
all covered; nothing is mocked because nothing here reaches outside the process.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


#: Settings are pinned rather than defaulted. `Settings` also reads `.env`, so a fixture
#: that only overrode what it cared about would inherit whichever queue the developer
#: happens to have configured — and dispatch behaviour would change with the file.
def _settings(**overrides) -> Settings:
    return Settings(
        **{
            "anthropic_api_key": "",
            "publisher_token": "",
            "queue_host": "",
            **overrides,
        }
    )


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(_settings())) as c:
        yield c


@pytest.fixture
def secured_client() -> TestClient:
    with TestClient(create_app(_settings(publisher_token="s3cret"))) as c:
        yield c


@pytest.fixture
def dispatching_client(monkeypatch) -> TestClient:
    """
    A client whose dispatcher has a broker, with the publish itself stubbed.

    The queue transport is exercised for real in mcp-action's own suite, against a real
    RabbitMQ. What is worth checking here is the decision either side of it: that a planned
    plan produces a job with run ids, and that an unplanned one never reaches the broker.
    """
    from app.dispatcher import QueueDispatcher

    published: list = []

    async def capture(self, job):
        published.append(job)

    monkeypatch.setattr(QueueDispatcher, "_publish", capture)

    with TestClient(create_app(_settings(queue_host="broker.test"))) as c:
        c.published = published
        yield c


def publish_body(ssh_definition) -> dict:
    return {"definitions": [ssh_definition.model_dump(mode="json", by_alias=True)]}


class TestCatalogue:
    def test_starts_empty(self, client):
        assert client.get("/health").json()["catalogue"]["tools"] == 0

    def test_publish_makes_a_tool_callable(self, client, ssh_definition):
        published = client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        assert published.status_code == 200
        assert published.json() == {"published": 1, "received": 1}
        assert [t["name"] for t in client.get("/api/v1/tools").json()] == ["apache_fleet"]

    def test_publish_drops_disabled_definitions(self, client, ssh_definition):
        ssh_definition.enabled = False

        response = client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        assert response.json() == {"published": 0, "received": 1}

    def test_publish_replaces_rather_than_merges(self, client, ssh_definition):
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        client.put("/api/v1/catalogue", json={"definitions": []})

        # A definition the gateway stops sending stops being callable.
        assert client.get("/api/v1/tools").json() == []

    def test_schema_is_derived_from_the_inputs(self, client, ssh_definition):
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        schema = client.get("/api/v1/tools").json()[0]["inputSchema"]

        assert schema["required"] == ["service"]
        assert schema["properties"]["service"]["enum"] == ["apache2", "nginx"]
        # A password input must never carry its default into the catalogue.
        assert "default" not in schema["properties"]["ssh_key"]


class TestTheToolScreenHonoursTheSameApproval:
    """
    The direct tool endpoint used to dispatch unconditionally.

    The reasoning was written into its own docstring — that no executor existed, so the
    step could only report as much. An executor arrived; the sentence stayed. What it was
    excusing became real, and on 21 September three clicks on a panel screen that said
    "Nothing is executed" ran `tail -f /var/log/messages` on a live host three times, two
    minutes each, with no approver recorded because there was nobody to record.

    The box was never decorative to the operator who ticked it. It was decorative to this
    endpoint.
    """

    @staticmethod
    def _published(client, ssh_definition, *, approval: bool):
        ssh_definition.actions[0].config.require_approval = approval
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        return client

    def test_an_action_needing_approval_is_not_dispatched(self, client, ssh_definition):
        client = self._published(client, ssh_definition, approval=True)

        body = client.post("/api/v1/executions", json={
            "toolName": "apache_fleet", "arguments": {"service": "nginx"},
        }).json()

        assert body["dispatch"]["status"] == "awaiting_approval"
        assert body["dispatch"].get("run_id") is None

        # The plan still comes back. Seeing what a tool would do is what this screen is
        # for, and withholding it would remove the reason to open it.
        assert body["plan"]["actions"][0]["resolved"] == "systemctl restart nginx"

    def test_sending_back_the_command_shown_dispatches_it(self, client, ssh_definition):
        # Approval is of a command, not of an intention: ask, read what it resolved to,
        # and say yes to that. Two calls and one decision.
        client = self._published(client, ssh_definition, approval=True)

        first = client.post("/api/v1/executions", json={
            "toolName": "apache_fleet", "arguments": {"service": "nginx"},
        }).json()
        shown = first["plan"]["actions"][0]["resolved"]

        second = client.post("/api/v1/executions", json={
            "toolName": "apache_fleet", "arguments": {"service": "nginx"},
            "expect": shown,
        }).json()

        assert second["dispatch"]["status"] != "awaiting_approval"

    def test_approving_something_else_is_refused(self, client, ssh_definition):
        # The guarantee, not a formality. Planning is not deterministic, so agreement to
        # one command must not carry to whatever the next plan happens to say.
        client = self._published(client, ssh_definition, approval=True)

        body = client.post("/api/v1/executions", json={
            "toolName": "apache_fleet", "arguments": {"service": "nginx"},
            "expect": "systemctl restart something-else",
        }).json()

        assert body["dispatch"]["status"] == "refused"
        assert "changed" in body["dispatch"]["reason"].lower()

    def test_approval_cannot_be_given_before_there_is_a_plan(self, client, ssh_definition):
        # The first call has nothing to approve with, and says so rather than inventing a
        # way to agree to a command nobody has seen.
        client = self._published(client, ssh_definition, approval=True)

        body = client.post("/api/v1/executions", json={
            "toolName": "apache_fleet", "arguments": {"service": "nginx"},
        }).json()

        assert body["dispatch"]["status"] == "awaiting_approval"
        assert "approve" in body["dispatch"]["reason"].lower()

    def test_an_action_without_the_box_still_runs(self, client, ssh_definition):
        # The gate is the operator's setting, not a new rule about this endpoint. A tool
        # nobody asked to gate must work from here exactly as it did.
        client = self._published(client, ssh_definition, approval=False)

        body = client.post("/api/v1/executions", json={
            "toolName": "apache_fleet", "arguments": {"service": "nginx"},
        }).json()

        assert body["dispatch"]["status"] != "awaiting_approval"


class TestExecutions:
    def test_plans_a_published_tool(self, client, ssh_definition):
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        response = client.post(
            "/api/v1/executions",
            json={"toolName": "apache_fleet", "arguments": {"service": "nginx"}},
        )

        body = response.json()
        assert body["status"] == "planned"
        assert body["plan"]["actions"][0]["resolved"] == "systemctl restart nginx"

    def test_accepts_an_inline_definition_without_publishing(self, client, ssh_definition):
        response = client.post(
            "/api/v1/executions",
            json={
                "definition": ssh_definition.model_dump(mode="json", by_alias=True),
                "arguments": {},
            },
        )

        assert response.json()["status"] == "planned"

    def test_unknown_tool_is_a_404(self, client):
        response = client.post("/api/v1/executions", json={"toolName": "nope"})

        assert response.status_code == 404

    def test_neither_tool_name_nor_definition_is_rejected(self, client):
        assert client.post("/api/v1/executions", json={}).status_code == 422

    def test_without_a_queue_nothing_is_dispatched(self, client, ssh_definition):
        """Planning only is a deployment choice, and the result says which one is in force."""
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        body = client.post(
            "/api/v1/executions", json={"toolName": "apache_fleet"}
        ).json()

        assert body["dispatch"]["status"] == "skipped"
        assert "No executor" in body["dispatch"]["reason"]

    def test_with_a_queue_a_planned_plan_is_published(self, dispatching_client, ssh_definition):
        dispatching_client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        body = dispatching_client.post(
            "/api/v1/executions", json={"toolName": "apache_fleet"}
        ).json()

        assert body["dispatch"]["status"] == "queued"
        assert body["dispatch"]["run_id"]
        # One run id per action, handed back before any result exists — otherwise the first
        # progress message arrives against an identifier nobody has seen.
        assert body["dispatch"]["action_run_ids"]
        assert len(dispatching_client.published) == 1

    def test_a_rejected_plan_never_reaches_the_broker(self, dispatching_client, ssh_definition):
        ssh_definition.actions[0].config.command = "systemctl restart {{unknown}}"
        dispatching_client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        body = dispatching_client.post(
            "/api/v1/executions", json={"toolName": "apache_fleet"}
        ).json()

        assert body["dispatch"]["status"] == "refused"
        assert dispatching_client.published == []

    def test_a_rejected_plan_is_not_dispatched(self, client, ssh_definition):
        ssh_definition.actions[0].config.command = "systemctl restart {{unknown}}"
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        body = client.post(
            "/api/v1/executions", json={"toolName": "apache_fleet"}
        ).json()

        assert body["status"] == "rejected"
        assert body["dispatch"]["status"] == "refused"


class TestToken:
    def test_gateway_routes_require_the_token_when_configured(self, secured_client):
        assert secured_client.put("/api/v1/catalogue", json={"definitions": []}).status_code == 401

    def test_correct_token_is_accepted(self, secured_client):
        response = secured_client.put(
            "/api/v1/catalogue",
            json={"definitions": []},
            headers={"X-MCP-Token": "s3cret"},
        )

        assert response.status_code == 200

    def test_health_stays_open_and_reports_the_check(self, secured_client):
        body = secured_client.get("/health").json()

        assert body["auth"]["token_required"] is True


class TestPrompts:
    """
    Routing a sentence to a tool.

    The model is replaced throughout: what is worth pinning is the decision either side of
    it — that a hallucinated tool never reaches the planner, that an empty catalogue is
    reported rather than guessed at, and that nothing is dispatched unless asked.
    """

    @staticmethod
    def _route(client, routed):
        client.app.state.prompt_router.route = _resolved(routed)

    def test_a_chosen_tool_is_planned(self, client, ssh_definition):
        from app.router import Routed

        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        self._route(client, Routed(tool_name="apache_fleet", arguments={},
                                   reasoning="restarts apache"))

        body = client.post("/api/v1/prompts", json={"prompt": "apache'yi yeniden başlat"}).json()

        assert body["tool_name"] == "apache_fleet"
        assert body["status"] == "planned"
        assert body["reasoning"]

    def test_nothing_is_dispatched_unless_asked(self, client, ssh_definition):
        from app.router import Routed

        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        self._route(client, Routed(tool_name="apache_fleet", arguments={}))

        body = client.post("/api/v1/prompts", json={"prompt": "restart"}).json()

        # A prompt is a sentence, and a sentence is a poor place to hide "and then do it".
        assert body["dispatch"] is None

    def test_no_match_is_reported_rather_than_guessed(self, client, ssh_definition):
        from app.router import Routed

        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        self._route(client, Routed(problem="No published tool matches this request"))

        body = client.post("/api/v1/prompts", json={"prompt": "bugün hava nasıl"}).json()

        assert body["tool_name"] is None
        assert body["plan"] is None
        assert "matches" in body["problem"]

    def test_an_empty_prompt_is_rejected(self, client):
        assert client.post("/api/v1/prompts", json={"prompt": ""}).status_code == 422


class TestAnActionThatNeedsApprovalDoesNotRunOnItsOwn:
    """
    The panel has had a "needs approval" box since the SSH action existed.

    It was stored, sent here, reported on the plan as `requires_approval` — and then
    nothing stood on it. A prompt with execute ticked dispatched the command regardless,
    which is the one thing the box exists to prevent.
    """

    @staticmethod
    def _client(client, ssh_definition, *, approval: bool):
        from app.router import Routed

        ssh_definition.actions[0].config.require_approval = approval
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        client.app.state.prompt_router.route = _resolved(
            Routed(tool_name="apache_fleet", arguments={"service": "apache2"})
        )
        return client

    def test_nothing_is_dispatched_until_somebody_approves(self, client, ssh_definition):
        client = self._client(client, ssh_definition, approval=True)

        body = client.post("/api/v1/prompts", json={
            "prompt": "apache'yi yeniden baslat", "execute": True,
        }).json()

        assert body["dispatch"]["status"] == "awaiting_approval"

        # The plan is still returned. It is the thing being approved, and a refusal with
        # nothing to look at would leave the operator with no way to say yes.
        assert body["plan"]["actions"][0]["resolved"] == "systemctl restart apache2"

    def test_approving_the_command_in_front_of_you_dispatches_it(self, client, ssh_definition):
        client = self._client(client, ssh_definition, approval=True)

        body = client.post("/api/v1/prompts", json={
            "prompt": "apache'yi yeniden baslat",
            "execute": True,
            "expect": "systemctl restart apache2",
        }).json()

        assert body["dispatch"]["status"] != "awaiting_approval"

    def test_approving_a_different_command_does_not_dispatch_this_one(
        self, client, ssh_definition
    ):
        # The two gates are separate questions: somebody said yes, and yes to *this*.
        client = self._client(client, ssh_definition, approval=True)

        body = client.post("/api/v1/prompts", json={
            "prompt": "apache'yi yeniden baslat",
            "execute": True,
            "expect": "systemctl stop apache2",
        }).json()

        assert body["dispatch"]["status"] == "refused"

    def test_a_rejected_plan_is_refused_rather_than_offered_for_approval(
        self, client, ssh_definition
    ):
        """
        There is nothing to approve. The guardrails refused the command the model wrote,
        so the plan carries none — and "needs approval" under a refusal asks somebody to
        decide about a command that is not there.
        """
        from app.router import Routed

        ssh_definition.actions[0].config.require_approval = True
        ssh_definition.actions[0].config.command_mode = "dynamic"
        ssh_definition.actions[0].config.allowed_commands = ["echo"]
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        client.app.state.prompt_router.route = _resolved(
            Routed(tool_name="apache_fleet", arguments={"service": "apache2"})
        )
        from app.providers import Authored

        client.app.state.planner._call_model = _resolved(  # noqa: SLF001
            Authored(text="echo hi > /tmp/x")
        )

        body = client.post("/api/v1/prompts", json={
            "prompt": "dosya yaz", "execute": True,
        }).json()

        assert body["status"] == "rejected"
        assert body["dispatch"]["status"] != "awaiting_approval"

    def test_an_action_without_the_box_runs_as_before(self, client, ssh_definition):
        client = self._client(client, ssh_definition, approval=False)

        body = client.post("/api/v1/prompts", json={
            "prompt": "apache'yi yeniden baslat", "execute": True,
        }).json()

        assert body["dispatch"]["status"] == "skipped"


class TestAnApprovedCommandIsTheOneThatRuns:
    """
    Planning is not deterministic, and an approval is about one particular command.

    A goal-loop step proposed as `systemctl start httpd` was approved on the strength of
    that command; the approval re-planned the same sentence and the model wrote the install
    command instead. It ran. Nobody had agreed to it — they had agreed to the other one.
    """

    def test_a_plan_that_still_reads_the_same_is_dispatched(self, client, ssh_definition):
        from app.router import Routed

        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        client.app.state.prompt_router.route = _resolved(
            Routed(tool_name="apache_fleet", arguments={"service": "apache2"})
        )

        body = client.post("/api/v1/prompts", json={
            "prompt": "apache'yi yeniden baslat",
            "execute": True,
            "expect": "systemctl restart apache2",
        }).json()

        # "skipped" rather than "queued" because this fixture has no queue behind it. What
        # matters is that it reached the dispatcher at all instead of being refused.
        assert body["dispatch"]["status"] == "skipped"

    def test_a_plan_that_changed_is_not_dispatched(self, client, ssh_definition):
        from app.router import Routed

        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        client.app.state.prompt_router.route = _resolved(
            Routed(tool_name="apache_fleet", arguments={"service": "apache2"})
        )

        body = client.post("/api/v1/prompts", json={
            "prompt": "apache'yi yeniden baslat",
            "execute": True,
            "expect": "systemctl stop apache2",
        }).json()

        assert body["dispatch"]["status"] == "refused"
        assert "approved" in body["dispatch"]["reason"]

        # The plan is still returned: what it now says is the thing to approve next, and a
        # refusal with nothing to look at would leave the goal stuck with no way forward.
        assert body["plan"]["actions"][0]["resolved"] == "systemctl restart apache2"

    def test_the_actions_set_aside_are_not_part_of_the_comparison(self):
        """
        They resolve to nothing — there was no point resolving what is not going to run —
        so comparing them refused every approval of a definition with more than one
        action. Approving a delete came back as "the plan changed since it was approved"
        against a plan that had not changed at all.
        """
        from app.api import _matches
        from app.planner import Plan, PlannedAction

        plan = Plan(tool="u", definition_id=1, actions=[
            PlannedAction(action_id=1, name="Delete", kind="rest", mode="static",
                          resolved="DELETE http://h/users/102"),
            PlannedAction(action_id=2, name="List", kind="rest", mode="static",
                          skipped=True, skip_reason="the request did not ask for this"),
        ])

        assert _matches(plan, ["DELETE http://h/users/102"])

    def test_a_plan_where_everything_was_set_aside_matches_nothing(self):
        # Nothing is going to run, so nothing can be the thing that was agreed to.
        from app.api import _matches
        from app.planner import Plan, PlannedAction

        plan = Plan(tool="u", definition_id=1, actions=[
            PlannedAction(action_id=1, name="Delete", kind="rest", mode="static",
                          skipped=True),
        ])

        assert not _matches(plan, ["DELETE http://h/users/102"])

    def test_an_ordinary_prompt_approves_nothing_in_particular(self, client, ssh_definition):
        # Empty expect is every prompt anybody types. It must not become a gate.
        from app.router import Routed

        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        client.app.state.prompt_router.route = _resolved(
            Routed(tool_name="apache_fleet", arguments={"service": "apache2"})
        )

        body = client.post("/api/v1/prompts", json={
            "prompt": "apache'yi yeniden baslat", "execute": True,
        }).json()

        assert body["dispatch"]["status"] == "skipped"


def _resolved(value):
    """Wraps a value in an awaitable, so a plain object can stand in for the router."""

    async def _call(*_args, **_kwargs):
        return value

    return _call


class TestPromptWithAChosenTool:
    """
    Naming a tool skips choosing one, and nothing else.

    The planning path still runs, so the guardrails still apply: picking your own tool is a
    shortcut past the model, never past the checks.
    """

    def test_a_named_tool_is_used_without_routing(self, client, ssh_definition):
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        # Would raise if it were called, which is the assertion: routing is skipped.
        def _fail(*_args, **_kwargs):
            raise AssertionError("the router chose a tool although one was named")

        client.app.state.prompt_router.route = _fail

        body = client.post(
            "/api/v1/prompts",
            json={"prompt": "restart it", "toolName": "apache_fleet"},
        ).json()

        assert body["tool_name"] == "apache_fleet"
        assert body["status"] == "planned"

    def test_a_tool_with_no_inputs_needs_no_model(self, client, ssh_definition):
        """Asking a model for an empty object costs a round trip and can only fail."""
        ssh_definition.inputs = []
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        def _fail(*_args, **_kwargs):
            raise AssertionError("the model was called for a tool with no inputs")

        client.app.state.prompt_router._backends.for_definition = _fail  # noqa: SLF001

        body = client.post(
            "/api/v1/prompts",
            json={"prompt": "go", "toolName": "apache_fleet"},
        ).json()

        assert body["tool_name"] == "apache_fleet"

    def test_an_unknown_tool_is_a_404(self, client, ssh_definition):
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        response = client.post(
            "/api/v1/prompts", json={"prompt": "x", "toolName": "yok_boyle_bir_sey"}
        )

        assert response.status_code == 404


class TestInputSources:
    """
    Where a parameter is allowed to get its value.

    The type says what a value looks like; the source says who may decide it. Conflating
    the two is how a model ends up inventing a host name.
    """

    @staticmethod
    def _with_sources(ssh_definition, **sources):
        for item in ssh_definition.inputs:
            if item.key in sources:
                item.source = sources[item.key]
        return ssh_definition

    def test_a_fixed_input_is_absent_from_the_published_schema(self, client, ssh_definition):
        """
        Publishing it would invite a client to send a value that is then ignored — worse
        than not offering it, because the client cannot tell the difference.
        """
        first = ssh_definition.inputs[0].key
        client.put("/api/v1/catalogue",
                   json=publish_body(self._with_sources(ssh_definition, **{first: "fixed"})))

        schema = client.get("/api/v1/tools").json()[0]["inputSchema"]

        assert first not in schema["properties"]
        assert first not in schema["required"]

    def test_a_fixed_input_cannot_be_overridden(self, client, ssh_definition):
        first = ssh_definition.inputs[0]
        first.source = "fixed"
        first.default_value = "sabit-deger"
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))

        body = client.post(
            "/api/v1/executions",
            json={"toolName": "apache_fleet", "arguments": {first.key: "ezmeye-calis"}},
        ).json()

        rendered = body["plan"]["actions"][0]["resolved"]
        assert "ezmeye-calis" not in rendered
        if "{{" not in rendered:
            assert "sabit-deger" in rendered

    def test_a_caller_input_is_published_but_hidden_from_the_model(self, client, ssh_definition):
        """
        Still part of the contract — an MCP client may supply it — but not offered to the
        model, which is the whole point: it may read a date out of a sentence, not a host.
        """
        first = ssh_definition.inputs[0].key
        client.put("/api/v1/catalogue",
                   json=publish_body(self._with_sources(ssh_definition, **{first: "caller"})))

        published = client.get("/api/v1/tools").json()[0]["inputSchema"]
        assert first in published["properties"]

        definition = client.app.state.catalogue.find("apache_fleet")
        assert first not in definition.prompt_schema()["properties"]

    def test_prompt_is_the_default_so_existing_definitions_are_unchanged(self, ssh_definition):
        assert all(item.source == "prompt" for item in ssh_definition.inputs)


class TestSealedCredentials:
    """
    Credentials survive the trip from the gateway to the executor.

    The gateway is a Spring service, so a `byte[]` arrives base64 encoded. Pydantic does
    not decode base64 into `bytes` — it encodes the string as UTF-8 — so the value became
    the ASCII of its own base64, and re-encoding it for the executor produced something
    mcp-cipher could only reject. The failure surfaced as "the credential could not be
    opened", three services away from the cause.
    """

    def test_base64_on_the_wire_becomes_real_bytes(self):
        import base64

        from app.models import SealedSecret

        sealed = b"\x01\x02\x03 not text"

        parsed = SealedSecret.model_validate(
            {"ciphertext": base64.b64encode(sealed).decode(), "keyId": "v1",
             "context": "password"}
        )

        assert parsed.ciphertext == sealed

    def test_the_job_carries_the_same_bytes_the_gateway_sealed(self, ssh_definition):
        import base64

        from app.jobs import build_job
        from app.models import SealedSecret

        sealed = b"\x9f\x8e\x00 binary"
        ssh_definition.actions[0].credentials = {
            "privateKey": SealedSecret.model_validate(
                {"ciphertext": base64.b64encode(sealed).decode(), "keyId": "v1",
                 "context": "ssh_private_key"}
            )
        }

        job, _ = build_job(ssh_definition, {}, "tester")
        carried = job.actions[0]["ssh"]["privateKey"]["ciphertext"]

        # Base64 again on the way out, because the executor reads JSON where []byte is
        # expected in exactly that form — but of the original bytes, not of their encoding.
        assert base64.b64decode(carried) == sealed


class TestSudoReachesTheExecutor:
    """
    The panel has had a "Run the command with sudo" box since the SSH action existed. It
    was stored, sent here and dropped: nothing put it in the job, so every command ran as
    the connecting user and an install that needed root failed with a permission error
    that read like a problem with the server.
    """

    def test_the_job_says_the_command_needs_root(self, ssh_definition):
        from app.jobs import build_job

        job, _ = build_job(ssh_definition, {"service": "apache2"}, "tester")

        assert job.actions[0]["ssh"]["sudo"] is True

    def test_the_command_itself_is_not_prefixed_here(self, ssh_definition):
        """
        The escalation is the executor's to apply, immediately before running.

        The allowlist an action declares is written against the command itself: a prefix of
        "systemctl" does not match "sudo systemctl", so a command prefixed before the
        guardrails would be refused by the very list that authorised it.
        """
        from app.jobs import build_job

        job, _ = build_job(ssh_definition, {"service": "apache2"}, "tester")

        assert job.actions[0]["ssh"]["command"] == "systemctl restart apache2"

    def test_an_action_without_the_box_ticked_says_so(self, ssh_definition):
        from app.jobs import build_job

        ssh_definition.actions[0].config.sudo = False
        job, _ = build_job(ssh_definition, {"service": "apache2"}, "tester")

        assert job.actions[0]["ssh"]["sudo"] is False


class TestAnUnfilledFilterIsNotAMissingInput:
    """
    A definition offering four optional search filters could not be used at all.

    `?search={{search}}&first_name={{first_name}}&last_name={{last_name}}&email={{email}}`
    with none of them supplied was refused as "Template refers to undefined input(s)".
    They were defined — declared, optional, with empty defaults — and `build_values` had
    simply not put them in, because there was nothing to put.

    A query string is a list of filters, and a filter with no value is not an empty filter.
    It is one nobody asked for.
    """

    @staticmethod
    def _definition():
        from app.models import Definition

        return Definition.model_validate({
            "id": 1, "name": "Users", "toolName": "users",
            "inputs": [
                {"key": "search", "type": "text", "required": False, "defaultValue": ""},
                {"key": "email", "type": "text", "required": False, "defaultValue": ""},
                {"key": "id", "type": "text", "required": False, "defaultValue": ""},
            ],
            "actions": [{
                "id": 1, "kind": "rest", "name": "List", "config": {
                    "method": "GET",
                    "url": "http://h/api/users?search={{search}}&email={{email}}",
                },
            }],
        })

    def test_unfilled_filters_drop_out_rather_than_refusing_the_plan(self, client):
        from app.jobs import build_job

        job, _ = build_job(self._definition(), {}, "tester")

        assert job.actions[0]["rest"]["url"] == "http://h/api/users"

    def test_a_filter_that_was_given_is_kept(self, client):
        from app.jobs import build_job

        job, _ = build_job(self._definition(), {"email": "a@b.c"}, "tester")

        assert job.actions[0]["rest"]["url"] == "http://h/api/users?email=a@b.c"

    def test_a_missing_path_placeholder_is_still_a_problem(self, client):
        """
        Dropping it would silently turn "delete user 13" into "delete users", which is a
        different request against a different resource.
        """
        from app.templating import render_url

        assert render_url("http://h/api/users/{{id}}", {}) == "http://h/api/users/{{id}}"

    def test_a_json_field_nobody_filled_is_dropped_too(self):
        """
        The same problem in the body. A PUT changing one field should send that field, not
        the other three set to a placeholder nobody replaced.
        """
        from app.templating import render_body

        body = '{"first_name":"{{first_name}}","email":"{{email}}"}'

        assert render_body(body, {"email": "a@b.c"}) == '{\n  "email": "a@b.c"\n}'
        assert render_body(body, {}) == "{}"

    def test_a_body_that_is_not_json_is_left_entirely_alone(self):
        # This cannot tell a field from a line of text, and guessing would be editing
        # somebody's payload on a hunch.
        from app.templating import render_body

        assert render_body("name={{name}}", {}) == "name={{name}}"

    def test_a_parameter_with_a_value_beside_the_placeholder_is_left_alone(self):
        # Half a value is not a missing one, and a literal parameter is nobody's filter.
        from app.templating import render_url

        assert render_url("http://h/u?q=user-{{name}}&limit=50", {}) == (
            "http://h/u?q=user-{{name}}&limit=50"
        )


class TestTwoProblemsNoLongerShareOneSentence:
    """
    "Template refers to undefined input(s): email" was said about an input plainly listed
    in the editor, which sends the reader looking for a typo that is not there.

    A placeholder nobody declared is a mistake in the definition, fixed by adding an
    input. A declared one with no value is a mistake in the request — or in nothing at
    all, if it was optional. They need different actions and now they read differently.
    """

    @staticmethod
    async def _plan(client, url, inputs):
        from app.models import Definition
        from app.planner import Planner
        from app.config import Settings

        definition = Definition.model_validate({
            "id": 1, "name": "U", "toolName": "u", "inputs": inputs,
            "actions": [{"id": 1, "kind": "rest", "name": "a",
                         "config": {"method": "GET", "url": url}}],
        })

        return await Planner(Settings()).plan(definition, {}, "listele")

    @pytest.mark.asyncio
    async def test_a_placeholder_nobody_declared_is_undefined(self, client):
        plan = await self._plan(client, "http://h/api/users/{{nosuch}}", [])

        assert any("undefined input" in problem for problem in plan.problems)

    @pytest.mark.asyncio
    async def test_a_declared_placeholder_with_no_value_says_so_instead(self, client):
        plan = await self._plan(
            client, "http://h/api/users/{{id}}",
            [{"key": "id", "type": "text", "required": False, "defaultValue": ""}],
        )

        assert any("No value for input" in problem for problem in plan.problems)
        assert not any("undefined input" in problem for problem in plan.problems)


class TestOnlyTheActionsAskedForRun:
    """
    "Find the users called mehmet" also posted, updated and deleted one.

    A definition's actions used to be a sequence: all of them ran, in order, because that
    is what a workflow means. But a definition is also how somebody groups the operations
    on one resource — list, create, update, delete against the same users API — and there
    the actions are alternatives, not steps.

    So the request decides. What it did not ask for is set aside, kept in the plan so it
    can be seen, and never built into the job.
    """

    @staticmethod
    def _definition():
        from app.models import Definition

        return Definition.model_validate({
            "id": 1, "name": "Users", "toolName": "users",
            "inputs": [
                {"key": "id", "type": "text", "required": True, "defaultValue": ""},
                {"key": "name", "type": "text", "required": False, "defaultValue": ""},
            ],
            "actions": [
                {"id": 1, "kind": "rest", "name": "List", "position": 0,
                 "config": {"method": "GET", "url": "http://h/users?name={{name}}"}},
                {"id": 2, "kind": "rest", "name": "Delete", "position": 1,
                 "config": {"method": "DELETE", "url": "http://h/users/{{id}}"}},
            ],
        })

    @staticmethod
    def _planner(chosen_ids, reason="asked for a list"):
        from app.config import Settings
        from app.planner import Planner, _Chosen

        planner = Planner(Settings())

        async def choose(request, definition, history=()):
            return _Chosen(actions=chosen_ids, reason=reason)

        planner._choose_actions = choose  # noqa: SLF001
        return planner

    @pytest.mark.asyncio
    async def test_the_choice_is_told_what_was_asked_before(self):
        """
        A step the goal loop wrote does not always carry the verb. Asked to "find the Yigits
        and delete them", its second step read "Processing the second user found with
        first_name 'Yigit'" — a record and no operation — and was answered with the listing
        action. The delete never came, and from the console the goal had stopped after one.
        """
        from app.config import Settings
        from app.planner import Planner, _Chosen

        seen = {}

        async def choose(request, definition, history=()):
            seen["history"] = list(history)
            return _Chosen(actions=[2], reason="the goal is to delete them")

        planner = Planner(Settings())
        planner._choose_actions = choose  # noqa: SLF001

        class _Turn:
            def __init__(self, prompt):
                self.prompt = prompt

        await planner.plan(
            self._definition(), {"id": "86"},
            "Processing the second user found with first_name 'Yigit'.",
            [_Turn("ismi Yigit olanlari bulup siler misin?")],
        )

        assert [turn.prompt for turn in seen["history"]] == [
            "ismi Yigit olanlari bulup siler misin?"
        ]

    def test_what_was_asked_before_is_the_questions_oldest_first(self):
        # The goal is the one that says "and delete them", so it must not be buried under
        # the steps it produced.
        from app.planner import _asked_before

        class _Turn:
            def __init__(self, prompt):
                self.prompt = prompt
                self.statement = "GET http://h/users"

        text = _asked_before([_Turn("bul ve sil"), _Turn("ilkini isle")])

        assert text.index("bul ve sil") < text.index("ilkini isle")
        assert "GET http://h/users" not in text

    def test_nothing_asked_before_adds_nothing(self):
        from app.planner import _asked_before

        assert _asked_before([]) == ""

    @pytest.mark.asyncio
    async def test_the_others_are_set_aside_rather_than_run(self):
        plan = await self._planner([1]).plan(
            self._definition(), {"name": "mehmet"}, "ismi mehmet olanlari getir"
        )

        ran = [a for a in plan.actions if not a.skipped]
        aside = [a for a in plan.actions if a.skipped]

        assert [a.action_id for a in ran] == [1]
        assert [a.action_id for a in aside] == [2]
        assert aside[0].skip_reason

    @pytest.mark.asyncio
    async def test_a_set_aside_action_is_not_built_into_the_job(self):
        # The point of the whole thing. Building from the definition rather than the plan
        # dispatched them anyway, which made choosing pointless.
        from app.jobs import build_job

        definition = self._definition()
        plan = await self._planner([1]).plan(
            definition, {"name": "mehmet"}, "ismi mehmet olanlari getir"
        )

        job, ids = build_job(definition, {"name": "mehmet"}, "tester", plan)

        assert len(job.actions) == 1
        assert job.actions[0]["rest"]["url"] == "http://h/users?name=mehmet"

    @pytest.mark.asyncio
    async def test_required_is_asked_of_the_actions_being_run(self):
        """
        `id` is required by the delete and meaningless to the list. Asking for it whatever
        runs is what made a four-action definition impossible to declare honestly: every
        input had to be optional, and then nothing was ever checked.
        """
        plan = await self._planner([1]).plan(
            self._definition(), {"name": "mehmet"}, "listele"
        )

        assert plan.status != "incomplete"

    @pytest.mark.asyncio
    async def test_and_still_demanded_when_that_action_runs(self):
        plan = await self._planner([2]).plan(self._definition(), {}, "sil")

        assert plan.status == "incomplete"
        assert any("id" in problem for problem in plan.problems)

    @pytest.mark.asyncio
    async def test_an_action_waiting_on_an_earlier_one_is_deferred_not_refused(self):
        """
        "Find the user called Mehmet Bulut and delete them" needs both actions, and the
        second cannot run yet: the id it deletes by is in the first one's answer. Asked to
        choose, a model picks both — reasonably, the request does ask for both — and the
        plan was refused whole for an id nobody could have supplied.
        """
        plan = await self._planner([1, 2]).plan(
            self._definition(), {"name": "mehmet"}, "mehmet'i bul ve sil"
        )

        ran = [a for a in plan.actions if not a.skipped]
        held = [a for a in plan.actions if a.skipped]

        assert [a.action_id for a in ran] == [1]
        assert [a.action_id for a in held] == [2]
        assert "waiting on id" in held[0].skip_reason
        assert plan.status == "planned"

    @pytest.mark.asyncio
    async def test_an_action_with_nothing_before_it_is_still_refused(self):
        # Nothing can answer for it later either. A delete with no target is a request
        # with no target, and saying so is the right answer.
        plan = await self._planner([2]).plan(self._definition(), {}, "sil")

        assert plan.status == "incomplete"
        assert any("id" in problem for problem in plan.problems)

    @pytest.mark.asyncio
    async def test_an_action_named_by_the_caller_is_not_chosen_again(self):
        """
        A goal-loop step taking up a deferred action knows which one was waiting — the plan
        recorded it. Asking again spends a model call to be told something already written
        down, and gives it a chance to answer differently: handed "delete the user with id
        59", it once picked the search instead.

        It also costs a request the rate limit noticed. Two extra calls per step is how a
        goal of five deletes became fifteen model calls.
        """
        from app.config import Settings
        from app.planner import Planner

        planner = Planner(Settings())
        asked = False

        async def choose(request, definition, history=()):
            nonlocal asked
            asked = True
            return None

        planner._choose_actions = choose  # noqa: SLF001

        plan = await planner.plan(
            self._definition(), {"id": "59"}, "sil", action_id=2
        )

        assert asked is False
        assert [a.action_id for a in plan.actions if not a.skipped] == [2]

    @pytest.mark.asyncio
    async def test_values_the_caller_supplied_win_over_the_sentence(self):
        """
        A goal-loop step carries the value it read out of the previous answer. Routing had
        to find it in a sentence before, and a sentence written as narration kept it out of
        reach — so the caller's own values are used, and they win.
        """
        from app.jobs import build_job

        definition = self._definition()
        plan = await self._planner([2]).plan(definition, {"id": "59"}, "sil", action_id=2)

        job, _ = build_job(definition, {"id": "59"}, "tester", plan)

        assert job.actions[0]["rest"]["url"] == "http://h/users/59"

    @pytest.mark.asyncio
    async def test_an_action_that_is_not_the_tools_is_refused(self):
        # A stale id, or a definition edited between the plan and the step.
        from app.config import Settings
        from app.planner import Planner

        plan = await Planner(Settings()).plan(
            self._definition(), {}, "sil", action_id=999
        )

        assert plan.status == "incomplete"
        assert any("not part of this tool" in problem for problem in plan.problems)

    @pytest.mark.asyncio
    async def test_a_choice_that_could_not_be_made_runs_nothing(self):
        # A model that failed to answer is not permission to run four writes.
        from app.config import Settings
        from app.planner import Planner

        planner = Planner(Settings())

        async def choose(request, definition, history=()):
            return None

        planner._choose_actions = choose  # noqa: SLF001

        plan = await planner.plan(self._definition(), {"id": "1"}, "bir sey")

        assert plan.status == "incomplete"
        assert plan.actions == []

    @pytest.mark.asyncio
    async def test_a_tools_call_with_no_sentence_cannot_choose(self):
        """
        Arguments and no prose. There is no way to tell which of four operations was
        meant, and guessing all of them is how a search also deleted something.
        """
        from app.config import Settings
        from app.planner import Planner

        plan = await Planner(Settings()).plan(self._definition(), {"id": "1"}, "")

        assert plan.status == "incomplete"
        assert plan.actions == []

    @pytest.mark.asyncio
    async def test_one_action_is_never_a_choice(self):
        # Asking a model about it would spend a call to be told the only answer.
        from app.config import Settings
        from app.models import Definition
        from app.planner import Planner

        single = Definition.model_validate({
            "id": 1, "name": "U", "toolName": "u", "inputs": [],
            "actions": [{"id": 1, "kind": "rest", "name": "List", "config": {
                "method": "GET", "url": "http://h/users"}}],
        })

        plan = await Planner(Settings()).plan(single, {}, "")

        assert [a.action_id for a in plan.actions] == [1]
        assert not plan.actions[0].skipped


class TestNobodyTypedThisOne:
    """
    A goal that found a user and then deleted them ran the delete unattended.

    Approval was decided from the action that had just *finished* — a search, a read — and
    the one about to run is not known until it has been planned. So a read was taken as
    licence for whatever came next, and what came next was a DELETE.

    The decision belongs where the plan is known.
    """

    @staticmethod
    def _client(client, ssh_definition, method):
        from app.router import Routed

        ssh_definition.actions[0] = ssh_definition.actions[0].model_copy(update={
            "kind": "rest",
            "config": ssh_definition.actions[0].config.model_copy(update={
                "method": method, "url": "http://h/users/1", "require_approval": None,
            }),
        })
        client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
        client.app.state.prompt_router.route = _resolved(
            Routed(tool_name="apache_fleet", arguments={})
        )
        return client

    def test_a_write_nobody_typed_is_held(self, client, ssh_definition):
        client = self._client(client, ssh_definition, "DELETE")

        body = client.post("/api/v1/prompts", json={
            "prompt": "sil", "execute": True, "unattended": True,
        }).json()

        assert body["dispatch"]["status"] == "awaiting_approval"

    def test_a_read_nobody_typed_still_runs(self, client, ssh_definition):
        # Which is what makes a goal of queries worth looping at all.
        client = self._client(client, ssh_definition, "GET")

        body = client.post("/api/v1/prompts", json={
            "prompt": "getir", "execute": True, "unattended": True,
        }).json()

        assert body["dispatch"]["status"] != "awaiting_approval"

    def test_a_write_somebody_typed_is_not_held_by_this_rule(self, client, ssh_definition):
        """
        They asked for it, and the action does not say a person has to see it first. This
        rule is about steps nobody wrote, not about writes in general.
        """
        client = self._client(client, ssh_definition, "DELETE")

        body = client.post("/api/v1/prompts", json={
            "prompt": "sil", "execute": True,
        }).json()

        assert body["dispatch"]["status"] != "awaiting_approval"

    class TestAndNobodyGaveItAValueEither:
        """
        The step the loop writes carries no values in its words, and routing was inventing
        them anyway.

        Asked to act on "the first user found with first_name 'Yigit'" — a sentence with no
        id in it — this model produced `id: 1` and explained that an earlier search had
        returned a user with that id. It had not: the search returned 59, 69, 86 and 97.
        The step reached approval as a DELETE against a record nobody had mentioned.

        Routing reads a sentence. When a person wrote the sentence that is the whole point;
        when the loop wrote it there is nothing in there to read, and an answer invented to
        fill the gap is worse than an empty field.
        """

        @staticmethod
        def _client(client, ssh_definition):
            from app.router import Routed

            ssh_definition.actions[0] = ssh_definition.actions[0].model_copy(update={
                "kind": "rest",
                "config": ssh_definition.actions[0].config.model_copy(update={
                    "method": "GET", "url": "http://h/users/{{id}}",
                    "require_approval": None,
                }),
            })
            ssh_definition = ssh_definition.model_copy(update={
                "inputs": [type(ssh_definition.inputs[0]).model_validate(
                    {"key": "id", "type": "text", "required": False}
                )],
            })
            client.put("/api/v1/catalogue", json=publish_body(ssh_definition))
            client.app.state.prompt_router.route = _resolved(
                Routed(tool_name="apache_fleet", arguments={"id": "1"})
            )
            return client

        def test_routing_does_not_get_to_fill_a_step_nobody_typed(
            self, client, ssh_definition
        ):
            client = self._client(client, ssh_definition)

            body = client.post("/api/v1/prompts", json={
                "prompt": "Processing the first user found with first_name 'Yigit'.",
                "execute": True, "unattended": True,
            }).json()

            assert "/users/1" not in body["plan"]["actions"][0]["resolved"]

        def test_what_the_loop_read_is_used(self, client, ssh_definition):
            # The values it took out of the answers are the point; only routing is doubted.
            client = self._client(client, ssh_definition)

            body = client.post("/api/v1/prompts", json={
                "prompt": "Processing the first user found with first_name 'Yigit'.",
                "execute": True, "unattended": True, "arguments": {"id": "59"},
            }).json()

            assert "/users/59" in body["plan"]["actions"][0]["resolved"]

        def test_the_answer_reports_what_the_plan_ran_on(self, client, ssh_definition):
            """
            Not what routing found. The two were the same until a step could arrive with
            values already read out of an answer; after that, reporting the sentence's
            version described a plan that had used something else — and the gateway, which
            keeps this to re-plan an approval from, kept the wrong half.
            """
            client = self._client(client, ssh_definition)

            body = client.post("/api/v1/prompts", json={
                "prompt": "Processing the first user found with first_name 'Yigit'.",
                "execute": True, "unattended": True, "arguments": {"id": "59"},
            }).json()

            assert body["arguments"] == {"id": "59"}

        def test_routing_still_reads_a_sentence_a_person_wrote(
            self, client, ssh_definition
        ):
            # "delete user 1" has the 1 in it. Reading that out is what routing is for.
            client = self._client(client, ssh_definition)

            body = client.post("/api/v1/prompts", json={
                "prompt": "1 numarali kullaniciyi getir", "execute": True,
            }).json()

            assert "/users/1" in body["plan"]["actions"][0]["resolved"]


class TestFollowingIsOptionalAndBounded:
    """
    `tail -f` used to hold the executor for ten minutes and come back with nothing.

    The command never ends, so the session ran until the job's own timeout, buffering
    output nobody could see, with the single worker blocked the whole time. Following makes
    the output arrive as it is written; the windows make the command end.

    Two windows, because one is not enough. The idle one is what a person means by watching
    a log — until it goes quiet — and every line printed starts it again. That alone is no
    bound at all on a log with a line every second, which is what the ceiling is for.
    """

    def test_an_action_that_does_not_follow_says_so(self, ssh_definition):
        from app.jobs import build_job

        job, _ = build_job(ssh_definition, {"service": "apache2"}, "tester")

        assert job.actions[0]["ssh"]["follow"] is False
        assert job.actions[0]["ssh"]["followIdleSeconds"] == 0
        assert job.actions[0]["ssh"]["followMaxSeconds"] == 0

    def test_a_followed_action_gets_both_windows_even_without_asking(self, ssh_definition):
        # A followed command with no end is the thing this exists to prevent.
        from app.jobs import FOLLOW_IDLE_DEFAULT_SECONDS, FOLLOW_MAX_SECONDS, build_job

        ssh_definition.actions[0].config.follow = True

        job, _ = build_job(ssh_definition, {"service": "apache2"}, "tester")

        assert job.actions[0]["ssh"]["follow"] is True
        assert job.actions[0]["ssh"]["followIdleSeconds"] == FOLLOW_IDLE_DEFAULT_SECONDS
        assert job.actions[0]["ssh"]["followMaxSeconds"] == FOLLOW_MAX_SECONDS

    def test_the_idle_window_is_capped(self, ssh_definition):
        from app.jobs import FOLLOW_IDLE_MAX_SECONDS, build_job

        ssh_definition.actions[0].config.follow = True
        ssh_definition.actions[0].config.follow_idle_seconds = 86_400

        job, _ = build_job(ssh_definition, {"service": "apache2"}, "tester")

        assert job.actions[0]["ssh"]["followIdleSeconds"] == FOLLOW_IDLE_MAX_SECONDS

    def test_an_idle_window_the_operator_chose_is_kept(self, ssh_definition):
        from app.jobs import build_job

        ssh_definition.actions[0].config.follow = True
        ssh_definition.actions[0].config.follow_idle_seconds = 30

        job, _ = build_job(ssh_definition, {"service": "apache2"}, "tester")

        assert job.actions[0]["ssh"]["followIdleSeconds"] == 30

    def test_the_ceiling_is_not_the_definitions_to_raise(self, ssh_definition):
        """
        It is a property of the executor's capacity, not of what this log is worth.

        A followed command holds a worker for its whole duration, and an action asking for
        a day would quietly take the executor out of service for one.
        """
        from app.jobs import FOLLOW_MAX_SECONDS, build_job

        ssh_definition.actions[0].config.follow = True
        ssh_definition.actions[0].config.follow_idle_seconds = 300

        job, _ = build_job(ssh_definition, {"service": "apache2"}, "tester")

        assert job.actions[0]["ssh"]["followMaxSeconds"] == FOLLOW_MAX_SECONDS


class TestDynamicJobsCarryTheAuthoredStatement:
    """
    A dynamic action's query lives on the plan and nowhere else.

    Building the job without it sent the executor the definition's stored query, which for
    a dynamic action is empty — the executor refused it, and the run was recorded as failed
    against a plan that had passed every guardrail.
    """

    def test_the_job_uses_the_query_the_model_wrote(self, dynamic_db_definition):
        from app.jobs import build_job
        from app.planner import Plan, PlannedAction

        action = dynamic_db_definition.actions[0]
        plan = Plan(
            tool=dynamic_db_definition.tool_name,
            definition_id=dynamic_db_definition.id,
            actions=[
                PlannedAction(
                    action_id=action.id, name=action.name, kind="db", mode="dynamic",
                    authored_by_model=True,
                    authored="SELECT count(*) FROM runs WHERE created_at >= '{{since}}'",
                    resolved="SELECT count(*) FROM runs WHERE created_at >= '2026-01-01'",
                )
            ],
        )

        job, _ = build_job(
            dynamic_db_definition, {"since": "2026-01-01"}, "tester", plan
        )

        # Rendered here, not on the plan: the placeholder is substituted on the way to the
        # executor exactly as a static action's would be.
        assert job.actions[0]["db"]["query"] == (
            "SELECT count(*) FROM runs WHERE created_at >= '2026-01-01'"
        )

    def test_a_dynamic_action_without_a_plan_is_refused_not_emptied(
        self, dynamic_db_definition
    ):
        from app.jobs import IncompleteJob, build_job

        with pytest.raises(IncompleteJob) as raised:
            build_job(dynamic_db_definition, {"since": "2026-01-01"}, "tester")

        assert "dynamic" in str(raised.value)

    def test_the_authored_statement_stays_out_of_the_response(self):
        from app.planner import PlannedAction

        planned = PlannedAction(
            action_id=1, name="Report", kind="db", mode="dynamic",
            authored="SELECT secret FROM vault",
            resolved="SELECT secret FROM vault",
        )

        # Excluded from serialisation: the plan is what a browser sees, and the authored
        # text is the dispatcher's business only.
        assert "authored" not in planned.model_dump(by_alias=True)
        assert "SELECT secret FROM vault" not in planned.model_dump_json().replace(
            '"resolved":"SELECT secret FROM vault"', ""
        )


class TestSeveralCommandsApprovedAtOnce:
    """
    A plan whose commands all resolve from the one sentence can be shown whole.

    "Write this script to /tmp and run it" is two commands and one decision. Asking twice
    spends a second model call and a second wait on a decision already made — and the
    person is reading the same card again.

    What may not be batched is a plan holding an action *waiting* on an earlier answer.
    That command does not exist yet, so there is nothing to put on the screen, and
    approving what you have not seen is what this path exists to prevent. Which plans
    qualify is the caller's decision; what is guaranteed here is that whatever runs is
    what was shown.
    """

    @staticmethod
    def _plan(*commands: str):
        from app.planner import Plan, PlannedAction

        return Plan(tool="t", definition_id=1, actions=[
            PlannedAction(action_id=index, name=f"a{index}", kind="ssh", mode="static",
                          resolved=command)
            for index, command in enumerate(commands, start=1)
        ])

    def test_both_commands_approved_and_both_planned(self):
        from app.api import _matches

        assert _matches(
            self._plan("cat > /tmp/f.py <<'E'\nprint(1)\nE", "python3 /tmp/f.py"),
            ["cat > /tmp/f.py <<'E'\nprint(1)\nE", "python3 /tmp/f.py"],
        )

    def test_the_order_they_are_listed_in_does_not_matter(self):
        """A plan is ordered by position; the screen is a list. Neither is a promise."""
        from app.api import _matches

        assert _matches(self._plan("one", "two"), ["two", "one"])

    def test_a_third_command_appearing_is_refused(self):
        """
        The reason this is equality and not containment.

        Planning is not deterministic. A re-plan that comes back holding an action nobody
        was shown must be refused, and a subset test would wave it through — which is
        exactly how an approval of a search could end up running a delete.
        """
        from app.api import _matches

        assert not _matches(self._plan("one", "two", "three"), ["one", "two"])

    def test_a_command_that_changed_is_refused(self):
        from app.api import _matches

        assert not _matches(self._plan("one", "two"), ["one", "different"])

    def test_one_of_the_approved_commands_going_missing_is_refused(self):
        """
        Fewer is as wrong as more. Somebody who agreed to "write it and run it" did not
        agree to "write it", and a plan that quietly dropped the second half would leave
        them believing something ran that never did.
        """
        from app.api import _matches

        assert not _matches(self._plan("one"), ["one", "two"])

    def test_the_same_command_twice_is_two_commands(self):
        """
        Sorted rather than set-compared. Collapsing the duplicate would let a plan run
        something twice on the strength of its having been shown once.
        """
        from app.api import _matches

        assert not _matches(self._plan("one", "one"), ["one"])

    def test_nothing_approved_still_matches_anything(self):
        """An ordinary prompt approves nothing in particular and is not being checked."""
        from app.api import _matches

        assert _matches(self._plan("one", "two"), [])


class TestNamingSeveralActions:
    """Approving names the actions, rather than letting the choice run again."""

    def test_only_the_named_actions_are_chosen(self):
        from app.models import Action, ActionConfig, Definition
        from app.planner import Plan, Planner

        def action(identifier: int) -> Action:
            return Action(id=identifier, kind="ssh", name=f"a{identifier}",
                          position=identifier,
                          config=ActionConfig(commandMode="static", command="echo hi"))

        definition = Definition(id=1, name="t", toolName="t",
                                actions=[action(1), action(2), action(3)])
        plan = Plan(tool="t", definition_id=1)

        import asyncio
        wanted, left = asyncio.run(Planner.__new__(Planner)._choose(
            "", definition, definition.actions, plan, {}, None, [1, 3]))

        assert [item.id for item in wanted] == [1, 3]
        assert [item.action_id for item in left] == [2]
        assert left[0].skip_reason == "the step is for another action"

    def test_an_action_that_is_not_part_of_the_tool_is_a_problem(self):
        from app.models import Action, ActionConfig, Definition
        from app.planner import Plan, Planner

        definition = Definition(id=1, name="t", toolName="t", actions=[
            Action(id=1, kind="ssh", name="a", position=0,
                   config=ActionConfig(commandMode="static", command="echo hi")),
            Action(id=2, kind="ssh", name="b", position=1,
                   config=ActionConfig(commandMode="static", command="echo hi")),
        ])
        plan = Plan(tool="t", definition_id=1)

        import asyncio
        wanted, _ = asyncio.run(Planner.__new__(Planner)._choose(
            "", definition, definition.actions, plan, {}, None, [99]))

        assert wanted == []
        assert "is not part of this tool" in plan.problems[0]


class TestRoutingOnlyOffersWhatTheCallerMayRun:
    """
    The restriction is enforced by not offering the tool, not by refusing the choice.

    On the gateway's execute path, planning and dispatch happen inside one call: by the
    time a tool name comes back, the job may already be on the broker. Refusing then would
    refuse work that had already started. A tool that is never offered cannot be chosen,
    which is the only form of this check that runs before the work does.
    """

    @staticmethod
    def _catalogue(*names: str):
        from app.catalogue import Catalogue
        from app.models import Definition

        catalogue = Catalogue()
        catalogue.replace([
            Definition(id=index, name=name, toolName=name)
            for index, name in enumerate(names, start=1)
        ])
        return catalogue

    def _router(self, *names: str):
        from app.config import Settings
        from app.router import PromptRouter

        class NoBackend:
            """Answers that no model is configured, rather than not being there at all."""

            @staticmethod
            def for_definition(*_args):
                return None

            @staticmethod
            def unavailable_reason(*_args):
                return "no model configured"

        router = PromptRouter.__new__(PromptRouter)
        router._catalogue = self._catalogue(*names)
        router._settings = Settings()
        router._backends = NoBackend()
        return router

    async def _problem(self, router, allowed):
        routed = await router.route("bir sey yap", allowed=allowed)
        return routed.problem

    def test_a_caller_with_nothing_allowed_is_told_the_catalogue_is_empty(self):
        """
        Which is true from where they stand, and says nothing about what exists.

        Listing the tools they may not reach would turn a refusal into an inventory.
        """
        import asyncio

        problem = asyncio.run(self._problem(self._router("a", "b"), ["c"]))

        assert "catalogue is empty" in problem

    def test_an_empty_allowance_means_no_restriction(self):
        """
        Not "may run nothing". An administrator is sent no list at all, and a list that
        meant both would offer them an empty catalogue.
        """
        import asyncio

        # Gets past the catalogue check and fails at the backend instead, which is how we
        # know the tools survived the narrowing.
        problem = asyncio.run(self._problem(self._router("a", "b"), []))

        assert "catalogue is empty" not in (problem or "")
