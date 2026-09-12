"""
REST surface.

Two audiences. The gateway publishes its catalogue and asks for decisions; an operator
looks at health and at what a call would resolve to. Both are plain HTTP, because the
gateway is a Spring service and should not have to speak MCP to reach this one.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.dispatcher import DispatchResult
from app.logcontext import acting
from app.models import Definition
from app.planner import Plan

router = APIRouter()


# --------------------------------------------------------------------- security

async def require_token(
    request: Request,
    x_mcp_token: Annotated[str | None, Header()] = None,
) -> None:
    """
    Shared secret check for the gateway facing routes.

    Skipped when no token is configured, so a local run needs no setup. That is a
    deliberate convenience with a sharp edge, which is why the health endpoint reports
    whether the check is active rather than leaving it invisible.
    """
    settings = request.app.state.settings

    if not settings.requires_token:
        return
    if x_mcp_token != settings.publisher_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-MCP-Token"
        )


Protected = Depends(require_token)


# --------------------------------------------------------------------- contracts

class PublishRequest(BaseModel):
    """The complete set of definitions the gateway wants exposed."""

    definitions: list[Definition] = Field(default_factory=list)


class PublishResponse(BaseModel):
    published: int
    received: int


class PriorTurn(BaseModel):
    """
    A question asked earlier in the same conversation, and what it became.

    The console keeps a thread and looked like a chat, but every prompt arrived alone: a
    follow-up such as "peki ya turkcell.com.tr icin?" had no antecedent, so the router had
    no idea which tool "peki ya" continued and the planner had no statement to vary.

    Deliberately thin. The sentence and the statement it produced are what a follow-up
    refers to; the rows that came back are not, and putting a result set in every
    subsequent prompt would spend the context window on data the model has no question
    about — besides copying production rows into somewhere new, which is the direction
    this stack has been moving away from.
    """

    model_config = ConfigDict(populate_by_name=True)

    prompt: str = ""
    tool_name: str | None = Field(default=None, alias="toolName")

    #: The command or query it resolved to, already masked.
    statement: str = ""

    @field_validator("prompt", "statement", mode="before")
    @classmethod
    def _absent_is_empty(cls, value: Any) -> Any:
        """
        A null from the sender means "not set", not a malformed request.

        The gateway's own turn has nullable columns — a turn the model answered on its own
        has no statement — and Jackson writes those as null. Rejecting the whole request
        for one of them meant a single answer-only turn in the history could stop every
        subsequent question in that conversation from being routed at all.
        """
        return "" if value is None else value


class PromptRequest(BaseModel):
    """A sentence, and who typed it."""

    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(min_length=1)
    actor: str = ""

    #: The acting user's id, for the log lines this request produces.
    #:
    #: Separate from `actor`, which is a label kept with the job and shown in the console.
    #: This one never leaves the logs, and it is an id rather than an address because a log
    #: store with no encryption is the wrong place for one.
    actor_id: str = Field(default="", alias="actorId")

    #: What everything older than the window came to, in a few sentences.
    #:
    #: Kept by the gateway, which owns the conversation. Without it a long session simply
    #: forgets its beginning: the window slides, and the question that established what
    #: everyone has been talking about falls off the end of it.
    summary: str = ""

    #: What was asked before this, oldest first.
    #:
    #: Supplied by the gateway, which owns the conversation. This service holds no history
    #: of its own and should not: it answers one request at a time, and a second copy of
    #: the thread here would be one more thing that can disagree with the first.
    history: list[PriorTurn] = Field(default_factory=list)

    #: Which tool to use, when the caller has already decided.
    #:
    #: Naming one turns this into a much smaller question: not *which* tool, only what to
    #: give it. A definition is written for a particular job, and an operator who knows
    #: which job they are doing should not have to hope a model agrees.
    tool_name: str | None = Field(default=None, alias="toolName")

    #: Decide only. The default is to stop at the plan, because a prompt is a sentence and
    #: a sentence is a poor place to hide "and then do it to production".
    execute: bool = False

    #: The command a person was shown and approved, when this request is that approval.
    #:
    #: Planning is not deterministic. A step proposed as `systemctl start httpd` and
    #: approved on the strength of that sentence was re-planned on the way to running and
    #: came back as the install command instead — so what ran was not what anybody had
    #: agreed to. Set this and nothing is dispatched unless the plan matches it.
    #:
    #: Not a way to supply a command: what runs is still what the planner writes and the
    #: guardrails pass. This only refuses to run anything else.
    expect: str = ""

    #: True when nobody typed this — a goal loop wrote it from an earlier answer.
    #:
    #: Anything it plans that changes something is held for approval, whatever the action's
    #: own setting says. The caller cannot make this decision: which action a step lands on
    #: is only known once it has been planned, and deciding from the action that *finished*
    #: is how a search that ended in a delete ran the delete unattended.
    unattended: bool = False

    #: The one action this request is for, when the caller already knows.
    #:
    #: A goal-loop step taking up a deferred action knows exactly which action was waiting
    #: — the plan recorded it. Asking a model to choose again spends a call to be told
    #: something already written down, and gives it a chance to choose differently: handed
    #: "delete the user with id 59", it once picked the search instead.
    action_id: int | None = Field(default=None, alias="actionId")

    #: Values the caller has already decided, which win over anything routing extracts.
    #:
    #: A goal-loop step carries the value it read out of the previous answer. Routing had
    #: to find it in a sentence before, and a sentence written as narration — "processing
    #: the first one (id 59) now" — kept the value out of reach.
    arguments: dict[str, str] = Field(default_factory=dict)


class PromptResponse(BaseModel):
    """What a prompt turned into."""

    #: An answer from the model itself, when no tool was the right thing to call.
    #:
    #: Kept apart from a plan, and it has to be: this is a model talking, not a system
    #: reporting. Shown the same way, a guess about how many accounts there are would read
    #: exactly like a count.
    answer: str = ""

    #: Which tool was chosen, and why. Absent when nothing matched.
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""

    #: Why there is no plan, when there is none.
    problem: str | None = None

    status: str | None = None
    plan: Plan | None = None
    dispatch: DispatchResult | None = None


class ExecutionRequest(BaseModel):
    """
    A request to decide what a tool call should do.

    Either name a tool already published, or carry the definition inline. Inline wins,
    which lets the gateway ask about something it has not published — a preview of an
    unsaved edit, for instance.
    """

    model_config = ConfigDict(populate_by_name=True)

    tool_name: str | None = Field(default=None, alias="toolName")
    definition: Definition | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)

    #: Who asked. Recorded in the decision and passed to the executor.
    actor: str | None = None


class ExecutionResponse(BaseModel):
    """The decision, and what became of it."""

    status: Literal["planned", "incomplete", "rejected"]
    plan: Plan
    dispatch: DispatchResult


# ------------------------------------------------------------------------ routes

@router.get("/health", tags=["ops"])
async def health(request: Request) -> dict[str, Any]:
    """Liveness, plus everything a caller might otherwise have to guess."""
    catalogue = request.app.state.catalogue
    settings = request.app.state.settings
    published_at = catalogue.published_at

    return {
        "status": "ok",
        "catalogue": {
            "tools": catalogue.size,
            "published_at": published_at.isoformat() if published_at else None,
        },
        "planner": {
            # Which providers this deployment can serve, so a definition naming one it
            # cannot reach is visible here rather than only when a call is refused.
            "providers": settings.configured_providers,
            "dynamic_mode_available": bool(settings.configured_providers),
        },
        "router": {
            # Which model decides that a sentence is a call to one particular tool. Named
            # here because a routing failure reads as "no tool matches", which looks like a
            # gap in the catalogue rather than a model that could not answer.
            "model": settings.router_model,
            "provider": settings.router_provider,
        },
        "configuration": {
            # Where these settings came from. A service quietly running on its local
            # fallback behaves correctly and is configured differently from its neighbours,
            # which is the kind of difference nobody finds by reading the code.
            "source": "local" if settings.remote_status else "mcp-config",
            "reason": settings.remote_status or None,
        },
        "auth": {"token_required": settings.requires_token},
        "executor": {
            "configured": settings.has_queue,
            "queue": settings.queue_name if settings.has_queue else None,
        },
    }


class TakenStep(BaseModel):
    """
    A step already taken, and the shape of what it returned.

    The shape, not the rows. "Total those" is answered by writing a SUM over the same
    table, which needs the columns and not the values; sending the whole result would put
    production data into every subsequent prompt to save an arithmetic the database does
    better anyway.
    """

    model_config = ConfigDict(populate_by_name=True)

    #: What was asked, in words. Not the SQL: shown statements, the step planner mirrors
    #: them and returns its next step already written, skipping the planner that has the
    #: schema.
    request: str = ""

    #: The query it produced. Shown so the goal's own turn — whose request is the whole
    #: goal — does not read as a step where nothing has happened yet.
    statement: str = ""

    row_count: int | None = Field(default=None, alias="rowCount")
    columns: list[str] = Field(default_factory=list)

    #: The first rows, already rendered. A handful, so a column of ids reads as ids.
    sample: str = ""

    #: What a shell step printed, trimmed. A command's result has no columns and no rows:
    #: "is httpd running" is answered by the words systemctl wrote, and a step planner
    #: shown only a row count would be deciding the next command with nothing to go on.
    output: str = ""

    #: Why the step produced nothing, when it failed.
    problem: str = ""

    @field_validator("request", "statement", "sample", "output", "problem", mode="before")
    @classmethod
    def _absent_is_empty(cls, value: Any) -> Any:
        """
        A null from the sender means "not set", not a malformed request.

        The gateway's columns are nullable — a step that succeeded has no problem, a step
        that returned no rows has no sample — and Jackson writes those as null. Rejecting
        the whole request for one of them ended the goal loop with a 422 that read, in the
        log, as the model deciding it was finished.
        """
        return "" if value is None else value

    @field_validator("columns", mode="before")
    @classmethod
    def _absent_is_empty_list(cls, value: Any) -> Any:
        return [] if value is None else value


class StepRequest(BaseModel):
    """A goal, and how far it has got."""

    model_config = ConfigDict(populate_by_name=True)

    goal: str = Field(min_length=1)

    #: The tool the goal was routed to. Its tables are what a step may name.
    tool_name: str | None = Field(default=None, alias="toolName")

    steps: list[TakenStep] = Field(default_factory=list)
    history: list[PriorTurn] = Field(default_factory=list)
    summary: str = ""
    limit: int = 5

    #: What the steps are made of: "db" for queries, "ssh" for commands on a server. The
    #: two need different instructions — one may only read, the other exists to change a
    #: machine — and a single prompt covering both told the shell planner it must not
    #: write, which is most of what installing something is.
    kind: str = "db"

    #: An action the plan set aside, and what it is waiting for.
    #:
    #: Its presence changes the question. Without it the model is asked whether the goal
    #: is met, which is a judgement; with it the plan has already answered that — something
    #: is waiting — and what is left is to write the request for it from what came back.
    waiting_for: str = Field(default="", alias="waitingFor")


class StepResponse(BaseModel):
    done: bool = True
    request: str = ""
    reason: str = ""

    #: The inputs a waiting action needed, read out of the answers so far.
    #:
    #: Empty for an ordinary step, where the request is the whole answer. Present when
    #: something was waiting: the tool and the action are already known, so the only open
    #: question is the value, and asking for the value directly is a smaller question than
    #: asking for a sentence that has to be read back into one.
    values: dict[str, str] = Field(default_factory=dict)


@router.post("/api/v1/steps", tags=["gateway"], dependencies=[Protected])
async def next_step(body: StepRequest, request: Request) -> StepResponse:
    """
    The next question a goal needs, or the news that it needs none.

    Returns a sentence rather than a plan, deliberately: the caller puts it back through
    the ordinary prompt path, so every step is planned, checked and masked exactly as a
    typed question is. The loop adds a decision and no new way to reach a database.
    """
    step = await request.app.state.prompt_router.next_step(
        body.goal, body.steps, body.history, body.summary, body.limit,
        request.app.state.catalogue.find(body.tool_name) if body.tool_name else None,
        body.kind, body.waiting_for,
    )

    return StepResponse(
        done=step.done, request=step.request, reason=step.reason, values=step.as_dict()
    )


class SummaryRequest(BaseModel):
    """Turns that have fallen out of the window, and the summary they fold into."""

    model_config = ConfigDict(populate_by_name=True)

    summary: str = ""
    turns: list[PriorTurn] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    summary: str = ""


@router.post("/api/v1/summaries", tags=["gateway"], dependencies=[Protected])
async def summarise(body: SummaryRequest, request: Request) -> SummaryResponse:
    """
    Folds turns that have dropped out of the window into a running summary.

    Separate from planning on purpose: it costs a model call, it is not on the path of any
    question, and a conversation whose summary could not be updated should still answer the
    next thing asked. The gateway calls this when its window slides, and keeps whatever it
    already had if this fails.
    """
    summary = await request.app.state.prompt_router.summarise(body.summary, body.turns)
    return SummaryResponse(summary=summary)


@router.put("/api/v1/catalogue", tags=["gateway"], dependencies=[Protected])
async def publish_catalogue(body: PublishRequest, request: Request) -> PublishResponse:
    """
    Replace the published catalogue.

    A full replacement, so a definition the gateway stops sending stops being callable.
    Disabled definitions may be included; they are dropped here.
    """
    published = request.app.state.catalogue.replace(body.definitions)
    return PublishResponse(published=published, received=len(body.definitions))


@router.post("/api/v1/executions", tags=["gateway"], dependencies=[Protected])
async def create_execution(body: ExecutionRequest, request: Request) -> ExecutionResponse:
    """
    Decide what a tool call resolves to.

    This is the endpoint the gateway calls. The decision is made here; handing it to an
    executor is a separate step, and today that step reports that no executor exists
    rather than pretending the work ran.
    """
    definition = _resolve_definition(body, request)

    plan = await request.app.state.planner.plan(definition, body.arguments)
    dispatch = await request.app.state.dispatcher.dispatch(
        definition, plan, body.arguments, actor=body.actor
    )

    return ExecutionResponse(status=plan.status, plan=plan, dispatch=dispatch)


@router.post("/api/v1/prompts", tags=["gateway"], dependencies=[Protected])
async def create_prompt(body: PromptRequest, request: Request) -> PromptResponse:
    """
    Turn a sentence into a tool call.

    Two steps that are kept apart on purpose: a model chooses the tool, and the ordinary
    planning path decides what that tool resolves to. The second step is the one with the
    guardrails, and routing a prompt must not be a way around them.

    Naming a tool skips the first step, not the second. It is a smaller question — what to
    give a tool, rather than which tool — and it is the caller's to answer when they already
    know which job they are doing.

    Nothing is dispatched unless the caller asks. A prompt is a sentence, and a sentence is
    a poor place to hide "and then do it to production".
    """
    # First thing, so everything logged from here on says who it was for — including the
    # guardrail refusals, which are the lines somebody comes back to ask about.
    acting(body.actor_id)

    router = request.app.state.prompt_router

    if body.tool_name:
        chosen = request.app.state.catalogue.find(body.tool_name)
        if chosen is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown tool: {body.tool_name}. Publish the catalogue first.",
            )
        routed = await router.arguments_for(
            chosen, body.prompt, body.history, body.summary
        )
    else:
        routed = await router.route(body.prompt, body.history, body.summary)

    if not routed.chosen:
        # Nothing matched, which is not the same as nothing to say. A question about the
        # conversation itself is answerable without running anything, and replying "no
        # published tool matches this request" to one made the console useless for the
        # questions people actually type into it.
        answer = await router.answer(body.prompt, body.history, body.summary)

        return PromptResponse(
            reasoning=routed.reasoning,
            problem=routed.problem or "No tool was chosen",
            answer=answer,
        )

    definition = request.app.state.catalogue.find(routed.tool_name)
    if definition is None:
        # The router checks the name against the catalogue, so this means the catalogue
        # changed between the two steps rather than that the model invented something.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{routed.tool_name} is no longer published",
        )

    arguments = _arguments_for(routed, body)

    plan = await request.app.state.planner.plan(
        definition, arguments, body.prompt, body.history, body.summary, body.action_id,
    )

    dispatch = None
    if body.execute:
        dispatch = await _dispatch(request, definition, plan, routed, body, arguments)

    return PromptResponse(
        tool_name=routed.tool_name,
        # What the plan ran on, not what routing found in the sentence. The two were the
        # same thing until a step could arrive with values already read out of an answer,
        # and reporting the sentence's version left the caller — and the panel's own
        # "arguments" block — describing a plan that had used something else.
        arguments=arguments,
        reasoning=routed.reasoning,
        problem=routed.problem,
        status=plan.status,
        plan=plan,
        dispatch=dispatch,
    )


async def _dispatch(request: Request, definition: Any, plan: Any, routed: Any,
                    body: PromptRequest, arguments: dict[str, str]) -> DispatchResult:
    """
    Carries out a plan, or says why it is not being carried out.

    Two things stand between a plan and the executor, and they are different questions.
    *Has anybody agreed to this* — an action the operator marked as needing approval runs
    only once a person has said yes to the command in front of them. And *is this still the
    thing they agreed to* — planning is not deterministic, so the command approved and the
    command about to run have to be compared rather than assumed equal.
    """
    approved = bool(body.expect.strip())

    # Only a plan that came out well has anything to approve. A rejected one has no command
    # in it — the guardrails refused the one there was — and offering it for approval would
    # put a button under a refusal, waiting on a decision nobody can usefully make.
    if plan.status != "planned":
        return await request.app.state.dispatcher.dispatch(
            definition, plan, arguments, actor=body.actor
        )

    if (_needs_approval(plan) or (body.unattended and _changes_something(plan))) \
            and not approved:
        return DispatchResult(
            status="awaiting_approval",
            reason=(
                "This action needs approval before it runs. Nothing has been dispatched."
            ),
        )

    if not _matches(plan, body.expect):
        return DispatchResult(
            status="refused",
            reason=(
                "The plan changed since it was approved; nothing is dispatched. "
                "Approve the command as it now reads."
            ),
        )

    return await request.app.state.dispatcher.dispatch(
        definition, plan, arguments, actor=body.actor
    )


def _arguments_for(routed: Any, body: PromptRequest) -> dict[str, str]:
    """
    The values this request runs with.

    A typed request and a step the loop wrote are not the same kind of thing, and routing
    is trusted differently in each.

    When somebody typed the sentence, routing reading values out of it is the whole point —
    "delete user 13" has the 13 in it — and the caller's own values win where both have
    something to say.

    When the loop wrote it, they do not. The sentence is machine-written and carries no
    values by design; asked to find one in "Processing the first user found with first_name
    'Yigit'", this model produced `id: 1` and explained that a previous search had returned
    a user with that id. It had not. The step went to approval as a DELETE against a record
    nobody had asked about.

    So an unattended step runs on what the loop read out of the answers, and nothing else.
    Where that is empty the plan says the input is missing, which is true, visible, and
    harmless — unlike a number that came from nowhere.
    """
    if body.unattended:
        return dict(body.arguments)

    return {**routed.arguments, **body.arguments}


def _needs_approval(plan: Any) -> bool:
    """Whether any action in this plan was marked as needing a person to say yes."""
    return any(action.requires_approval for action in plan.actions)


def _changes_something(plan: Any) -> bool:
    """Whether anything this plan would carry out is a write."""
    return any(action.writes for action in plan.actions if not action.skipped)


def _matches(plan: Any, expected: str) -> bool:
    """
    Whether the plan still says what somebody approved.

    Empty ``expected`` means nobody approved anything in particular — an ordinary prompt —
    and everything matches. Otherwise every action the plan would run must resolve to the
    command that was shown; a plan with no actions matches nothing, because there is
    nothing there to be the thing agreed to.

    The actions set aside are not among them. They resolve to nothing — there was no point
    resolving what is not going to run — so comparing them refused every approval of a
    definition with more than one action.
    """
    wanted = expected.strip()
    if not wanted:
        return True

    running = [action for action in plan.actions if not action.skipped]

    return bool(running) and all(action.resolved.strip() == wanted for action in running)


@router.get("/api/v1/tools", tags=["ops"])
async def list_tools(request: Request) -> list[dict[str, Any]]:
    """The catalogue as this service currently holds it."""
    return [
        {
            "name": definition.tool_name,
            "description": definition.tool_description,
            "inputSchema": definition.input_schema(),
            "definitionId": definition.id,
            "model": definition.model_identifier,
            "actionCount": len(definition.actions),
        }
        for definition in request.app.state.catalogue.all()
    ]


def _resolve_definition(body: ExecutionRequest, request: Request) -> Definition:
    if body.definition is not None:
        return body.definition

    if not body.tool_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide either toolName or an inline definition",
        )

    definition = request.app.state.catalogue.find(body.tool_name)
    if definition is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown tool: {body.tool_name}. Publish the catalogue first.",
        )
    return definition
