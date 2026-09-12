"""
Decides what a tool call should do.

This service plans; it does not execute. Every path here ends in a :class:`Plan` that
describes exactly which action would run, against which targets, with which command or
query — and nothing is sent anywhere.

Two kinds of decision live here:

* **Mechanical.** Which action, and what the template renders to. When a definition has
  one action and it is static, no model is involved at all: substituting placeholders is
  string work, and paying for a model call to do it would be waste.
* **Authored.** The command or query itself, when the action is in dynamic mode. That is
  the only reason a model is called, and whatever it writes is put through the guardrails
  before it appears in a plan.

Which model answers is the definition's choice, not this module's: Claude, ChatGPT and a
self hosted open model all reach the same code here. See :mod:`app.providers`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import Settings
from app.guardrails import (
    GuardrailVerdict,
    check_command,
    check_query,
    check_static_command,
    check_static_query,
    echoes_the_request,
    first_command,
    unrestricted,
    repeats_earlier,
    unrequested_filters,
)
from app.models import Action, Definition
from app.providers import (
    FALLBACK_MODEL,
    Authored,
    BackendRegistry,
    ModelBackend,
    ModelOptions,
)
from app.templating import (
    build_values,
    find_unresolved,
    render,
    render_body,
    render_url,
    secret_keys,
    substituted_values,
)

logger = logging.getLogger(__name__)

#: How many earlier turns a model is shown. See :meth:`Planner._thread`.
_THREAD_LIMIT = 6

#: How many times a model may be asked to write one command or query.
#:
#: Two: one attempt, and one more with the guardrail's refusal in hand. A model that
#: ignores a specific refusal twice will not be talked round by a third ask.
_AUTHOR_ATTEMPTS = 2

class Warning(BaseModel):
    """
    Something observed about a statement that its reader should know.

    A code and the thing observed, never a sentence: the reader's language is the panel's
    business, and what this can honestly claim is narrow enough that phrasing it here would
    overstate it. ``unrequested_filter`` says a value is not in the request — which is not
    the same as "nobody asked for it", because a question in Turkish about aktif accounts
    asks for exactly the ``status = 'active'`` it reports.
    """

    #: ``unrequested_filter``, ``repeats_earlier`` or ``request_as_value``.
    code: str

    #: The filter, or the statement, that was observed.
    detail: str = ""


class PlannedAction(BaseModel):
    """One action, resolved to what would actually happen."""

    action_id: int
    name: str
    kind: Literal["rest", "ssh", "db"]
    mode: Literal["static", "dynamic"]

    #: Hosts the action would touch. Empty for REST and database actions.
    targets: list[str] = Field(default_factory=list)

    #: Rendered command, query or request line, with secrets masked.
    resolved: str = ""

    #: Set when the model authored the command or query.
    authored_by_model: bool = False

    #: Whether carrying this out changes anything.
    #:
    #: Decided where the action's settings are, not guessed at from its kind: a REST call
    #: writes or does not depending on its method, and a database action on whether it was
    #: declared read only. A command always might.
    writes: bool = False

    #: Inputs this action needs that only an earlier one can answer for.
    #:
    #: The names, not the sentence about them. The sentence is for a person reading the
    #: plan; a model asked to write the next step needs to know it is looking for an id.
    waiting_for: list[str] = Field(default_factory=list)

    #: True when this action belongs to the definition but the request did not ask for it.
    #:
    #: Kept in the plan rather than left out. Somebody reading a plan with one call in it
    #: has to be able to tell that the other three were considered and set aside, which is
    #: a different fact from their not existing.
    skipped: bool = False
    skip_reason: str = ""

    #: What the model actually wrote, before rendering and masking.
    #:
    #: Excluded from serialisation, so it never reaches a response body: this is the
    #: template, and the plan's job is to be read. The dispatcher needs it because a
    #: dynamic action has nothing stored on the definition to rebuild from — the query
    #: exists only here, and without it the executor was handed an empty statement.
    authored: str = Field(default="", exclude=True, repr=False)

    #: Populated when a guardrail rejected what the model produced.
    rejected_reasons: list[str] = Field(default_factory=list)

    #: What is worth reading before trusting this answer.
    #:
    #: Not rejections. Narrowing a result is usually right, and a plan blocked for every
    #: `where` would make dynamic queries useless. These are here so that the person
    #: reading the answer knows what it leaves out — the one thing a wrong-but-successful
    #: query never tells them.
    #:
    #: Coded rather than phrased, so the screen can put the sentence in the reader's own
    #: language and the record keeps what was actually observed.
    warnings: list[Warning] = Field(default_factory=list)

    requires_approval: bool = False


class Plan(BaseModel):
    """What a tool call resolved to. Nothing in here has been executed."""

    tool: str
    definition_id: int
    model: str | None = None
    status: Literal["planned", "rejected", "incomplete"] = "planned"
    actions: list[PlannedAction] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)

    #: Worth reading before trusting the answer, but not a reason to refuse the plan.
    warnings: list[Warning] = Field(default_factory=list)

    #: Keys whose values were masked. Present so a caller knows a secret was involved.
    masked_inputs: list[str] = Field(default_factory=list)


def _set_aside(action: Action, reason: str) -> PlannedAction:
    """An action kept in the plan so it can be seen, and not run."""
    return PlannedAction(
        action_id=action.id, name=action.name, kind=action.kind,
        mode="dynamic" if action.is_dynamic else "static",
        skipped=True, skip_reason=reason,
    )


#: HTTP methods that only ask.
_READ_METHODS = {"GET", "HEAD", "OPTIONS"}


def _writes(action: Action) -> bool:
    """
    Whether carrying this action out changes anything.

    By its settings rather than by its kind. A REST call is a read or a write depending on
    its method; a database action on whether the operator declared it read only. A command
    is never assumed to be a read — what it does is written at run time.
    """
    if action.kind == "rest":
        return (action.config.method or "GET").strip().upper() not in _READ_METHODS

    if action.kind == "db":
        return not action.config.read_only

    return True


class _Chosen(BaseModel):
    """Which of a definition's actions this request needs."""

    actions: list[int] = Field(
        default_factory=list,
        description=(
            "The ids of the actions this request needs, in the order they should run. "
            "Empty when none of them answers it."
        ),
    )
    reason: str = Field(
        default="",
        description="One sentence on why those and not the others.",
    )


class _AuthoredCommand(BaseModel):
    """Structured answer expected from the model in dynamic mode."""

    command: str = Field(description="The single shell command to run, with no shell operators beyond those needed.")
    reasoning: str = Field(default="", description="One sentence on why this command answers the request.")


class _AuthoredQuery(BaseModel):
    """Structured answer expected from the model for a dynamic database action."""

    query: str = Field(description="A single SQL statement, without a trailing semicolon.")
    reasoning: str = Field(default="", description="One sentence on why this query answers the request.")


def _asked_before(history: Sequence[Any]) -> str:
    """
    What the conversation has been about, for a choice that needs the verb.

    Only the questions, not the statements they became: what is missing from a loop's step
    is the operation the goal asked for, and that is in the sentence somebody typed. Oldest
    first, so the goal itself — which is the one that says "and delete them" — is at the
    top rather than buried under the steps it produced.
    """
    asked = [
        str(getattr(turn, "prompt", "") or "").strip()
        for turn in history
    ]
    asked = [line for line in asked if line]

    if not asked:
        return ""

    return "Asked earlier in this conversation, oldest first:\n" + "\n".join(
        f"- {line}" for line in asked[-6:]
    ) + "\n\n"


class Planner:
    """Turns a tool call into a plan."""

    def __init__(self, settings: Settings, cipher: Any = None) -> None:
        self._settings = settings
        self._backends = BackendRegistry(settings, cipher)

    async def plan(
        self,
        definition: Definition,
        arguments: dict[str, Any],
        request: str = "",
        history: Sequence[Any] = (),
        summary: str = "",
        action_id: int | None = None,
    ) -> Plan:
        """
        Resolve every action of ``definition`` against the supplied arguments.

        ``request`` is what the person actually asked, when there was a sentence. It is
        passed to a model writing a command or query and to nothing else: routing had it,
        the planner did not, and a definition with no declared inputs left the model with
        the action's name and nothing more. Asked for three columns of the last three
        rows, it wrote ``SELECT * FROM tblAccounts LIMIT 100`` — a fair answer to the only
        question it was given.

        Empty for a tools/call from a real MCP client, which has arguments and no prose.
        """
        values = build_values(definition.inputs, arguments)
        masked = secret_keys(definition.inputs)

        plan = Plan(
            tool=definition.tool_name,
            definition_id=definition.id,
            model=definition.model_identifier,
            masked_inputs=sorted(masked & values.keys()),
        )

        ordered = sorted(definition.actions, key=lambda item: item.position)
        chosen, skipped = await self._choose(
            request, definition, ordered, plan, values, action_id, history
        )

        if plan.problems:
            plan.status = "incomplete"
            return plan

        # Required, but only of the actions actually being run. The field lives on the
        # definition and the templates say which action needs it, so an id required by a
        # delete does not block a list that never mentions one — which is what made a
        # definition with four actions impossible to declare honestly.
        needed = {key for action in chosen for key in action.inputs_used()}

        missing_required = [
            item.key
            for item in definition.inputs
            if item.required and item.key in needed and item.key not in values
        ]
        if missing_required:
            plan.status = "incomplete"
            plan.problems.append(
                f"Missing required input(s): {', '.join(sorted(missing_required))}"
            )
            return plan

        for action in chosen:
            planned = await self._plan_action(
                definition, action, values, masked, request, history, summary
            )
            plan.actions.append(planned)

        # Listed, not silently dropped. Somebody reading a plan with one call in it needs
        # to be able to tell that the other three were considered and left out.
        plan.actions.extend(skipped)

        if any(item.rejected_reasons for item in plan.actions):
            plan.status = "rejected"
            plan.problems.extend(
                reason for item in plan.actions for reason in item.rejected_reasons
            )

        # Kept apart from problems: the plan is still planned, and a caller that treated
        # these as failures would refuse work that is very likely correct.
        plan.warnings.extend(
            warning for item in plan.actions for warning in item.warnings
        )

        return plan

    async def _choose(
        self, request: str, definition: Definition, ordered: list[Action], plan: Plan,
        values: dict[str, str], action_id: int | None = None,
        history: Sequence[Any] = (),
    ) -> tuple[list[Action], list[PlannedAction]]:
        """
        Splits a definition's actions into the ones this request needs and the rest.

        A single action is not a choice, and asking a model about it would spend a call to
        be told the only answer. Everything else asks — including the case where no
        sentence was typed, because a tools/call with arguments and no prose has no way to
        say which of four operations it meant, and guessing all of them is how a search
        also deleted something.
        """
        if len(ordered) <= 1:
            return ordered, []

        # Named by the caller: a goal-loop step taking up an action the plan set aside
        # knows which one it was. Asking again spends a model call to be told something
        # already written down — and gives it a chance to answer differently.
        if action_id is not None:
            named = [action for action in ordered if action.id == action_id]

            if named:
                return named, [
                    _set_aside(action, "the step is for another action")
                    for action in ordered if action.id != action_id
                ]

            plan.problems.append(f"Action {action_id} is not part of this tool")
            return [], []

        if not request.strip():
            plan.problems.append(
                "This tool has several actions and the request said which in no words at "
                "all. Call one action's tool directly, or ask in a sentence."
            )
            return [], []

        chosen = await self._choose_actions(request, definition, history)

        if chosen is None:
            # Not a licence to run everything. A model that could not answer is exactly
            # when running four writes would be least defensible.
            plan.problems.append(
                "Which of this tool's actions the request needs could not be decided; "
                "nothing was planned"
            )
            return [], []

        wanted = [action for action in ordered if action.id in set(chosen.actions)]

        if not wanted:
            plan.problems.append(
                "None of this tool's actions answers the request"
                + (f": {chosen.reason}" if chosen.reason else "")
            )
            return [], []

        wanted, deferred = self._defer_unfillable(wanted, values)

        left = [
            _set_aside(action, chosen.reason or "the request did not ask for this")
            for action in ordered
            if action.id not in {item.id for item in wanted}
            and action.id not in {item.action_id for item in deferred}
        ]

        return wanted, left + deferred

    @staticmethod
    def _defer_unfillable(
        wanted: list[Action], values: dict[str, str]
    ) -> tuple[list[Action], list[PlannedAction]]:
        """
        Holds back a chosen action whose inputs only an earlier one can supply.

        "Find the user called Mehmet Bulut and delete them" needs both actions, and the
        second cannot run yet: the id it deletes by is in the first one's answer, which
        does not exist at planning time. Asked to choose, a model picks both — reasonably,
        because the request does ask for both — and the plan was then refused whole for an
        id nobody could have supplied.

        So it is set aside rather than refused, and the goal loop takes it up once the
        first action has answered. Only when something runs before it: an action nobody
        can fill and nothing precedes is a request with no target, and refusing that is
        the right answer.
        """
        runnable: list[Action] = []
        deferred: list[PlannedAction] = []

        for action in wanted:
            unfillable = sorted(key for key in action.inputs_used() if key not in values)

            if unfillable and runnable:
                deferred.append(PlannedAction(
                    action_id=action.id, name=action.name, kind=action.kind,
                    mode="dynamic" if action.is_dynamic else "static",
                    skipped=True,
                    waiting_for=unfillable,
                    skip_reason=(
                        "waiting on " + ", ".join(unfillable)
                        + ", which an earlier action has to answer first"
                    ),
                ))
                continue

            runnable.append(action)

        return runnable, deferred

    async def _choose_actions(
        self, request: str, definition: Definition, history: Sequence[Any] = ()
    ) -> _Chosen | None:
        """
        Asks which actions the request needs.

        The definition's own model, not the router's: this is a question about what these
        particular actions do, and the operator chose a model for that.

        Given what was asked before it, because a step the goal loop wrote does not always
        carry the verb. Asked to "find the Yigits and delete them", the loop's second step
        read "Processing the second user found with first_name 'Yigit'" — which names a
        record and no operation, and was answered with the listing action. The delete never
        came, and from the console the goal had simply stopped after one.

        Returns ``None`` when the choice could not be made, which the caller treats as
        "run nothing". A model that failed to answer is not permission to run four writes.
        """
        backend = self._backends.for_definition(
            definition.model_provider, definition.model_endpoint
        )
        if backend is None:
            return None

        catalogue = "\n".join(
            f"{action.id}. {action.name} — {action.summary()}"
            + (f" ({action.description})" if action.description else "")
            for action in sorted(definition.actions, key=lambda item: item.position)
        )

        result = await backend.author(
            model=definition.model_identifier or FALLBACK_MODEL,
            system=(
                "You choose which of a tool's actions a request needs.\n\n"
                "The actions belong to one tool because they act on the same thing — "
                "listing, creating, updating and deleting the same resource, say. A "
                "request usually needs one of them. Choose only what it asks for: an "
                "action nobody asked for is a write nobody asked for, and reading a "
                "record is not a reason to also replace it.\n\n"
                "Choose more than one only when the request plainly asks for more than "
                "one thing and the order between them is fixed. If it depends on how the "
                "first one turns out — \"add them if they are not there\" — choose the "
                "first one alone; what comes after is decided when its answer is in.\n\n"
                "Return an empty list when none of them answers the request. Saying so is "
                "an answer; picking the nearest one is not."
            ),
            user=(
                f"{_asked_before(history)}The request:\n{request}\n\n"
                f"The actions:\n{catalogue}"
            ),
            schema=_Chosen,
        )

        if not result.ok or not isinstance(result.parsed, _Chosen):
            logger.warning("Actions could not be chosen: %s", result.error)
            return None

        return result.parsed

    # ------------------------------------------------------------------ actions

    async def _plan_action(
        self,
        definition: Definition,
        action: Action,
        values: dict[str, str],
        masked: set[str],
        request: str = "",
        history: Sequence[Any] = (),
        summary: str = "",
    ) -> PlannedAction:
        planned = PlannedAction(
            action_id=action.id,
            name=action.name,
            kind=action.kind,
            mode="dynamic" if action.is_dynamic else "static",
            targets=self._targets(action, values),
            requires_approval=bool(action.config.require_approval),
            writes=_writes(action),
        )

        if action.kind == "rest":
            planned.resolved = self._render_rest(action, values, masked)
            self._note_unresolved(definition, action, values, planned)
            return planned

        if not action.is_dynamic:
            template = action.config.command if action.kind == "ssh" else action.config.query
            planned.resolved = render(template, values, mask=masked)
            self._note_unresolved(definition, action, values, planned)
            self._check_static(definition, action, template, values, planned)
            return planned

        await self._author(
            definition, action, values, masked, planned, request, history, summary
        )
        return planned

    def _check_static(
        self,
        definition: Definition,
        action: Action,
        template: str | None,
        values: dict[str, str],
        planned: PlannedAction,
    ) -> None:
        """
        Run the action's own guardrails over a command the operator templated.

        Skipped for a while, on the reasoning that a static command was reviewed when it
        was saved. That reasoning only covers the template: the arguments are supplied per
        call, and a definition that declares a blocked pattern means it whether the
        pattern arrived from the template or from an input.

        The unmasked command is what gets checked. The masked one is what the plan shows,
        and checking that instead would let a secret's contents past every rule.
        """
        unmasked = render(template, values)
        substituted = substituted_values(template, values)

        # Which inputs are bodies of text rather than words. A file's contents cannot pass
        # the shell-metacharacter scan and should not have to; the heredoc rule that
        # replaces it is in check_static_command.
        blocks = {
            declared.key for declared in definition.inputs if declared.type == "block"
        }

        verdict = (
            check_static_command(unmasked, action.config, substituted, blocks)
            if action.kind == "ssh"
            else check_static_query(unmasked, action.config, substituted)
        )

        if not verdict.allowed:
            planned.rejected_reasons.extend(verdict.reasons)

    async def _author(
        self,
        definition: Definition,
        action: Action,
        values: dict[str, str],
        masked: set[str],
        planned: PlannedAction,
        request: str = "",
        history: Sequence[Any] = (),
        summary: str = "",
    ) -> None:
        """Ask the definition's model to write the command or query, then check it."""
        if action.kind == "db" and not self._schema_for(action.config):
            # Refused before the model is called, because a model asked to write SQL
            # against a database it knows nothing about does not say so — it invents a
            # plausible table name and the statement fails on the server, or worse reads
            # the wrong table. Nothing downstream can tell that apart from a real answer.
            planned.rejected_reasons.append(
                "No schema is known for this action; read the schema or write a schema "
                "hint before asking the model for a query"
            )
            return

        backend = self._backends.for_model(definition)
        if backend is None:
            planned.rejected_reasons.append(
                self._backends.unavailable_reason(
                    definition.model_provider, definition.model_endpoint
                )
            )
            return

        planned.authored_by_model = True
        authored = ""
        verdict: GuardrailVerdict | None = None

        # Two attempts, and the second is told exactly why the first was refused.
        #
        # The instructions already say not to chain, and half the time a request with an
        # "if it is not already there" in it produced a `||` anyway — the shell's own way
        # of saying it. Repeating the rule more loudly is what had already been tried. The
        # guardrail, meanwhile, knows precisely what was wrong with the command it just
        # read, and that is a better instruction than any wording written in advance.
        #
        # Once, not until it succeeds: a model that ignores a specific refusal twice is not
        # going to be talked round by a third ask, and a loop here would spend somebody's
        # money finding that out.
        for attempt in range(_AUTHOR_ATTEMPTS):
            result = await self._call_model(
                backend, definition, action, values, request, history, summary,
                refused=verdict.reasons if verdict else (),
            )
            if not result.ok:
                planned.rejected_reasons.append(result.error or "The model produced nothing")
                return

            authored = result.text or ""
            left_for_later = ""

            # A model told "one command, and only one" writes a chain anyway, often enough
            # that the instruction cannot be the whole answer: "apache kur ve calistir" is
            # one goal and two commands, and `install && start` is the obvious way to write
            # it. The prompt already asks for the first one only; this takes it. What the
            # chain's tail asked for is the next step's business, and the goal loop asks for
            # it once this one has run.
            if action.kind == "ssh":
                authored, left_for_later = first_command(authored)

            # Applied identically whatever wrote it. A weaker model, or a server that could
            # not be held to a schema, is exactly the case these checks exist for — so the
            # provider must not be able to influence how hard they are.
            verdict = (
                check_command(authored, action.config)
                if action.kind == "ssh"
                else check_query(authored, action.config)
            )

            if verdict.allowed:
                break

            logger.warning(
                "Guardrail rejected a generated %s for action %s (attempt %d): %s",
                action.kind, action.id, attempt + 1, "; ".join(verdict.reasons),
            )

        if verdict is None or not verdict.allowed:
            # The rejected text is kept out of the plan on purpose: it is unvetted.
            planned.rejected_reasons.extend(verdict.reasons if verdict else [])
            return

        planned.authored = authored
        planned.resolved = render(authored, values, mask=masked)

        if left_for_later:
            # Said, not silently done. Somebody reading "apache kur ve calistir" and one
            # install command has to be able to tell that the start was understood and
            # deferred, rather than never seen.
            planned.warnings.append(
                Warning(code="first_command_only", detail=left_for_later)
            )

        if action.kind == "ssh" and unrestricted(action.config.allowed_commands):
            # Said on every run, not once at configuration time. Somebody reading a command
            # a week later has no way to tell whether it was one of a handful an operator
            # chose or anything the model felt like, and those are different facts.
            planned.warnings.append(
                Warning(code="unrestricted_commands", detail=planned.resolved)
            )

        if action.kind == "db":
            # Against everything a value could legitimately have come from: the sentence,
            # the arguments that came with it, and whatever the operator wrote about this
            # action. A literal in none of them was the model's own idea.
            statements = [getattr(turn, "statement", "") for turn in history]

            # Deterministic, and it has to be: the wording that produces a repeat is
            # endless. Naming one shape in the router's instructions fixed that shape and
            # the next phrasing arrived the following day, re-ran a query from seven turns
            # earlier and presented its rows as "all previous results".
            if repeats_earlier(authored, statements):
                planned.warnings.append(Warning(code="repeats_earlier", detail=authored))

            planned.warnings.extend(
                Warning(code="request_as_value", detail=filter_)
                for filter_ in echoes_the_request(authored, request)
            )

            # The filter itself, not a sentence about it. What this can honestly say is
            # "this value is not in the request" — whether that makes it wrong is the
            # reader's judgement, and the reader is looking at a screen that can put the
            # question in their own language.
            planned.warnings.extend(
                Warning(code="unrequested_filter", detail=filter_)
                for filter_ in unrequested_filters(
                    authored,
                    request,
                    " ".join(values.values()),
                    action.config.guidance or "",
                    action.config.schema_hint or "",
                    # A follow-up carries its value from the turn it continues: "peki ya
                    # turkcell.com.tr icin" once, then "ya gecen ay?" — the domain was
                    # asked for, just not in this sentence.
                    " ".join(
                        f"{getattr(turn, 'prompt', '')} {getattr(turn, 'statement', '')}"
                        for turn in history
                    ),
                )
            )

    async def _call_model(
        self,
        backend: ModelBackend,
        definition: Definition,
        action: Action,
        values: dict[str, str],
        request: str = "",
        history: Sequence[Any] = (),
        summary: str = "",
        refused: Sequence[str] = (),
    ) -> Authored:
        """
        One request to whichever model the definition names.

        The definition's own system prompt is used verbatim, with the action's guardrails
        appended, so the operator's instructions stay the primary influence and the
        limits are stated to the model as well as enforced afterwards.

        The prompt is not tuned per provider. A capable model and a small local one get
        the same instructions, because the instructions describe the task rather than
        coax a particular model, and a per provider variant would be one more thing to
        keep in step every time an action's options change.
        """
        model = definition.model_identifier or FALLBACK_MODEL

        result = await backend.author(
            model=model,
            system=self._system_prompt(definition, action),
            user=self._user_prompt(action, values, request, history, summary, refused),
            schema=_AuthoredCommand if action.kind == "ssh" else _AuthoredQuery,
            # The operator's own settings, from the model they chose in the panel. Until
            # these were carried, a temperature set there reached nothing that made a
            # request, and the model sampled at its provider's default.
            options=ModelOptions.of(definition),
        )

        if not result.ok:
            logger.warning(
                "Model %s (%s) did not author for action %s: %s",
                model, definition.model_provider or "anthropic", action.id, result.error,
            )
        return result

    # ------------------------------------------------------------------ prompts

    def _system_prompt(self, definition: Definition, action: Action) -> str:
        parts = [definition.system_prompt.strip()]
        config = action.config

        # A model has no clock, and it will not say so: asked for "the last five runs" it
        # invented a 2023 date range against 2026 data and returned nothing, successfully.
        # An empty answer to a question with an answer is the worst shape a failure takes.
        # Deliberately just the date. Adding "and as a Unix timestamp that is 1788175352"
        # made it worse rather than better: the number became something to use, and the
        # next three queries all carried an `inserttime >= <now>` nobody had asked for.
        parts.append(
            f"Today's date is {datetime.now(UTC).date().isoformat()} (UTC). Use it for any "
            "relative period the request actually asks for."
        )

        if action.kind == "ssh":
            if unrestricted(config.allowed_commands):
                parts.append(
                    "You are writing a single shell command. This action accepts any "
                    "command, so write the one the request actually needs rather than "
                    "bending it towards a familiar one."
                )
            else:
                parts.append(
                    "You are writing a single shell command. It must start with one of "
                    f"these prefixes: {', '.join(config.allowed_commands) or 'none configured'}."
                )
            # Told as well as enforced. A command refused afterwards costs a round trip
            # and reaches the operator as "rejected" — the reason is in the log, and the
            # thing they wanted is not on the screen.
            parts.append(
                "One command, and only one. Do not chain with ; or &&, do not pipe, do "
                "not redirect with > or <, and do not substitute with $( ) or backticks. "
                "Anything after such a mark is a second command, and only the first one "
                "was permitted.\n\n"
                "Most requests that look like two commands are one. Installing two "
                "packages is a single install with both names on it, and asking about "
                "two services is a single status call. Look for the one command that "
                "does the whole thing before concluding you need two.\n\n"
                "Use the package manager the host actually has. Do not assume one from "
                "habit: the operator's guidance below, and the request itself, say which "
                "family of system this is.\n\n"
                "If the request genuinely needs two commands, write the first one only. "
                "It will run, its result will come back, and the second can be asked for "
                "then. Half the work done is worth more than a command refused."
            )
            if config.blocked_patterns:
                parts.append(
                    "It must never contain any of: " + ", ".join(config.blocked_patterns) + "."
                )
            if config.command_guidance:
                parts.append(config.command_guidance)
        else:
            parts.append(
                "You are writing a single SQL statement. Permitted statement types: "
                f"{', '.join(config.allowed_operations) or 'none configured'}."
            )
            if schema := self._schema_for(config):
                parts.append("Schema you may use:\n" + schema)
            if config.guidance:
                parts.append(config.guidance)
            # Every rule the model kept breaking, together and last, because that is where
            # a model weighs an instruction most. Each line here is one query that came
            # back wrong from a live database, not a precaution.
            rules = [
                # Invented on its own: `status = 'active'`, and a cut-off timestamp that
                # had not happened yet. Nobody asked for either, and the answer was empty.
                "Write only what was asked for. Do not add a condition the request does "
                "not call for — no status filter, no time range, no cut-off of your own.",
                # `<= '2026-08-31'` means midnight, so everything that happened that day
                # was excluded. Returned nothing, successfully.
                "A bare date literal means midnight at the start of that day. Write an "
                "inclusive upper bound as `< the following day`, never as `<= that day`.",
                # Written by habit, and rejected by the guardrail afterwards — a correct
                # outcome by way of a useless one, since the operator only sees "rejected".
                "Write literal values into the statement. Do not use bind parameters such "
                "as $1 or :name — nothing supplies them, and the statement would fail.",
                # The action was called "Yeni sorgu", its purpose "Bireysel yaani db", and
                # the definition's own prompt began "Bireysel ortamdaki local database".
                # Asked how many accounts there were, the model took "bireysel" out of
                # those labels and wrote `where domain = 'bireysel'` — against a column
                # that happened to exist. It answered 0 where the answer was 585, and
                # reported success. A label naming an environment is the likeliest thing
                # in the prompt to be mistaken for a value, because it reads like one.
                "The action's name, its purpose, the schema and the notes above say where "
                "the statement runs. They are not part of the request. Never turn a word "
                "from them into a value, a filter or a column name.",
            ]

            if config.max_rows:
                # "Return at most 100 rows" was read as "return 100": asked for three, it
                # wrote LIMIT 100. The ceiling and the request are different numbers.
                rules.append(
                    f"Never return more than {config.max_rows} rows. If fewer were asked "
                    "for, limit to that smaller number instead."
                )

            parts.append("\n".join(rules))

        parts.append(
            "Produce only the command or query itself. Do not wrap it in a code fence "
            "and do not chain several statements."
        )
        return "\n\n".join(part for part in parts if part)

    def _schema_for(self, config: Any) -> str:
        """
        What the model is told about the database.

        The read schema first, because it is what the database actually says; a hand
        written hint can be months out of date and the model has no way to tell. Both are
        given when both exist: the read one has the columns, the written one usually has
        what they mean, and the second is the part a model cannot work out for itself.
        """
        generated = (config.generated_schema or "").strip()
        written = (config.schema_hint or "").strip()

        if generated and written:
            return f"{generated}\n\n-- notes from the operator:\n{written}"
        return generated or written

    def _user_prompt(
        self,
        action: Action,
        values: dict[str, str],
        request: str = "",
        history: Sequence[Any] = (),
        summary: str = "",
        refused: Sequence[str] = (),
    ) -> str:
        """
        The request, plus the inputs.

        Secrets are omitted rather than masked: the model has no use for them, and the
        placeholder is substituted after generation anyway.
        """
        lines: list[str] = []

        # Before the request, because a follow-up refers backwards and the model has to
        # have read what it refers to. The console kept a thread and looked like a chat,
        # but every prompt arrived alone: "peki ya turkcell.com.tr icin?" had no
        # antecedent, and the model wrote a statement for a question nobody had asked.
        if thread := self._thread(history, summary):
            lines.append(thread + "\n")

        # Then the question. Everything below it is context for answering it, and for a
        # while it was all the model got.
        if request:
            lines.append(f"The request, in the words it was made in:\n{request}\n")

        # Marked as context, and it has to be. Bare "Action: …" and "Purpose: …" read like
        # more of the request: a model asked how many accounts there were, against an
        # action whose purpose was "Bireysel yaani db", filtered on `domain = 'bireysel'`
        # and returned nothing. These lines name where the work happens, not what to do.
        lines.append(
            "Where this runs. Naming, not instruction — nothing below is part of the "
            "request:"
        )
        lines.append(f"  Action: {action.name}")
        if action.description:
            lines.append(f"  Purpose: {action.description}")

        if values:
            lines.append("")
            lines.append("Inputs available as placeholders:")
            lines.extend(f"  {{{{{key}}}}} = {value}" for key, value in sorted(values.items()))
        lines.append("")

        lines.append(
            "Write the command or query. You may reference an input by its placeholder "
            "instead of its literal value."
        )

        # Last, because it is about the attempt that just failed and the model weighs the
        # end of a prompt most. Phrased as what happened rather than as a rule: a rule was
        # already given and this is the evidence that it was not followed.
        if refused:
            lines.append(
                "\nYour previous attempt was refused:\n"
                + "\n".join(f"- {reason}" for reason in refused)
                + "\nWrite one that does not have that problem. If the request seems to "
                "need two commands, do the first part only."
            )

        return "\n".join(lines)

    @staticmethod
    def _thread(history: Sequence[Any], summary: str = "") -> str:
        """
        The statements this conversation has already produced.

        Oldest first, and capped: a follow-up nearly always varies the last thing that ran,
        and a long thread would push the schema out of the model's attention to answer a
        question about the previous sentence.

        The statements are here and the results are not. "The same but for last month" is
        answered by seeing the query; it is not answered any better by seeing its rows, and
        putting a result set into every subsequent prompt would copy production data into
        somewhere new for nothing.
        """
        recent = [
            turn for turn in history
            if getattr(turn, "prompt", "") or getattr(turn, "statement", "")
        ][-_THREAD_LIMIT:]

        lines: list[str] = []

        # Before the turns, because it covers what happened before them. A summary at the
        # end would read as the most recent thing said, which is the opposite of true.
        if summary:
            lines.append(f"Earlier in this conversation, in summary:\n{summary}\n")

        if not recent:
            return "\n".join(lines)

        lines.append("Then, oldest first:" if summary
                     else "Earlier in this conversation, oldest first:")
        for turn in recent:
            if getattr(turn, "prompt", ""):
                lines.append(f"- asked: {turn.prompt}")
            if getattr(turn, "statement", ""):
                lines.append(f"  which ran: {turn.statement}")

        lines.append(
            "The request below may continue one of these — the same question for a "
            "different value, or a narrowing of the last answer. Read it that way when it "
            "does not stand on its own."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ helpers

    def _targets(self, action: Action, values: dict[str, str]) -> list[str]:
        """Hosts an SSH action would reach, mirroring the gateway's resolution."""
        if action.kind != "ssh":
            return []

        mode = action.config.target_mode
        if mode == "group":
            # The gateway already counted them; it does not send the member list with
            # the definition, so the count is reported rather than invented.
            count = action.resolved_target_count
            group = action.host_group_name or "group"
            return [f"<{count} host(s) in {group}>"] if count else []
        if mode == "list":
            return [render(host, values) for host in action.config.hosts]
        if mode == "single" and action.config.host:
            return [render(action.config.host, values)]
        return []

    def _render_rest(
        self, action: Action, values: dict[str, str], masked: set[str]
    ) -> str:
        config = action.config
        line = f"{config.method or 'GET'} {render_url(config.url, values, mask=masked)}"

        headers = [
            f"{header.get('key', '')}: {render(header.get('value'), values, mask=masked)}"
            for header in config.headers
            if header.get("key")
        ]
        body = render_body(config.body, values, mask=masked) if config.method != "GET" else ""

        return "\n".join(filter(None, [line, *headers, "", body])).strip()

    def _note_unresolved(
        self, definition: Definition, action: Action, values: dict[str, str],
        planned: PlannedAction,
    ) -> None:
        """
        Reports the placeholders this action cannot fill, and says which kind they are.

        Two very different problems used to share one sentence. A placeholder nobody
        declared is a mistake in the definition, fixed by adding an input. A declared one
        with no value is a mistake in the request — or in nothing at all, if it was
        optional — and adding an input would not help.

        Being told "undefined input: email" about an input plainly listed in the editor
        sends the reader to look for a typo that is not there.
        """
        config = action.config

        # The URL is rendered first rather than checked as a template. A query parameter
        # nobody filled in is dropped, and what is dropped is not missing: an optional
        # filter that was not asked for is the ordinary case, not a broken definition.
        # Everything else — the path included — is still held to the template.
        templates: list[str | None] = [
            render_url(config.url, values),
            render_body(config.body, values),
            config.command, config.query, config.host,
        ]
        templates.extend(header.get("value") for header in config.headers)
        templates.extend(config.hosts)

        missing = find_unresolved(templates, values)
        if not missing:
            return

        declared = {item.key for item in definition.inputs}

        undeclared = [key for key in missing if key not in declared]
        unfilled = [key for key in missing if key in declared]

        if undeclared:
            planned.rejected_reasons.append(
                "Template refers to undefined input(s): " + ", ".join(undeclared)
            )

        if unfilled:
            planned.rejected_reasons.append(
                "No value for input(s): " + ", ".join(unfilled)
                + ". They are declared but nothing supplied one, and the template needs "
                "them where they are used"
            )
