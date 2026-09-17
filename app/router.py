"""
Chooses a tool for a prompt.

The step before planning. A user types what they want; something has to decide which of
the published tools that is, and with what arguments. That decision is a model's, made
against the catalogue's own JSON Schemas — the same schemas an MCP client would see, so
the panel is choosing from exactly what a real caller could call.

Choosing nothing is a first-class answer. A prompt that matches no tool gets said so,
with the reason, rather than being forced into the nearest match — the nearest match to
"is the database up?" might be a tool that restarts it.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.catalogue import Catalogue
from app.config import Settings
from app.models import Definition
from app.providers import FALLBACK_MODEL, BackendRegistry

logger = logging.getLogger(__name__)

#: How many earlier turns the model is shown.
#:
#: A follow-up nearly always continues the last thing said. Sending the whole thread would
#: push the catalogue out of the model's attention to answer a question about the previous
#: sentence.
_THREAD_LIMIT = 6


class _Choice(BaseModel):
    """What the model decided, in the shape it is asked to answer in."""

    tool_name: str = Field(
        default="",
        description="Exactly one tool name from the catalogue, or empty if none fits.",
    )
    arguments_json: str = Field(
        default="{}",
        description="A JSON object of arguments for that tool, matching its input schema.",
    )
    reasoning: str = Field(
        default="",
        description="One sentence on why this tool, or why none of them.",
    )


class Routed(BaseModel):
    """The outcome of routing a prompt."""

    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""

    #: Set when no tool was chosen, or when the choice could not be used.
    problem: str | None = None

    #: An answer from the model itself, when no tool was the right thing to call.
    #:
    #: Not everything asked in a console is a database question. "What did I just run?",
    #: "why did that return nothing", "explain this query" are all answerable from the
    #: conversation, and answering "no published tool matches this request" to them made
    #: the console feel broken for exactly the questions a person asks in one.
    #:
    #: Carried separately from a plan, and it has to be: this is a model talking, not a
    #: system reporting. Presenting the two the same way would let a guess about how many
    #: accounts there are be read as a count.
    answer: str = ""

    @property
    def chosen(self) -> bool:
        return self.tool_name is not None


class _Arguments(BaseModel):
    """Arguments for a tool the caller already chose."""

    arguments_json: str = Field(
        default="{}",
        description="A JSON object matching the tool's input schema.",
    )
    reasoning: str = Field(
        default="",
        description="One sentence on how the arguments were read from the request.",
    )


class _Reply(BaseModel):
    """An answer the model gives on its own, with no tool behind it."""

    answer: str = Field(
        default="",
        description=(
            "The answer, or an explanation of what would be needed to give one. "
            "Empty if the question cannot be answered without running something."
        ),
    )


#: How a statement begins, for telling a written query from a request for one.
_SQL_OPENING = re.compile(r"^\s*(select|with|insert|update|delete|show|describe)\b", re.I)


def _field_values(name: str, answers: str) -> list[str]:
    """
    Every value the answers give for a field of this name, in the order they appear.

    Matched as a field rather than as text. Checking that the value merely *appears*
    somewhere is no check at all for a short one: asked for an id out of a list of Yigits,
    this model answered 1, and "1" appears in any body long enough to have a digit in it.
    The step was a DELETE against a record that matched nothing anybody asked for.
    """
    pattern = re.compile(
        r'"' + re.escape(name) + r'"\s*:\s*(?:"([^"]*)"|([^,}\]\s]+))'
    )

    # `quoted or bare`, not a None check: findall gives an unmatched group the empty
    # string, so asking whether it is None picks the empty half every time.
    return [
        (quoted or bare).strip()
        for quoted, bare in pattern.findall(answers)
        if (quoted or bare).strip()
    ]


def _record_with(name: str, value: str, answers: str) -> str:
    """
    The object in the answers where this field has this value.

    Fields were being read one at a time, each from wherever in the text it happened to
    appear first, and the result described nobody: `id` 86 arrived with the `last_name` and
    `email` of 69, the record that had just been deleted. Substituted into a search that
    took all three, it matched nothing and the goal ended there with two users left.

    A value is only meaningful with the others it was written beside. This finds the braces
    around one field and returns what is inside them, so the rest are read from the same
    record or not at all.
    """
    match = re.search(
        r'"' + re.escape(name) + r'"\s*:\s*(?:"' + re.escape(value) + r'"|'
        + re.escape(value) + r'(?=[,}\]\s]|$))',
        answers,
    )
    if match is None:
        return ""

    # Backwards to the brace this field sits directly inside, counting any object that
    # opened and closed before it — a nested one belongs to a neighbouring field, not to
    # the record being looked for.
    depth = 0
    start = -1
    for index in range(match.start() - 1, -1, -1):
        if answers[index] == "}":
            depth += 1
        elif answers[index] == "{":
            if depth == 0:
                start = index
                break
            depth -= 1

    if start < 0:
        return ""

    depth = 0
    for index in range(start, len(answers)):
        if answers[index] == "{":
            depth += 1
        elif answers[index] == "}":
            depth -= 1
            if depth == 0:
                return answers[start:index + 1]

    # An answer trimmed mid-record. What there is of it is still one record's worth.
    return answers[start:]


def _only_values_that_were_read(
    step: _Step, steps: Sequence[Any], definition: Definition | None = None
) -> _Step:
    """
    Drops any value that is not in the answers it was supposed to be read out of.

    Asked for the id it had found, this model answered `{"id": "For each user found, delete
    them one by one. First, delete user with id 59."}` — the field filled with the sentence
    the instruction was trying to get out of it. Substituted, that becomes a URL nobody
    meant to call.

    The rule is the contract stated plainly: read the value out of the answers. A value
    that appears nowhere in them was not read out of them, whatever else it may be, and
    the step is better off without it — the planner then says the input is missing, which
    is true and legible, rather than acting on a sentence.
    """
    if not step.values:
        return step

    answers = _answers_of(steps)

    # Only inputs the tool actually declares. Asked for the id, this model has answered
    # under the name `user_id` — a plausible name for a field that does not exist, which
    # substituted into nothing and left the real input empty without saying so.
    declared = (
        {item.key for item in definition.inputs if item.key} if definition else None
    )

    kept = [
        item.model_copy(update={"value": item.value.strip()})
        for item in step.values
        if item.name
        and item.value.strip()
        and (declared is None or item.name in declared)
        and item.value.strip() in _field_values(item.name, answers)
    ]

    for item in step.values:
        if item.name and not any(kept_item.name == item.name for kept_item in kept):
            logger.warning(
                "A step value for %r was dropped: not an input of this tool, or not in "
                "the answers it was to be read from", item.name
            )

    return step if len(kept) == len(step.values) else step.model_copy(update={"values": kept})


def _read_from_answers(
    step: _Step, steps: Sequence[Any], wanted: str, waiting_for: str = ""
) -> _Step:
    """
    Takes the waiting input's value out of the answers, without asking anybody.

    The question never needed a model. The step is waiting on `id`, the previous answer has
    an `id` field, and the first one is the one to act on — that is a lookup, and asking for
    it got an empty field two times in three, a sentence once, and once the number 1.

    Only fills what the step left empty, and only what the answers actually say. A model
    that did give a usable value keeps it: it read the same answers, and it may have had a
    reason to pick a later row.

    Asked for every declared input when nothing in particular was waiting, because the plan
    does not always record a deferral — the selection sometimes takes the search alone — and
    the step after it still needs its id from somewhere. An input the answers say nothing
    about is left alone; nothing is invented here.
    """
    if not wanted:
        return step

    answers = _answers_of(steps)
    already = {item.name for item in step.values if item.value.strip()}
    names = [part.strip() for part in wanted.split(",") if part.strip()]

    # One record, chosen before anything is read out of it.
    #
    # The model's own value picks it when there is one: it read these answers and said
    # which row it meant, and "the second user found" is a choice, not a phrasing. Only
    # then the first row carrying what is being waited for.
    anchor = next(
        (item for item in step.values if item.name and item.value.strip()), None
    )
    if anchor is not None:
        record = _record_with(anchor.name, anchor.value.strip(), answers)
    else:
        record = ""
        for name in names:
            values = _field_values(name, answers)
            if values:
                record = _record_with(name, values[0], answers)
                break

    # Without a record, only the one input actually being waited for is filled — from the
    # answers at large. Filling the rest that way is what mixed two people together.
    source = record or answers
    fillable = names if record else [name for name in names if name == waiting_for]

    found = list(step.values)
    for name in fillable:
        if name in already:
            continue

        values = _field_values(name, source)
        if values:
            found.append(_Value(name=name, value=values[0]))

    if len(found) == len(step.values):
        return step

    # Said out loud, because the reason is prose and the value is not always in it. A step
    # whose reason read "processing the first one (id 59)" was filled with id 37 — the model
    # named one record and the lookup found another, because the search it was reading had
    # come back unfiltered. Nothing ran: it was held for approval, and the card showed the
    # command. But the sentence beside it described a different record, which is the worst
    # way for an approval screen to be wrong.
    read = ", ".join(
        f"{item.name}={item.value}" for item in found[len(step.values):]
    )

    return step.model_copy(update={
        "values": found,
        "reason": f"{step.reason} (read from the previous answer: {read})".strip(),
    })


def _unhandled(
    steps: Sequence[Any], definition: Definition | None
) -> list[tuple[str, str]]:
    """
    Values an earlier answer gave that no later statement has acted on.

    Only for a field already being used that way: at least one of its values must appear in
    a statement, and at least one must not. That is what tells an identifier from anything
    else — every row of a search for "Gizem" has `first_name` Gizem and the search itself
    names it, so nothing is outstanding there; the `id` column has two of three spent, so
    the third stands out. A field with none of its values in any statement — `email`, here —
    is not being acted on at all and says nothing about what is left.
    """
    if definition is None:
        return []

    answers = _answers_of(steps)
    acted = " ".join(str(getattr(item, "statement", "") or "") for item in steps)

    left: list[tuple[str, str]] = []
    for item in definition.inputs:
        if not item.key:
            continue

        values = _field_values(item.key, answers)
        spent = [value for value in values if value in acted]

        if not spent or len(spent) == len(values):
            continue

        left.extend((item.key, value) for value in values if value not in acted)

    return left


def _still_to_do(
    step: "_Step", steps: Sequence[Any], definition: Definition | None
) -> "_Step | None":
    """
    Refuses "done" from a step that is holding a record nothing has touched.

    Asked whether three users named Gizem had been deleted after two of them had, this
    model answered:

        done: true
        reason: "Two users named 'Gizem' have been deleted so far; the third one (id 96)
                 also needs to be deleted to fully meet the goal."
        values: {"id": "96"}

    Its own reason says the goal is not met and its own value names the record left over.
    Twice in three runs. The console showed two approvals for three records and stopped,
    which is exactly the failure the deferral mechanism was built for — and deferrals stop
    being recorded once the selection takes one action at a time.

    Whether a record has been acted on is not a judgement: the statements are right there.
    A value that appears in none of them is work nobody has done.

    The leftovers are worked out here, but *whether they matter* is left to the model. A
    goal can legitimately be "find them and delete the first", and a rule that chased every
    unhandled row would turn that into "delete all of them". So a record is only taken as
    outstanding when the model's own answer names it — in its values, its request, or its
    reason — while claiming to be finished. We do not decide the goal is unfinished; we
    refuse a verdict the same answer contradicts.

    Returns the step to run instead, or ``None`` when "done" is credible.
    """
    left = [
        (name, value) for name, value in _unhandled(steps, definition)
        if value in step.request or value in step.reason
        or any(item.value.strip() == value for item in step.values)
    ]

    if not left:
        return None

    named = ", ".join(f"{name} {value}" for name, value in left)
    logger.info("A step said it was done while holding %s; continuing", named)

    # Written here rather than taken from the model. Having decided it was finished, it
    # words the request as a report — "the users named Gizem have been deleted" — and
    # routing that produces a listing, not the operation the goal asked for.
    return step.model_copy(update={
        "done": False,
        "request": (
            f"The goal is not met yet. Carry it out for the record with {named}, "
            f"which no step so far has acted on."
        ),
        "reason": (
            f"{step.reason} — but {named} appears in no step taken so far, "
            f"so the goal is not finished."
        ).strip(" —"),
    })


def _nothing_to_act_on(wanted: str) -> "_Step":
    """
    The goal ends here, because what it was waiting on is not in the answers.

    A search that matched nothing is an answer. Asked to find the Yigits and delete them,
    where the name was typed "Yiğit" and the records read "Yigit", the search returned zero
    rows and the goal proposed the delete anyway — with the id left as `{{id}}`, because
    there was no id to put there. The guardrails refused it, and what the operator saw was
    a failed DELETE rather than "nobody by that name".

    Saying so is both truthful and safer: an unfillable write is not a step waiting to be
    approved, it is a step that cannot exist.
    """
    return _Step(
        done=True,
        reason=(
            f"The answers so far give no {wanted}, so there is nothing to act on. "
            f"A search that matched nothing ends the goal here."
        ),
    )


def _answers_of(steps: Sequence[Any]) -> str:
    """Everything the steps so far have printed, as one body to read values out of."""
    return "\n".join(
        str(getattr(item, "output", "") or "") + "\n" + str(getattr(item, "sample", "") or "")
        for item in steps
    )


def _input_names(definition: Definition | None) -> str:
    """The tool's input keys, for a model that may be able to fill one from an answer."""
    if definition is None:
        return ""

    return ", ".join(item.key for item in definition.inputs if item.key)


def _in_words(step: "_Step") -> "_Step":
    """
    Keeps a step from arriving already written as SQL.

    Told not to, and shown the statements the earlier steps produced, this model writes the
    next step as a query anyway — the format in front of it wins over the instruction. A
    step that arrives written skips the planner, which is the only part of the chain with
    the schema, and it reads as SQL in a console where every other turn is a sentence.

    The reason is prose describing the same step, so it stands in. When it is not usable
    the query is kept: a step in the wrong shape still runs, and losing it would cost more
    than the shape does.
    """
    if not step.request or not _SQL_OPENING.match(step.request):
        return step

    if step.reason and not _SQL_OPENING.match(step.reason):
        logger.info("A step arrived as SQL; using its reason as the request instead")
        return step.model_copy(update={"request": step.reason})

    return step


class _Value(BaseModel):
    """One input a step needs, and what it is.

    A list of these rather than a free-form object. Asked for `{"id": "59"}` as a mapping,
    this model returned an empty one every time — an open dictionary is the shape that gets
    left out. Named fields it fills.
    """

    name: str = Field(default="", description="The input's name, such as id.")
    value: str = Field(
        default="",
        description=(
            "The value, copied out of the answer exactly as it appears there. A value, not "
            "a sentence about it: 59, not 'delete the user with id 59'."
        ),
    )


class _Step(BaseModel):
    """The next thing to ask, or the news that there is nothing left to ask."""

    done: bool = Field(
        default=True,
        description="True when the goal has been met by the steps already taken.",
    )
    request: str = Field(
        default="",
        description=(
            "The next sub-request, written as a sentence, as though somebody had typed "
            "it. Empty when done."
        ),
    )
    reason: str = Field(
        default="",
        description="One sentence on why this step, or why the goal is met.",
    )
    values: list[_Value] = Field(
        default_factory=list,
        description=(
            "The inputs this step needs, read out of the answers so far. One entry per "
            "input. Empty when the step needs none."
        ),
    )

    def as_dict(self) -> dict[str, str]:
        """The values keyed by name, which is how everything downstream wants them."""
        return {item.name: item.value for item in self.values if item.name and item.value}


class _Summary(BaseModel):
    """A conversation folded down to what a later question might refer back to."""

    summary: str = Field(
        default="",
        description="What was asked and what it produced, in a few sentences.",
    )


class PromptRouter:
    """Turns a sentence into a tool call, or into a reason it is not one."""

    def __init__(self, settings: Settings, catalogue: Catalogue, cipher: Any = None) -> None:
        self._settings = settings
        self._catalogue = catalogue
        # Routing uses the ROUTER_* settings rather than a definition's model: nothing is
        # chosen yet when it runs, so there is no definition to take a key from.
        self._backends = BackendRegistry(settings, cipher)

    def _with_thread(self, prompt: str, history: Sequence[Any], summary: str = "") -> str:
        """The request, preceded by what it may be continuing."""
        thread = self._thread(history, summary)
        return f"{thread}\n\nThe request:\n{prompt}" if thread else prompt

    @staticmethod
    def _thread(history: Sequence[Any], summary: str = "") -> str:
        """
        What was asked before, as the model should read it.

        Oldest first, because that is the order it happened in and a follow-up refers
        backwards. Capped: a long session would otherwise push the catalogue itself out of
        the model's attention, and a follow-up almost always continues the last thing said
        rather than something twenty turns ago.
        """
        recent = [turn for turn in history if getattr(turn, "prompt", "")][-_THREAD_LIMIT:]

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
            lines.append(f"- asked: {turn.prompt}")
            if getattr(turn, "tool_name", None):
                lines.append(f"  answered with: {turn.tool_name}")
            if getattr(turn, "statement", ""):
                lines.append(f"  which ran: {turn.statement}")

        return "\n".join(lines)

    async def arguments_for(
        self,
        definition: Definition,
        prompt: str,
        history: Sequence[Any] = (),
        summary: str = "",
    ) -> Routed:
        """
        Fills one named tool's arguments from a prompt.

        Used when the caller has already chosen the tool, which is a different and much
        smaller question than choosing one: there is nothing to get wrong about *which*
        thing runs, only about what it is given.

        A tool with no inputs skips the model entirely. Asking a model to produce an empty
        object costs a round trip and introduces a way for it to fail.
        """
        fillable = [item for item in definition.inputs if item.source == "prompt"]

        if not fillable:
            # Nothing here is the model's to decide: every input is either fixed or the
            # caller's. Asking anyway costs a round trip and can only introduce a guess.
            return Routed(tool_name=definition.tool_name)

        backend = self._backends.for_definition(
            self._settings.router_provider, self._settings.router_endpoint
        )
        if backend is None:
            return Routed(
                tool_name=definition.tool_name,
                problem=self._backends.unavailable_reason(
                    self._settings.router_provider, self._settings.router_endpoint
                ),
            )

        result = await backend.author(
            model=self._settings.router_model or FALLBACK_MODEL,
            system=(
                "You fill in the arguments for one tool the user has already chosen.\n\n"
                f"Tool: {definition.tool_name}\n"
                f"Description: {definition.tool_description}\n"
                f"Input schema:\n{json.dumps(definition.prompt_schema(), indent=2)}\n\n"
                "Read the values from the request. Leave an argument out rather than "
                "inventing one: an omitted argument falls back to the input's own default, "
                "and a guessed one silently changes what runs."
            ),
            user=self._with_thread(prompt, history, summary),
            schema=_Arguments,
        )

        if not result.ok or not isinstance(result.parsed, _Arguments):
            return Routed(
                tool_name=definition.tool_name,
                problem=result.error or "The model produced no arguments",
            )

        return self._read_arguments(definition.tool_name, result.parsed)

    def _read_arguments(self, tool_name: str, parsed: _Arguments) -> Routed:
        try:
            arguments = json.loads(parsed.arguments_json or "{}")
        except json.JSONDecodeError:
            return Routed(
                tool_name=tool_name,
                reasoning=parsed.reasoning,
                problem="The model's arguments were not valid JSON; none were used",
            )

        if not isinstance(arguments, dict):
            return Routed(
                tool_name=tool_name,
                reasoning=parsed.reasoning,
                problem="The model's arguments were not an object; none were used",
            )

        return Routed(tool_name=tool_name, arguments=arguments, reasoning=parsed.reasoning)

    async def next_step(
        self,
        goal: str,
        steps: Sequence[Any] = (),
        history: Sequence[Any] = (),
        summary: str = "",
        limit: int = 5,
        definition: Definition | None = None,
        kind: str = "db",
        waiting_for: str = "",
    ) -> _Step:
        """
        Decomposes a goal into one more question, or says the goal is met.

        Asked once per step rather than all at once, because a step is often only
        answerable after the one before it: "list the fees, then total them" needs the
        columns of the first result before the second can be written.

        What comes back is a **sentence**, not a plan. The step then goes through the
        ordinary planning path — the same guardrails, the same dynamic SQL, the same
        masking — so the loop adds a decision and no new way to reach a database.

        A goal that is met returns ``done``. So does one this cannot make progress on: a
        loop that cannot say "I am finished" runs until its cap and calls that an answer.
        """
        if len(steps) >= limit:
            return _Step(done=True, reason=f"Stopped at the {limit} step limit")

        backend = self._backends.for_definition(
            self._settings.router_provider, self._settings.router_endpoint
        )
        if backend is None:
            return _Step(done=True, reason="No model is configured to plan a next step")

        taken = "\n".join(self._describe(index, step) for index, step in enumerate(steps, 1))

        # The tables it may name. Without them it invents one — asked to split a goal about
        # accounts it proposed a step against a "personel" table nobody has, which the
        # planner would then have had to refuse, or worse, guess at.
        tables = definition.reaches() if definition is not None else []
        reaches = f"Tables this tool reaches: {', '.join(tables)}\n\n" if tables else ""

        result = await backend.author(
            model=self._settings.router_model or FALLBACK_MODEL,
            system=(
                self._waiting_rules(waiting_for) if waiting_for
                else self._shell_rules(limit) if kind == "ssh"
                else self._call_rules(limit, _input_names(definition)) if kind == "rest"
                else self._query_rules(limit, reaches)
            ),
            user=(
                f"{self._with_thread('', history, summary)}\n\n"
                f"The goal:\n{goal}\n\n"
                + (f"Steps taken so far:\n{taken}" if taken else "No steps taken yet.")
            ),
            schema=_Step,
        )

        if not result.ok or not isinstance(result.parsed, _Step):
            # Stopping is the safe failure. A loop that cannot decide keeps going, and a
            # step nobody chose is worse than a goal left half met.
            logger.warning("A next step could not be decided: %s", result.error)
            return _Step(done=True, reason=result.error or "The model produced no step")

        checked = _only_values_that_were_read(result.parsed, steps, definition)

        if checked.done:
            # Nothing is filled in for a step that says there is no next step: the values
            # exist to be substituted into one, and adding them here would manufacture the
            # very contradiction the next line looks for.
            return _still_to_do(checked, steps, definition) or checked

        wanted = waiting_for or _input_names(definition)
        filled = _read_from_answers(checked, steps, wanted, waiting_for)

        if waiting_for and not filled.as_dict().get(waiting_for):
            return _nothing_to_act_on(waiting_for)

        return _in_words(filled)

    @staticmethod
    def _query_rules(limit: int, reaches: str) -> str:
        """The step rules for a goal made of queries."""
        return (
                "You break one goal into the smallest number of read-only steps that meet "
                "it, one step at a time.\n\n"
                "You are given the goal and the steps already taken, each with the query "
                "that ran and a summary of what came back — the row count, the column "
                "names and the first few rows. You do not get the full result, and you do "
                "not need it: your job is to decide what to ask next, not to do the "
                "arithmetic yourself.\n\n"
                "Write the next step in words, as a person would type it — never as "
                "SQL. Writing the query is the planner's job and it has the schema; a "
                "step that arrives already written skips the one part of this that knows "
                "the columns.\n\n"
                "It "
                "goes to the same planner that answered the steps above, so it must stand "
                "on its own — \"the same as before\" means nothing to a reader who was "
                "not there.\n\n"
                "Use only the tables listed below and the words of the goal itself. You "
                "are not shown the schema and you must not invent a table, a column or a "
                "value: the planner has the schema and will write the query. Naming a "
                "table nobody has is how a step becomes a query against nothing.\n\n"
                + reaches
                + "Say done as soon as the steps taken meet the goal — and not before. "
                "A goal that names two things needs two steps: \"ali and veli as two "
                "separate tables\" is not met by one query returning both, because two "
                "tables is what was asked for. A goal that one query does answer must "
                "not be split.\n\n"
                "Say done also when the steps have stopped making progress: repeating a "
                "query that already ran answers nothing and looks exactly like an "
                "answer.\n\n"
                f"Never propose more than {limit} steps in total, and never propose "
                "anything that writes, changes or deletes. These steps only read."
        )

    @staticmethod
    def _waiting_rules(waiting_for: str) -> str:  # noqa: D401
        """
        The rules when the plan already knows there is more to do.

        A different question from the others, and deliberately so. "Is the goal met?" is a
        judgement, and asked it twice about "find the user and delete them" this model
        answered done — with a reason describing the deletion it had not performed. Here
        the plan has answered that part: an action was set aside because it needed
        something only an earlier one could supply, and that something has now arrived.

        So there is nothing left to judge. What is left is to read the value out of the
        answer and write the request that was waiting for it.
        """
        return (
            "An action of this tool is ready to run except for one thing, and you are being "
            f"asked for that thing: {waiting_for}. The steps below have run and their "
            "answers are in front of you.\n\n"
            f"Put it in `values`, keyed by name: {{\"{waiting_for.split(', ')[0]}\": "
            "\"59\"}}. Copied out of the answer exactly as it appears there, with no words "
            "around it — a value, not a sentence about the value. That field is the answer; "
            "which action runs and which tool it belongs to are already decided.\n\n"
            "This used to be asked for as a sentence, and the sentence came back as a "
            "description of the situation — \"the search returned several; processing the "
            "first one (id 59) now\" — with the value inside it and no request to act on. "
            "There is no sentence to get wrong now: read the value, and give the value.\n\n"
            "If the answers hold several, give one — the first. What is left is asked for "
            "again after this has run, one at a time.\n\n"
            "`request` is a short line for the person reading the history, not an "
            "instruction to anybody: \"delete the user with id 59\" is a good one.\n\n"
            "Say done, with `values` empty, only if the answers show there is nothing to "
            "act on — a search that found no one leaves nothing to delete — or if a step "
            "failed, in which case say so rather than inventing a value that is not there."
        )

    @staticmethod
    def _call_rules(limit: int, inputs: str = "") -> str:
        """
        The step rules for a goal made of HTTP calls.

        The kind this was really wanted for. "Find the user and add them if they are not
        there" cannot be decided in advance — whether the second call happens at all
        depends on what the first one answered — which is the one thing a plan cannot
        express and a loop can.
        """
        return (
            "You break one goal into the smallest number of HTTP calls that meet it, one "
            "at a time. Each step is one request to one endpoint.\n\n"
            "You are given the goal and the calls already made, each with the request that "
            "went out and what came back.\n\n"
            "Write the next step in words, as a person would type it — never as a URL or a "
            "method. Choosing the endpoint is the planner's job: it has the tool's actions "
            "and knows which one does what.\n\n"
            "Read the answer before deciding. A search that returned the record means "
            "there is nothing to create; a search that returned none means there is. That "
            "is the whole reason this is asked after each step rather than all at once — "
            "if you find yourself guessing what the last call returned, say done "
            "instead.\n\n"
            "These steps may change data — creating and updating are what they are for — "
            "and a person approves each one that does before it runs. Ask for what the "
            "goal asked for and nothing beyond it: reading a record is not a reason to "
            "update it, and one that already exists is not a reason to make another.\n\n"
            "Say done as soon as the calls made meet the goal — and not before. The goal's "
            "own words say what has to happen: \"find the user and delete them\" is two "
            "things, and finding them is not both. If your reason for saying done "
            "describes something still to do, it is not done — that sentence is the next "
            "step, so write it as one.\n\n"
            "Say done when a call failed: a different endpoint after an error is guessing, "
            "and guessing against somebody's data is how one bad step becomes five.\n\n"
            + (
                "The tool takes these inputs: " + inputs + ". If an answer above gives you "
                "one of them and the step needs it, put it in `values` keyed by name.\n\n"
                "A value, not a sentence about it. `{\"id\": \"59\"}` — copied out of the "
                "answer exactly as it appears there, no words around it. `{\"id\": \"delete "
                "the first user, id 59\"}` is not a value; substituted, it becomes a URL "
                "nobody meant to call, and it is dropped.\n\n"
                "One value per input, and one step at a time. If four match, this step is "
                "the first of them; the rest are asked for again after it has run.\n\n"
                if inputs else ""
            )
            + f"Never propose more than {limit} steps in total."
        )

    @staticmethod
    def _shell_rules(limit: int) -> str:
        """
        The step rules for a goal made of commands on a server.

        Separate from the query rules because almost every sentence in those is wrong here.
        A shell goal exists to change a machine — "install apache and start it" is two
        writes and nothing else — so the read-only rule cannot carry over. What replaces it
        is not permission to do anything: a person approves each of these before it runs,
        and a step that reads before it writes is the one worth proposing.
        """
        return (
            "You break one goal into the smallest number of shell steps that meet it, one "
            "step at a time. Each step runs a single command on a server.\n\n"
            "You are given the goal and the steps already taken, each with the command "
            "that ran and what it printed.\n\n"
            "Write the next step in words, as a person would type it — never as a shell "
            "command. Writing the command is the planner's job: it knows the host, its "
            "package manager and what the operator allowed, and a step that arrives "
            "already written skips all of it.\n\n"
            "One command's worth of work per step. Installing a package and starting the "
            "service it provides are two steps, because they are two commands; installing "
            "two packages is one, because one install takes both names.\n\n"
            "Write the step as one action, not as a sentence with \"and then\" in it. A "
            "step reading \"finish the installation and then start the service\" is two "
            "things again, and the planner answering it picks whichever it reads first — "
            "which was the install that had already run. Say only the part that has not "
            "happened yet: \"start the apache service\".\n\n"
            "Never restate work the steps above already did. They ran; repeating one wastes "
            "the step and looks exactly like progress.\n\n"
            "The step goes to the same planner that answered the steps above, so it must "
            "stand on its own — \"now start it\" means nothing to a reader who was not "
            "there. Name the thing.\n\n"
            "These steps may change the machine — that is what installing and starting "
            "are — and a person approves each one before it runs. Propose the work the "
            "goal asked for and nothing beyond it: no tidying up, no hardening, no "
            "restarting anything the goal did not name. Never propose deleting data, "
            "reformatting a disk, or rebooting.\n\n"
            "Say done as soon as the steps taken meet the goal — and not before. A goal "
            "that says install and start is not met by the install alone.\n\n"
            "Say done also when a step has failed, or when the steps have stopped making "
            "progress: guessing a different command after one did not work is how a goal "
            "becomes five commands nobody asked for.\n\n"
            f"Never propose more than {limit} steps in total."
        )

    @staticmethod
    def _describe(index: int, step: Any) -> str:
        """
        One step, as the model reads it: what was asked and what shape came back.

        The sentence, not the SQL. Shown the statements, this model mirrored them and
        returned its next step already written as a query — skipping the one part of the
        chain that has the schema. It works in intents; the planner works in SQL.
        """
        lines = [f"{index}. asked: {getattr(step, 'request', '') or '(nothing)'}"]

        if getattr(step, "statement", ""):
            lines.append(f"   which ran: {step.statement}")

        columns = getattr(step, "columns", None)
        if columns:
            lines.append(f"   columns: {', '.join(columns)}")

        rows = getattr(step, "row_count", None)
        if rows is not None:
            lines.append(f"   rows: {rows}")

        sample = getattr(step, "sample", "")
        if sample:
            lines.append(f"   first rows: {sample}")

        output = getattr(step, "output", "")
        if output:
            lines.append(f"   printed: {output}")

        problem = getattr(step, "problem", "")
        if problem:
            lines.append(f"   failed: {problem}")

        return "\n".join(lines)

    async def answer(self, prompt: str, history: Sequence[Any] = (), summary: str = "") -> str:
        """
        Answers from the conversation, when no tool was the right thing to call.

        Reached only after routing found nothing. Plenty of what gets typed into a console
        is not a request to run anything — "what did I just run?", "why did that come back
        empty", "explain that query" — and replying "no published tool matches this
        request" to those made the console useless for the questions people actually ask
        in one.

        The model is told, in as many words, that it has reached nothing and read nothing:
        it has this conversation and what it already knows, and that is all. A question
        needing data it was not given gets said so rather than guessed at, because a guess
        about how many accounts exist is indistinguishable from a count until somebody
        checks — and nobody checks an answer that looks like one.
        """
        backend = self._backends.for_definition(
            self._settings.router_provider, self._settings.router_endpoint
        )
        if backend is None:
            return ""

        result = await backend.author(
            model=self._settings.router_model or FALLBACK_MODEL,
            system=(
                "You are answering inside an operations console, and no tool matched this "
                "request.\n\n"
                "You have not run anything and you cannot. You have this conversation and "
                "what you already know, and nothing else — no database, no server, no "
                "file.\n\n"
                "Answer the question if the conversation or ordinary knowledge is enough: "
                "what was asked before, what has been run so far, what a statement above "
                "does, why something came back empty, what a term means.\n\n"
                "You have the questions and the statements they produced. You do not have "
                "the rows any of them returned — those are not kept here. Asked to list "
                "what has been found so far, list what was asked and what ran, and say "
                "plainly that the rows themselves are not in front of you and can be seen "
                "on each result in the console.\n\n"
                "If the answer would need data you have not been given, say that plainly "
                "and say what would have to be run to get it. Never state a number, a name "
                "or a fact about their systems that is not written above: a guess reads "
                "exactly like an answer, and nobody checks an answer.\n\n"
                "Reply in the language the request was written in. Leave the answer empty "
                "if there is genuinely nothing useful to say."
            ),
            user=self._with_thread(prompt, history, summary),
            schema=_Reply,
        )

        if not result.ok or not isinstance(result.parsed, _Reply):
            return ""

        return result.parsed.answer.strip()

    async def summarise(self, existing: str, turns: Sequence[Any]) -> str:
        """
        Folds turns that have fallen out of the window into the running summary.

        Folding rather than re-reading: the turns that just dropped out go in with the
        summary so far, so the cost is the same on the thousandth question as on the
        fifty-first. Re-summarising the whole conversation every time would grow without
        bound, which is the thing the window exists to prevent.

        Returns the existing summary unchanged when there is nothing to add or no model to
        ask. A conversation that keeps its window and loses its tail is worse than one that
        keeps both, and much better than one that stops answering.
        """
        described = [
            f"- asked: {getattr(turn, 'prompt', '')}"
            + (f"\n  which ran: {turn.statement}" if getattr(turn, "statement", "") else "")
            for turn in turns
            if getattr(turn, "prompt", "") or getattr(turn, "statement", "")
        ]

        if not described:
            return existing

        backend = self._backends.for_definition(
            self._settings.router_provider, self._settings.router_endpoint
        )
        if backend is None:
            logger.warning("No model is configured to summarise a conversation")
            return existing

        result = await backend.author(
            model=self._settings.router_model or FALLBACK_MODEL,
            system=(
                "You keep a running summary of a conversation between an operator and a "
                "system that queries databases and runs commands.\n\n"
                "Fold the new turns into the summary you are given. Keep what a later "
                "question could refer back to: what was asked about, which values were "
                "used, what was established. Drop what nothing can refer to — greetings, "
                "repeated attempts, exact row counts.\n\n"
                "Write it as a few plain sentences, in the language the operator used. "
                "Never invent a result: if a turn's outcome is not stated here, say what "
                "was asked and stop."
            ),
            user=(
                f"The summary so far:\n{existing or '(nothing yet)'}\n\n"
                "New turns to fold in, oldest first:\n" + "\n".join(described)
            ),
            schema=_Summary,
        )

        if not result.ok or not isinstance(result.parsed, _Summary):
            # Keeping the old summary is the safe failure: the window still works, and the
            # gateway will offer the same turns again next time.
            logger.warning("A conversation could not be summarised: %s", result.error)
            return existing

        return result.parsed.summary.strip() or existing

    async def route(self, prompt: str, history: Sequence[Any] = (),
                    summary: str = "", allowed: Sequence[str] = ()) -> Routed:
        """
        Chooses a tool for a sentence.

        ``allowed`` narrows the catalogue to what this caller may actually run. Empty means
        no restriction rather than nothing — an administrator would otherwise be offered an
        empty catalogue, and "may run anything" and "may run nothing" have to look
        different.

        Narrowing here rather than refusing afterwards, and that is not only politeness. On
        the gateway's execute path, planning and dispatch happen inside one call: by the
        time a tool name comes back, the job may already be on the broker. A tool that is
        never offered cannot be chosen, which is the only form of this check that runs
        before the work does.
        """
        tools = self._catalogue.all()

        if allowed:
            permitted = set(allowed)
            tools = [tool for tool in tools if tool.tool_name in permitted]

        if not tools:
            return Routed(problem="The catalogue is empty; there is nothing to call")

        backend = self._backends.for_definition(
            self._settings.router_provider, self._settings.router_endpoint
        )
        if backend is None:
            return Routed(
                problem=self._backends.unavailable_reason(
                    self._settings.router_provider, self._settings.router_endpoint
                )
            )

        # The thread goes in the user message rather than the system prompt: it changes
        # every turn, and a system prompt that changed every turn would defeat any caching
        # a provider does on it.
        result = await backend.author(
            model=self._settings.router_model or FALLBACK_MODEL,
            system=self._system_prompt(tools),
            user=self._with_thread(prompt, history, summary),
            schema=_Choice,
        )

        if not result.ok or not isinstance(result.parsed, _Choice):
            return Routed(problem=result.error or "The model produced no choice")

        return self._interpret(result.parsed, tools)

    def _interpret(self, choice: _Choice, tools: list[Definition]) -> Routed:
        """
        Reads the model's answer, refusing anything the catalogue does not contain.

        A hallucinated tool name is treated as no choice at all. Accepting one would mean
        looking it up, failing, and reporting a 404 about a tool the user never mentioned.
        """
        if not choice.tool_name:
            return Routed(reasoning=choice.reasoning,
                          problem="No published tool matches this request")

        known = {tool.tool_name for tool in tools}
        if choice.tool_name not in known:
            logger.warning("Model chose an unknown tool: %s", choice.tool_name)
            return Routed(
                reasoning=choice.reasoning,
                problem=f"The model chose {choice.tool_name!r}, which is not published",
            )

        try:
            arguments = json.loads(choice.arguments_json or "{}")
        except json.JSONDecodeError:
            return Routed(
                tool_name=choice.tool_name,
                reasoning=choice.reasoning,
                problem="The model's arguments were not valid JSON; none were used",
            )

        if not isinstance(arguments, dict):
            return Routed(
                tool_name=choice.tool_name,
                reasoning=choice.reasoning,
                problem="The model's arguments were not an object; none were used",
            )

        return Routed(
            tool_name=choice.tool_name,
            arguments=arguments,
            reasoning=choice.reasoning,
        )

    def _system_prompt(self, tools: list[Definition]) -> str:
        """
        Describes the catalogue as the model will see it.

        The JSON Schema comes from the same method that serves `tools/list`, so the model
        is choosing from exactly what a real MCP client could call — not a summary of it
        that could drift.
        """
        described = [
            {
                "name": tool.tool_name,
                "description": tool.tool_description,
                # What the tool touches, which is often the only thing that says what it
                # is *for*. A description is written once and tends to repeat the tool's
                # own name; the tables come from the database itself and cannot. Without
                # this, a question about accounts against a tool named
                # "bireysel_local_yaanidb" and described as "Bireysel local yaanidb" was
                # answered "no published tool matches" — correctly, on the evidence given.
                **({"reaches": reaches} if (reaches := tool.reaches()) else {}),
                # The prompt subset, not the full schema: an input the model may not
                # decide should not be described to it as something to fill in.
                "inputSchema": tool.prompt_schema(),
            }
            for tool in tools
        ]

        return (
            "You route a request to one of the tools below, or to none of them.\n\n"
            f"Tools:\n{json.dumps(described, indent=2, ensure_ascii=False)}\n\n"
            "A tool's \"reaches\" lists the tables it can read or the commands it may "
            "run. It is usually a better guide than the description: a question about "
            "accounts belongs to the tool that reaches an accounts table, whatever either "
            "of them is called.\n\n"
            "Choose the single tool that does what was asked. Fill its arguments from the "
            "request, using the input schema; leave an argument out rather than inventing "
            "a value for it.\n\n"
            "A request may be a follow-up: \"and for X?\", \"the same for last month\", "
            "\"that one again\". A follow-up is not a new subject — it continues the turn "
            "before it and belongs to the same tool as that turn. Read the request "
            "together with the conversation above it before deciding whether anything "
            "matches; a sentence that means nothing on its own usually means the previous "
            "question with one value changed.\n\n"
            "A question *about* the conversation is not a request to run anything, and it "
            "outranks the follow-up rule above. What was asked before, what has been run "
            "so far, what a statement above does, why something came back empty, what a "
            "term means — all of these get an empty tool name.\n\n"
            "This includes a request to list, repeat or summarise what has already "
            "happened. \"List the results you have brought so far\" is a question about "
            "this conversation; it is not an instruction to run the last statement again, "
            "and running it again answers nothing while looking like an answer.\n\n"
            "The test is whether what is being asked for is *new* data. Same question, "
            "different value — a tool. A question whose subject is the conversation "
            "itself — no tool.\n\n"
            "If no tool does what was asked, return an empty tool name and say why. Do "
            "not pick the closest one: a request to check something is not a request to "
            "change it, and guessing is worse than saying nothing matched."
        )
