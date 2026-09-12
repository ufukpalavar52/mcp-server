"""
Placeholder resolution.

Action templates carry ``{{key}}`` markers that stand for a dynamic input. Resolution
happens here rather than inside the planner so the rules stay testable on their own and
identical for every action kind.
"""

from __future__ import annotations

import json

import re
from typing import Any

from app.models import DefinitionInput

PLACEHOLDER = re.compile(r"\{\{\s*([\w.-]+)\s*\}\}")

#: Input types whose value is a secret and must never appear in a rendered template.
SECRET_TYPES = frozenset({"password"})


class UnresolvedPlaceholderError(Exception):
    """Raised when a template refers to a key no input provides."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        super().__init__(f"Unresolved placeholders: {', '.join(sorted(keys))}")


def build_values(
    inputs: list[DefinitionInput], arguments: dict[str, Any]
) -> dict[str, str]:
    """
    Merge the call arguments with the declared defaults.

    A supplied argument always wins, except for a fixed input — see below. A missing one
    falls back to the input's default, which is how an optional input keeps working when
    the caller omits it.
    """
    values: dict[str, str] = {}

    for declared in inputs:
        # A fixed input ignores whatever arrived. It is absent from the tool's schema, so
        # a value for it was never asked for; honouring one would make "not negotiable"
        # depend on the caller being polite.
        if declared.source == "fixed":
            if declared.default_value:
                values[declared.key] = declared.default_value
            continue

        if declared.key in arguments and arguments[declared.key] is not None:
            values[declared.key] = _stringify(arguments[declared.key])
        elif declared.default_value:
            values[declared.key] = declared.default_value

    return values


def secret_keys(inputs: list[DefinitionInput]) -> set[str]:
    """Keys whose value must be masked in anything that leaves this process."""
    return {item.key for item in inputs if item.type in SECRET_TYPES}


def render(
    template: str | None,
    values: dict[str, str],
    *,
    mask: set[str] | None = None,
) -> str:
    """
    Substitute every ``{{key}}`` in ``template``.

    Keys listed in ``mask`` render as a fixed marker instead of their value, so a plan
    can be shown or logged without leaking a secret. An unknown key is left untouched;
    :func:`find_unresolved` is what turns that into an error, keeping rendering total.
    """
    if not template:
        return ""

    masked = mask or set()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in masked:
            return "••••••••"
        return values.get(key, match.group(0))

    return PLACEHOLDER.sub(replace, template)


def find_unresolved(templates: list[str | None], values: dict[str, str]) -> list[str]:
    """Every placeholder key used by the templates that ``values`` cannot fill."""
    missing: set[str] = set()

    for template in templates:
        if not template:
            continue
        for match in PLACEHOLDER.finditer(template):
            if match.group(1) not in values:
                missing.add(match.group(1))

    return sorted(missing)


def render_url(url: str | None, values: dict[str, str], *, mask: set[str] | None = None) -> str:
    """
    Renders a URL, dropping the query parameters nobody filled in.

    A query string is a list of filters, and a filter with no value is not an empty filter
    — it is one that was not asked for. `?search={{search}}&email={{email}}` with neither
    supplied has to become no query string at all, not `?search=&email=`, which many APIs
    read as "match the empty string".

    Only whole parameters are dropped, and only when the value is exactly one placeholder
    with nothing behind it. `?page={{page}}&limit=50` keeps its limit; `?q=user-{{name}}`
    is left to the ordinary rules, because half a value is not a missing one.

    The path is untouched. `/users/{{id}}` with no id is a real problem and stays one —
    dropping it would silently turn "delete user 13" into "delete users".
    """
    if not url:
        return ""

    head, separator, query = url.partition("?")
    if not separator:
        return render(url, values, mask=mask)

    kept = [
        parameter
        for parameter in query.split("&")
        if parameter and not _is_unfilled(parameter, values)
    ]

    rendered = render(head, values, mask=mask)
    if not kept:
        return rendered

    return rendered + "?" + "&".join(render(parameter, values, mask=mask) for parameter in kept)


def _is_unfilled(parameter: str, values: dict[str, str]) -> bool:
    """Whether this `key={{placeholder}}` pair has nothing to put in it."""
    _, separator, value = parameter.partition("=")
    if not separator:
        return False

    match = PLACEHOLDER.fullmatch(value.strip())

    return match is not None and not values.get(match.group(1))


def render_body(body: str | None, values: dict[str, str], *, mask: set[str] | None = None) -> str:
    """
    Renders a request body, dropping the JSON fields nobody filled in.

    The same reasoning as :func:`render_url`, for the same reason: a PATCH or a PUT that
    changes one field should send that field, not the other three set to a placeholder
    nobody replaced. `{"email": "{{email}}"}` with no email is not an empty e-mail — it is
    a field this request is not about.

    Only whole fields whose value is exactly one unfilled placeholder go. A body that is
    not JSON is left entirely alone: this cannot tell a field from a line of text, and
    guessing at one would be editing somebody's payload on a hunch.
    """
    if not body or not body.strip():
        return ""

    try:
        document = json.loads(body)
    except ValueError:
        return render(body, values, mask=mask)

    if not isinstance(document, dict):
        return render(body, values, mask=mask)

    kept = {
        key: value
        for key, value in document.items()
        if not _is_unfilled_value(value, values)
    }

    return render(json.dumps(kept, ensure_ascii=False, indent=2), values, mask=mask)


def _is_unfilled_value(value: object, values: dict[str, str]) -> bool:
    """Whether this field holds one placeholder and nothing was supplied for it."""
    if not isinstance(value, str):
        return False

    match = PLACEHOLDER.fullmatch(value.strip())

    return match is not None and not values.get(match.group(1))


def unresolved_in_url(url: str | None, values: dict[str, str]) -> list[str]:
    """The placeholders a URL still needs after the unfilled parameters have been dropped."""
    return find_unresolved([render_url(url, values)], values)


def keys_used(templates: list[str | None]) -> set[str]:
    """Every placeholder these templates refer to, filled or not."""
    used: set[str] = set()

    for template in templates:
        if template:
            used.update(match.group(1) for match in PLACEHOLDER.finditer(template))

    return used


def substituted_values(
    template: str | None, values: dict[str, str]
) -> dict[str, str]:
    """
    The values a template actually pulls in, keyed by placeholder.

    Only what the template references: a guardrail that inspects arguments should judge
    what reached the command, not every input the definition happens to declare.
    """
    if not template:
        return {}

    return {
        match.group(1): values[match.group(1)]
        for match in PLACEHOLDER.finditer(template)
        if match.group(1) in values
    }


def _stringify(value: Any) -> str:
    """
    Render an argument the way a shell or a SQL statement would read it.

    Booleans are lower cased because ``True`` is Python syntax, not something a remote
    command would understand.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
