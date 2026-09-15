"""
The tool's JSON Schema, which is the whole of what a client knows about its inputs.

The gateway builds the same schema for the panel from its own enum. The two are written
separately and must agree: a client that is told an input takes one shape while the panel
draws another is a disagreement nobody sees until something is typed into the wrong box.
"""

from app.models import Definition, DefinitionInput


def definition(*inputs: DefinitionInput) -> Definition:
    return Definition(id=1, name="script", toolName="script", inputs=list(inputs))


def test_block_asks_for_a_box() -> None:
    """
    A file's contents cannot be typed into a one-line field.

    Both of these used to reach a client as a plain string, and the panel's run screen —
    which generates its form from this schema and nothing else — drew each of them as a
    single line. That made a definition built around a heredoc impossible to call from
    the panel that created it.
    """
    schema = definition(DefinitionInput(key="content", type="block")).input_schema()
    content = schema["properties"]["content"]

    assert content["type"] == "string"
    assert content["format"] == "textarea"


def test_textarea_asks_for_a_box_too() -> None:
    schema = definition(DefinitionInput(key="notes", type="textarea")).input_schema()

    assert schema["properties"]["notes"] == {"type": "string", "format": "textarea"}


def test_the_hint_says_nothing_about_the_scan() -> None:
    """
    ``block`` and ``textarea`` ask to be drawn the same way and are governed differently.

    The scan exemption is read from the input's own type, never from here: this schema
    travels through a catalogue, and a format any caller can write is no place to keep a
    safety decision. So the hint matching is the point, not an oversight — the ceiling is
    the only thing the schema says about the difference, and it says it as a number rather
    than as permission.
    """
    block = definition(DefinitionInput(key="v", type="block")).input_schema()
    textarea = definition(DefinitionInput(key="v", type="textarea")).input_schema()

    assert block["properties"]["v"]["format"] == textarea["properties"]["v"]["format"]
    assert "maxLength" in block["properties"]["v"]
    assert "maxLength" not in textarea["properties"]["v"]


def test_plain_text_stays_a_line() -> None:
    schema = definition(DefinitionInput(key="path", type="text")).input_schema()

    assert "format" not in schema["properties"]["path"]


def test_the_rest_of_the_property_survives() -> None:
    """The new entries sit in the same lookup as `date` and `password`."""
    schema = definition(
        DefinitionInput(
            key="content", type="block", label="File contents", required=True,
            defaultValue="#!/bin/sh",
        )
    ).input_schema()

    assert schema["properties"]["content"]["description"] == "File contents"
    assert schema["properties"]["content"]["default"] == "#!/bin/sh"
    assert schema["required"] == ["content"]


def test_a_block_publishes_its_ceiling() -> None:
    """
    A client should be able to refuse before sending rather than after.

    The guardrail enforces this either way — a schema is a description, not a gate — but a
    caller that knows the ceiling can say so while the text is still in front of whoever
    wrote it.
    """
    from app.models import BODY_LIMIT

    schema = definition(DefinitionInput(key="content", type="block")).input_schema()

    assert schema["properties"]["content"]["maxLength"] == BODY_LIMIT


def test_a_textarea_publishes_no_ceiling() -> None:
    """
    Here the two part company.

    A textarea is an ordinary value that happens to be drawn in a bigger box; it keeps the
    shell-metacharacter scan, and nothing about it is unbounded in the way a file's
    contents are. Giving it the same ceiling would be inventing a rule nobody asked for.
    """
    schema = definition(DefinitionInput(key="notes", type="textarea")).input_schema()

    assert "maxLength" not in schema["properties"]["notes"]
