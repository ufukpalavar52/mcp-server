"""
Payload models mirroring the gateway's REST contract.

Hand written rather than generated: the surface is small, and an explicit model makes
a contract change fail here with a clear error instead of deep inside the planner.
"""

from __future__ import annotations

import base64
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ActionKind = Literal["rest", "ssh", "db"]
#: What a value looks like, and — for ``block`` alone — how it may be substituted.
#:
#: ``block`` is a body of text destined for a quoted heredoc: a file's contents, a config,
#: a script. It is the one type exempt from the shell-metacharacter scan every other
#: substituted value gets, because a python file with no newline in it is not a python
#: file. What replaces that scan is narrower and stronger — see
#: :func:`app.guardrails.check_static_command`: the command must put it in a *quoted*
#: heredoc, where the shell expands nothing, and the value may not contain the heredoc's
#: terminator. Nothing in it can end the command; there is no command syntax in scope.
#:
#: ``textarea`` is not this. It means a bigger box in the panel and nothing more, and it
#: keeps the ordinary scan: quietly exempting every definition that already used one would
#: be a change to what those definitions permit, made without anybody asking for it.
InputType = Literal[
    "text", "password", "number", "textarea", "block", "select", "boolean", "date"
]

#: Where an input is allowed to get its value.
#:
#: The type says what a value looks like; this says who is allowed to decide it. They are
#: different questions and conflating them is how a model ends up inventing a hostname.
InputSource = Literal["prompt", "caller", "fixed"]


class DefinitionInput(BaseModel):
    """A dynamic input; its ``key`` is what ``{{key}}`` refers to in a template."""

    key: str
    label: str = ""
    type: InputType = "text"
    required: bool = False
    default_value: str = Field(default="", alias="defaultValue")
    placeholder: str = ""
    options: list[str] = Field(default_factory=list)

    #: Who may decide this value.
    #:
    #: * ``prompt`` — a model may read it from what the user asked, and a caller may also
    #:   supply it directly. The default, because it is what every input did before this
    #:   field existed.
    #: * ``caller`` — only an explicit argument fills it. A model is not shown it and
    #:   cannot invent one, which is what you want for a host name or an account id.
    #: * ``fixed`` — always the declared default. It is left out of the tool's schema
    #:   entirely, so no client can offer it and none can override it. This is the only
    #:   way to say "this part of the command is not negotiable".
    source: InputSource = "prompt"

    model_config = ConfigDict(populate_by_name=True)


class ActionConfig(BaseModel):
    """
    Type specific settings of an action.

    One model for all three kinds, matching how the gateway stores them: a single JSON
    document whose meaningful subset is decided by the action's ``kind``.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # REST
    method: str | None = None
    url: str | None = None
    headers: list[dict[str, str]] = Field(default_factory=list)
    body: str | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")

    # SSH
    target_mode: str | None = Field(default=None, alias="targetMode")
    host: str | None = None
    hosts: list[str] = Field(default_factory=list)
    strategy: str | None = None
    concurrency: int | None = None
    batch_size: int | None = Field(default=None, alias="batchSize")
    stop_on_error: bool | None = Field(default=None, alias="stopOnError")
    port: int | None = None
    user: str | None = None
    working_dir: str | None = Field(default=None, alias="workingDir")
    auth: str | None = None
    command_mode: str | None = Field(default=None, alias="commandMode")
    command: str | None = None
    allowed_commands: list[str] = Field(default_factory=list, alias="allowedCommands")
    blocked_patterns: list[str] = Field(default_factory=list, alias="blockedPatterns")
    command_guidance: str | None = Field(default=None, alias="commandGuidance")
    require_approval: bool | None = Field(default=None, alias="requireApproval")
    sudo: bool | None = None

    #: Stream the command's output while it runs, rather than only at the end.
    #:
    #: Off unless the action asks. A command that finishes in a second gains nothing from
    #: being watched, and every chunk is a message on the broker. What this is for is the
    #: commands that do not finish on their own — ``tail -f``, a long install — where the
    #: whole value is in seeing the lines as they arrive.
    follow: bool | None = None

    #: How long a followed command may stay quiet before it is stopped, in seconds.
    #:
    #: Idle rather than total, because a log is watched until it goes quiet, not for a
    #: fixed span: a fixed minute cuts off a busy log mid-sentence and spends the whole
    #: minute on a silent one. Every line printed starts it again.
    follow_idle_seconds: int | None = Field(default=None, alias="followIdleSeconds")

    # Database
    engine: str | None = None
    database: str | None = None
    query_mode: str | None = Field(default=None, alias="queryMode")
    query: str | None = None
    schema_hint: str | None = Field(default=None, alias="schemaHint")

    #: Tables the model is told about, and the only ones introspection reads.
    schema_tables: list[str] = Field(default_factory=list, alias="schemaTables")

    #: The schema as the database itself reports it, read by the executor.
    generated_schema: str | None = Field(default=None, alias="generatedSchema")
    generated_schema_at: str | None = Field(default=None, alias="generatedSchemaAt")
    guidance: str | None = None
    allowed_operations: list[str] = Field(default_factory=list, alias="allowedOperations")
    max_rows: int | None = Field(default=None, alias="maxRows")
    read_only: bool | None = Field(default=None, alias="readOnly")


class SealedSecret(BaseModel):
    """
    A credential this service holds but cannot read.

    Passed through from the gateway to the executor untouched. Nothing here decrypts it —
    that happens in the executor, at the moment of use, through mcp-cipher — so a plan and
    the process that produced it are both worthless to anyone who takes them.
    """

    model_config = ConfigDict(populate_by_name=True)

    #: The sealed bytes.
    #:
    #: Base64 on the wire, because that is how Jackson serialises a ``byte[]`` and the
    #: gateway is where this comes from. Pydantic does *not* decode base64 into ``bytes``
    #: — it encodes the string as UTF-8 — so without the validator below the value became
    #: the ASCII of its own base64, and re-encoding it for the executor produced something
    #: the cipher could only reject.
    ciphertext: bytes
    key_id: str = Field(default="", alias="keyId")
    context: str = ""

    @field_validator("ciphertext", mode="before")
    @classmethod
    def _decode_base64(cls, value: Any) -> Any:
        if isinstance(value, str):
            return base64.b64decode(value)
        return value


class Action(BaseModel):
    """One step of a definition."""

    model_config = ConfigDict(populate_by_name=True)

    id: int
    kind: ActionKind
    name: str
    description: str = ""
    position: int = 0
    config: ActionConfig = Field(default_factory=ActionConfig)
    host_group_id: int | None = Field(default=None, alias="hostGroupId")
    host_group_name: str | None = Field(default=None, alias="hostGroupName")
    resolved_target_count: int = Field(default=0, alias="resolvedTargetCount")

    #: Every host this action would reach, with the group already expanded. The count above
    #: is enough to describe the work; these names are what it takes to do it.
    hosts: list[str] = Field(default_factory=list)

    #: Expected SSH host key per host. An executor refuses a host it has no key for.
    host_keys: dict[str, str] = Field(default_factory=dict, alias="hostKeys")

    #: Sealed credentials by role: privateKey, passphrase, password.
    credentials: dict[str, SealedSecret] = Field(default_factory=dict)

    def templates(self) -> list[str | None]:
        """Every field of this action that can carry a `{{placeholder}}`."""
        config = self.config

        return [
            config.url, config.body, config.command, config.query, config.host,
            *(header.get("value") for header in config.headers),
            *config.hosts,
        ]

    def inputs_used(self) -> set[str]:
        """
        The inputs this action refers to.
        
        Which is how "required" becomes a per-action question without moving the field:
        the templates already say what each action needs, and an input no selected action
        mentions is not required for this request whatever the definition says.
        """
        from app.templating import keys_used

        return keys_used(self.templates())

    def summary(self) -> str:
        """One line describing what this action does, for a model choosing between them."""
        if self.kind == "rest":
            return f"{self.config.method or 'GET'} {self.config.url or ''}".strip()
        if self.kind == "db":
            return (self.config.query or "a query written at run time").strip()
        if self.kind == "ssh":
            return (self.config.command or "a command written at run time").strip()
        return self.kind

    @property
    def is_dynamic(self) -> bool:
        """Whether the model has to write the command or query for this action."""
        if self.kind == "ssh":
            return self.config.command_mode == "dynamic"
        if self.kind == "db":
            return self.config.query_mode == "dynamic"
        return False


class ModelParams(BaseModel):
    """
    Request settings the operator chose for a model.

    Every field is optional, and absent means "whatever the provider does by default"
    rather than a value invented here — a model that rejects a sampling parameter must not
    be sent one just because this class has a field for it.
    """

    model_config = ConfigDict(populate_by_name=True)

    max_tokens: int | None = Field(default=None, alias="maxTokens")
    temperature: float | None = None
    timeout_ms: int | None = Field(default=None, alias="timeoutMs")

    #: Anthropic's own controls. Read so they round trip; nothing here sends them yet.
    effort: str | None = None
    thinking: str | None = None


class Definition(BaseModel):
    """A definition is exactly one MCP tool."""

    model_config = ConfigDict(populate_by_name=True)

    id: int
    name: str
    tool_name: str = Field(alias="toolName")
    tool_description: str = Field(default="", alias="toolDescription")
    model_id: int | None = Field(default=None, alias="modelId")
    model_name: str | None = Field(default=None, alias="modelName")
    model_identifier: str | None = Field(default=None, alias="modelIdentifier")

    #: Which service answers for this model. Defaults to Anthropic for a payload written
    #: before the gateway started sending it, so an older catalogue keeps working.
    model_provider: str | None = Field(default=None, alias="modelProvider")

    #: Where the model lives, when it is not at its provider's default address. This is
    #: what makes a self hosted open model reachable without changing any code.
    model_endpoint: str = Field(default="", alias="modelEndpoint")

    #: The model's API key, still sealed. Opened through mcp-cipher at the moment of use,
    #: so a key entered once in the panel is the key this service calls with.
    model_api_key: SealedSecret | None = Field(default=None, alias="modelApiKey")

    #: What the operator set for this model in the panel. Carried for the same reason as
    #: the key: this service makes the request, so a temperature set anywhere else is a
    #: setting that does nothing.
    model_params: ModelParams = Field(default_factory=lambda: ModelParams(),
                                      alias="modelParams")
    system_prompt: str = Field(default="", alias="systemPrompt")
    inputs: list[DefinitionInput] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    enabled: bool = True

    def reaches(self) -> list[str]:
        """
        What this tool actually touches, for a model deciding whether to call it.

        The catalogue held this all along and the router was never shown it. A tool called
        ``bireysel_local_yaanidb``, described as "Bireysel local yaanidb", with no inputs,
        gave a router nothing to match a question about accounts against — and a model that
        answers "no tool matches" to a question one of its tools answers perfectly is doing
        the only reasonable thing with what it was given.

        Names, not columns. The tables are the part that says what a tool is *about*; the
        thirty columns under each are what the planner needs later, and putting them here
        would push every other tool's description out of the router's attention.
        """
        found: list[str] = []

        for action in self.actions:
            for line in (action.config.generated_schema or "").splitlines():
                # One shape across every engine: mcp-action writes "table <name> (".
                if line.startswith("table ") and line.rstrip().endswith("("):
                    name = line[len("table "):].rstrip(" (").strip()
                    if name and name not in found:
                        found.append(name)

            # An SSH action reaches commands rather than tables, and which ones it may run
            # is the same kind of fact: it says what the tool is for.
            for command in action.config.allowed_commands:
                if command and command not in found:
                    found.append(command)

        return found

    def prompt_schema(self) -> dict[str, Any]:
        """
        The subset of the schema a model may fill from what the user asked.

        Narrower than :meth:`input_schema` on purpose. An input marked ``caller`` is left
        out, so the model is not shown it and cannot invent one — which is the point of the
        distinction: reading a date range out of a sentence is reasonable, guessing a host
        name is not.
        """
        full = self.input_schema()
        fillable = {item.key for item in self.inputs if item.source == "prompt"}

        properties = {
            key: value
            for key, value in full["properties"].items()
            if key in fillable
        }

        return {
            "type": "object",
            "properties": properties,
            "required": [key for key in full["required"] if key in fillable],
            "additionalProperties": False,
        }

    def input_schema(self) -> dict[str, Any]:
        """
        The tool's JSON Schema, derived from its inputs.

        Derived rather than carried: the gateway builds the same schema for its own
        panel, and shipping it separately would create two copies that can disagree.
        A password input contributes its type but never its default, so a secret cannot
        reach a client through the catalogue.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for item in self.inputs:
            if not item.key:
                continue

            # A fixed input is not part of the contract. Publishing it would invite a
            # client to send a value that is then ignored, which is worse than not
            # offering it: the client has no way to tell the difference.
            if item.source == "fixed":
                continue

            prop: dict[str, Any] = {"type": _JSON_TYPES.get(item.type, "string")}

            if fmt := _JSON_FORMATS.get(item.type):
                prop["format"] = fmt
            if item.type == "select" and item.options:
                prop["enum"] = list(item.options)
            if item.label:
                prop["description"] = item.label
            if item.default_value and item.type != "password":
                prop["default"] = _coerce(item.type, item.default_value)

            properties[item.key] = prop
            if item.required:
                required.append(item.key)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


_JSON_TYPES: dict[str, str] = {"number": "number", "boolean": "boolean"}
_JSON_FORMATS: dict[str, str] = {"date": "date", "password": "password"}


def _coerce(input_type: str, raw: str) -> Any:
    """Keeps a schema default in the JSON type of the property it belongs to."""
    if input_type == "number":
        try:
            return int(raw) if raw.isdigit() else float(raw)
        except ValueError:
            return raw
    if input_type == "boolean":
        return raw.lower() == "true"
    return raw
