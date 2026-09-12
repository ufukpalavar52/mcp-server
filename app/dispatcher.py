"""
Handing a plan to whatever executes it.

Two implementations. :class:`NullDispatcher` produces the plan and stops, which is what a
deployment without a broker gets; :class:`QueueDispatcher` publishes an executable job to
RabbitMQ for mcp-action.

The split is the reason the seam was defined before anything used it: adding the real one
meant implementing the protocol and binding it, with no change to the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from app.config import Settings
from app.logcontext import current_actor
from app.jobs import IncompleteJob, Job, build_job
from app.models import Definition
from app.planner import Plan

logger = logging.getLogger(__name__)


class DispatchResult(BaseModel):
    """What happened to a plan after it was decided."""

    status: Literal["skipped", "queued", "refused", "awaiting_approval"] = "skipped"
    reason: str = ""

    #: Set once a real executor accepts the work.
    run_id: str | None = None

    #: Per action, the identifier the executor will report progress against.
    action_run_ids: dict[int, str] = Field(default_factory=dict)


class Dispatcher(Protocol):
    """
    Anything that can take a plan and make it happen.

    The definition and the arguments come along with the plan because a plan alone is not
    executable: its commands are masked and its targets are a description. Rebuilding the
    real command is the dispatcher's job, and it needs the same inputs the planner had.
    """

    async def dispatch(
        self,
        definition: Definition,
        plan: Plan,
        arguments: dict[str, Any],
        *,
        actor: str | None,
    ) -> DispatchResult:
        ...


class NullDispatcher:
    """
    Accepts a plan and does nothing with it.

    Deliberately not a silent no-op: it says so in the result, so a caller can tell
    "planned but nothing ran" apart from "ran successfully" without reading the code.
    """

    async def dispatch(
        self,
        definition: Definition,
        plan: Plan,
        arguments: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> DispatchResult:
        if plan.status != "planned":
            return DispatchResult(
                status="refused",
                reason=f"Plan is {plan.status}; nothing is dispatched unless it is planned",
            )

        logger.info(
            "Plan for %s would dispatch %d action(s) on behalf of %s; no executor is wired",
            plan.tool, len(plan.actions), actor or "unknown",
        )
        return DispatchResult(
            status="skipped",
            reason="No executor is configured; the plan was produced but not run",
        )


class QueueDispatcher:
    """
    Publishes executable jobs to RabbitMQ for mcp-action.

    Only a ``planned`` plan is dispatched. An incomplete or rejected one is refused here as
    well as by the executor: the executor's check is the one that counts, because a broker
    sits between the two, but sending work that is known to be unacceptable would put a
    command on a queue for no reason at all.

    Publishing is not the same as running. ``queued`` means the broker accepted the message
    and nothing more — whether it succeeded arrives later on ``mcp.results``, which is the
    gateway's to consume.

    Bound only when a queue is configured; a deployment without one gets
    :class:`NullDispatcher` instead. There is deliberately no "no queue" branch here — it
    would be unreachable, and a message nobody can ever see is worse than none.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._connection: Any = None

    async def dispatch(
        self,
        definition: Definition,
        plan: Plan,
        arguments: dict[str, Any],
        *,
        actor: str | None = None,
    ) -> DispatchResult:
        if plan.status != "planned":
            return DispatchResult(
                status="refused",
                reason=f"Plan is {plan.status}; nothing is dispatched unless it is planned",
            )

        try:
            # The id comes from the request context rather than a parameter: every caller
            # of this already passes the label, and threading a second one through four
            # signatures to reach the same place is how a parameter ends up accepted and
            # never sent.
            job, action_run_ids = build_job(
                definition, arguments, actor or "", plan, actor_id=current_actor()
            )
        except IncompleteJob as exc:
            logger.error("Refusing to dispatch %s: %s", definition.tool_name, exc)
            return DispatchResult(status="refused", reason=str(exc))

        try:
            await self._publish(job)
        except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
            # A broker that will not take the message is a dependency being down. The plan
            # is still valid and worth returning; what changes is that nothing will run.
            logger.error("Could not publish job %s: %s", job.run_id, exc)
            return DispatchResult(
                status="refused",
                reason=f"The job queue is unavailable: {exc}",
            )

        logger.info(
            "Dispatched run %s for %s with %d action(s)",
            job.run_id, definition.tool_name, len(job.actions),
        )

        return DispatchResult(
            status="queued",
            reason="Published to the executor queue",
            run_id=job.run_id,
            action_run_ids=action_run_ids,
        )

    async def _publish(self, job: Job) -> None:
        import aio_pika

        connection = await aio_pika.connect_robust(self._settings.queue_url)

        async with connection:
            channel = await connection.channel()

            # Declared by the publisher too, and idempotently: a planner started against a
            # fresh broker works instead of failing on a queue somebody forgot to create.
            await channel.declare_queue(self._settings.queue_name, durable=True)

            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=job.model_dump_json().encode(),
                    content_type="application/json",
                    # Persistent: a broker restart must not discard accepted work.
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=self._settings.queue_name,
            )
