"""
Who asked, on every log line of the request that serves them.

The identity arrives on the request body and was used only to label the job. A log could
say what was planned and not for whom, so "everything this person did" had no answer that
did not involve reading the gateway's database.

The id rather than the email: it answers the same question, and it keeps addresses out of
six log files and a log store with no encryption and thirty days of retention. This stack
seals query output for that reason; writing personal data into the logs beside it would
undo the arrangement one line at a time.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

#: The acting user, for the duration of one request.
#:
#: A ContextVar rather than a thread local, because the server is async: one thread serves
#: many requests, interleaved, and a thread local would hand whichever of them spoke last
#: to whichever is logging now. Each task gets its own copy of this.
_actor: ContextVar[str] = ContextVar("actor", default="-")


def acting(actor_id: str | int | None) -> None:
    """Names the acting user for everything logged from here on in this request."""
    _actor.set(str(actor_id) if actor_id not in (None, "") else "-")


def current_actor() -> str:
    """The acting user's id, or an empty string when there is nobody to name."""
    actor = _actor.get()
    return "" if actor == "-" else actor


class ActorFilter(logging.Filter):
    """
    Puts the current actor on every record, so the format string can use it.

    A filter rather than an adapter: an adapter has to be reached for at each call site,
    and a line logged through a plain module logger — which is most of them, and all of the
    ones in libraries — would go out unlabelled.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.actor = _actor.get()
        return True
