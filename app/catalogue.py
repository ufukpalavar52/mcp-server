"""
In-memory catalogue.

This service connects to nothing: no database, no gateway, no queue. Everything it
knows arrives in a request. The gateway publishes the definitions it wants exposed, this
module holds them, and both the MCP surface and the execution endpoint read from here.

The consequence is worth stating plainly: an empty catalogue is the normal state at
start up, and stays that way until the gateway publishes. That is a deliberate trade —
the alternative is this service holding gateway credentials and reaching back into it.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from app.models import Definition

logger = logging.getLogger(__name__)


class Catalogue:
    """
    The published definitions, replaced wholesale on each publish.

    Wholesale rather than incremental on purpose: the gateway knows the complete set, and
    a full replacement means a deleted definition disappears here too. An incremental
    protocol would need deletions carried separately, and a missed one would leave a tool
    callable after it was removed.
    """

    def __init__(self) -> None:
        self._definitions: dict[str, Definition] = {}
        self._published_at: datetime | None = None
        self._lock = threading.Lock()

    def replace(self, definitions: list[Definition]) -> int:
        """Replaces the catalogue. Returns how many definitions are now published."""
        with self._lock:
            self._definitions = {item.tool_name: item for item in definitions if item.enabled}
            self._published_at = datetime.now(UTC)

        logger.info(
            "Catalogue published: %d enabled tool(s) of %d received",
            len(self._definitions), len(definitions),
        )
        return len(self._definitions)

    def all(self) -> list[Definition]:
        with self._lock:
            return sorted(self._definitions.values(), key=lambda item: item.tool_name)

    def find(self, tool_name: str) -> Definition | None:
        with self._lock:
            return self._definitions.get(tool_name)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._definitions)

    @property
    def published_at(self) -> datetime | None:
        with self._lock:
            return self._published_at
