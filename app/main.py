"""
Application entry point.

One process, two surfaces over the same catalogue and the same planner: the MCP
streamable HTTP transport at ``/mcp`` for MCP clients, and plain REST for the gateway.

The service opens no outbound connection except the model call that dynamic actions
need. No database, no gateway, no queue.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI

from app.api import router
from app.catalogue import Catalogue
from app.config import Settings, get_settings
from app.cipher import CipherClient
from app.dispatcher import NullDispatcher, QueueDispatcher
from app.logcontext import ActorFilter
from app.router import PromptRouter
from app.mcp_server import build_mcp_server
from app.planner import Planner

logger = logging.getLogger(__name__)


#: What the console and the file both use. The level is the first thing anybody looks for
#: and the name says which part of the server spoke.
LOG_FORMAT = "%(asctime)s %(levelname)-5s [actor=%(actor)s] %(name)s: %(message)s"

#: Ten megabytes, five files behind it. Bounded because a log that fills a disk takes the
#: service down with it, and it does that at the least convenient moment.
LOG_BYTES = 10 * 1024 * 1024
LOG_BACKUPS = 5


def _start_logging(settings: Settings) -> None:
    """
    Console always, and a file when one is configured.

    Both, not either: the console is what an IDE shows and what somebody watching a
    terminal reads, and the file is what a log shipper can tail after they have gone home.
    Choosing between them would mean losing one of those.

    A directory that cannot be written is reported and then let go. A service that refuses
    to start because it could not open a log file has turned an inconvenience into an
    outage.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    directory = (settings.log_dir or "").strip()
    if directory:
        path = Path(directory).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(
                    path / "mcp-server.log",
                    maxBytes=LOG_BYTES,
                    backupCount=LOG_BACKUPS,
                    encoding="utf-8",
                )
            )
        except OSError as failure:
            logging.getLogger(__name__).warning(
                "No log file: %s could not be written (%s)", path, failure
            )

    # On the handlers, not the root logger: a filter on a logger runs only for records
    # made through that logger, and a record from a library's own logger would reach the
    # handler with no `actor` attribute at all — which the format string turns into an
    # exception, swallowed, and the line is lost.
    for handler in handlers:
        handler.addFilter(ActorFilter())

    logging.basicConfig(
        level=settings.log_level.upper(),
        format=LOG_FORMAT,
        handlers=handlers,
        # basicConfig does nothing at all when the root logger already has handlers, which
        # is the case under uvicorn --reload and in a test that built an app before.
        force=True,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Builds the application. Accepts settings so tests can supply their own."""
    settings = settings or get_settings()
    _start_logging(settings)

    catalogue = Catalogue()
    cipher = CipherClient(settings.cipher_address, settings.cipher_token)
    planner = Planner(settings, cipher)
    # A broker turns this from a service that describes work into one that causes it, so
    # the choice is made by configuration rather than by a flag: no queue, no dispatch.
    dispatcher = QueueDispatcher(settings) if settings.has_queue else NullDispatcher()
    mcp_server = build_mcp_server(catalogue, planner, dispatcher)

    # Building the ASGI app is what creates the session manager, so it has to happen
    # before the lifespan below can start it.
    mcp_app = mcp_server.streamable_http_app(streamable_http_path="/")

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting MCP server on %s:%d", settings.host, settings.port)
        logger.info("Catalogue is empty until the gateway publishes to PUT /api/v1/catalogue")

        if cipher.configured:
            logger.info(
                "Cipher at %s: %s", settings.cipher_address,
                "reachable" if cipher.reachable() else "not answering")
        else:
            logger.warning(
                "No cipher configured; a model's own API key cannot be opened")

        logger.info(
            "Model providers available: %s",
            ", ".join(settings.configured_providers),
        )
        if not settings.has_anthropic_key and not settings.has_openai_key:
            logger.warning(
                "No hosted model key configured; only definitions whose model names its "
                "own endpoint can be planned dynamically"
            )
        if settings.has_queue:
            logger.info("Dispatching to %s on %s:%d",
                        settings.queue_name, settings.queue_host, settings.queue_port)
        else:
            logger.warning("No queue configured; plans are produced but never executed")

        if not settings.requires_token:
            logger.warning(
                "No publisher token configured; the gateway routes are unauthenticated"
            )

        async with mcp_server.session_manager.run():
            yield

        logger.info("MCP server stopped")

    app = FastAPI(
        title="MCP Server",
        description=(
            "Publishes a pushed tool catalogue over MCP and decides what a tool call "
            "resolves to. It plans; it does not execute, and it connects to nothing."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.settings = settings
    app.state.catalogue = catalogue
    app.state.planner = planner
    app.state.dispatcher = dispatcher
    app.state.prompt_router = PromptRouter(settings, catalogue, cipher)
    app.state.cipher = cipher

    app.include_router(router)
    app.mount(settings.mcp_path, mcp_app)

    return app


app = create_app()


def main() -> None:
    """Development entry point: ``python -m app.main``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
