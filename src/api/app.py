"""
Builds the FastAPI application around a ClauseTriageService.

create_app(): attaches the service to app.state (handlers reach it through routers.get_service),
              includes the contract, weight and server routers plus any extra routers (python -m
              src.api adds routers/review.py), limits request bodies, maps service errors to status
              codes, and mounts the reviewer page (src/ui/reviewer.py), which calls the same service

Errors never carry a stack trace to the client: service errors keep their message (it is written
for reviewers), validation errors list the location and reason of each problem without echoing the
input, and anything unexpected is logged and answered with a plain 500.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.limits import BodySizeLimitMiddleware, default_limit_for
from src.api.routers import contracts, system, weights
from src.api.service import (ClauseTriageService, ConflictError, InvalidInputError, NotFoundError, ServiceError,
                             TooLargeError, UnsupportedTypeError)

logger = logging.getLogger(__name__)

UI_PATH = "/ui"
# Gradio's own upload cap for the mounted page, the same as the standalone demo page's
UI_MAX_FILE_SIZE = "25mb"
# Most specific first: the first matching class decides the status code
_STATUS_OF: tuple[tuple[type[ServiceError], int], ...] = (
    (NotFoundError, 404), (ConflictError, 409), (TooLargeError, 413), (UnsupportedTypeError, 415),
    (InvalidInputError, 400), (ServiceError, 400),
)


def create_app(service: ClauseTriageService, *, routers: Sequence[APIRouter] = (),
               ui_path: str | None = UI_PATH, host: str = "127.0.0.1", port: int = 8000) -> FastAPI:
    """
    The API around service, with extra routers included after the built-in ones. ui_path mounts
    the reviewer page (src.ui.reviewer.build_reviewer_app on the same service) there; None leaves it
    out. host and port only tell Gradio where it is served.
    """
    app = FastAPI(title="ClauseTriage API", version="0.1.0",
                  description="Local research prototype without authentication. Do not expose it on a network. "
                              "Scores use team-set weights that no legal expert has validated; not legal advice.")
    app.state.service = service
    for router in (system.router, contracts.router, weights.router, *routers):
        app.include_router(router)
    app.add_middleware(BodySizeLimitMiddleware, limit_for=default_limit_for(ui_path))
    _add_error_handlers(app)

    # The reviewer page, in the same process and over the same service as the endpoints
    if ui_path is not None:
        import gradio as gr

        from src.ui.reviewer import build_reviewer_app

        app = gr.mount_gradio_app(app, build_reviewer_app(service), path=ui_path, server_name=host,
                                  server_port=port, max_file_size=UI_MAX_FILE_SIZE)
    return app


def _add_error_handlers(app: FastAPI) -> None:
    """Map service errors to status codes, trim validation errors, and hide unexpected ones."""
    async def service_error(request: Request, error: ServiceError) -> JSONResponse:
        """A service error's own message, with the status code of its class."""
        status = next(code for kind, code in _STATUS_OF if isinstance(error, kind))
        return JSONResponse(status_code=status, content={"detail": str(error)})

    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        """422 listing where and why the request is invalid, without echoing the (possibly huge) input."""
        problems = [{"loc": list(item.get("loc", ())), "msg": item.get("msg", ""), "type": item.get("type", "")}
                    for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": problems})

    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        """Log the traceback server-side and answer with a plain 500."""
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    app.add_exception_handler(ServiceError, service_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(Exception, unexpected_error)
