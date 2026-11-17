"""
Request-body limits, enforced before a body is read in full.

BodySizeLimitMiddleware is a plain ASGI middleware. For each request it looks up the limit for the
path, refuses a declared Content-Length over it with 413 before reading anything, and counts the
bytes of bodies sent without a length (chunked) as they arrive, stopping with 413 as soon as the
count passes the limit. The upload handler and src/ingest then apply the exact file limits.

Limits:
UPLOAD_BODY_LIMIT  POST /contracts/upload: MAX_FILE_BYTES plus 1 MB for the multipart envelope
TEXT_BODY_LIMIT    POST /contracts/text: MAX_TEXT_CHARS of pasted text at up to 6 bytes per character
                   once JSON-escaped (\\uXXXX), plus 1 MB
JSON_BODY_LIMIT    every other API path (weight sets, previews): 1 MB
Paths under the Gradio mount are left to Gradio, which applies its own 25 MB upload limit.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from fastapi import HTTPException

from src.ingest.documents import MAX_FILE_BYTES, MAX_TEXT_CHARS, MB

# ASGI shapes (the same as starlette.types, spelled out to avoid a direct starlette dependency)
Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

UPLOAD_BODY_LIMIT = MAX_FILE_BYTES + MB
TEXT_BODY_LIMIT = 6 * MAX_TEXT_CHARS + MB
JSON_BODY_LIMIT = MB
TOO_LARGE = "The request body is too large."


def default_limit_for(ui_path: str | None) -> Callable[[str], int | None]:
    """The limit function the app uses: None (no limit here) under ui_path, else the limits above."""
    def limit_for(path: str) -> int | None:
        """Body limit in bytes for a request path."""
        if ui_path and (path == ui_path or path.startswith(ui_path.rstrip("/") + "/")):
            return None
        if path == "/contracts/upload":
            return UPLOAD_BODY_LIMIT
        if path == "/contracts/text":
            return TEXT_BODY_LIMIT
        return JSON_BODY_LIMIT
    return limit_for


class BodySizeLimitMiddleware:
    """Refuse HTTP request bodies over limit_for(path) bytes with 413, without buffering them."""

    def __init__(self, app: ASGIApp, limit_for: Callable[[str], int | None]) -> None:
        """Wrap app; limit_for(path) returns the byte limit for a path, or None for no limit."""
        self.app, self.limit_for = app, limit_for

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check the declared length, then count streamed bytes, for HTTP requests only."""
        limit = self.limit_for(scope["path"]) if scope["type"] == "http" else None
        if limit is None:
            await self.app(scope, receive, send)
            return

        # A declared length over the limit is refused before reading a byte of the body
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            status = 413 if declared.isdigit() else 400
            await _send_json(send, status, TOO_LARGE if status == 413 else "Invalid Content-Length header.")
            return

        # Otherwise count bytes as they arrive; raising here ends the handler's body read with 413
        received = 0

        async def limited_receive() -> Message:
            """Pass messages through, raising 413 once the body passes the limit."""
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=TOO_LARGE)
            return message

        await self.app(scope, limited_receive, send)


async def _send_json(send: Send, status: int, detail: str) -> None:
    """Send a complete JSON error response {"detail": detail}."""
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                            (b"connection", b"close")]})
    await send({"type": "http.response.body", "body": body})
