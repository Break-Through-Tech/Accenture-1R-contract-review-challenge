"""Tests for src/api/limits.py: body limits by path, declared lengths, and streamed bodies without a length."""
import asyncio

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.limits import (JSON_BODY_LIMIT, TEXT_BODY_LIMIT, UPLOAD_BODY_LIMIT, BodySizeLimitMiddleware,
                            default_limit_for)
from src.ingest.documents import MAX_FILE_BYTES, MAX_TEXT_CHARS, MB


def echo_app(limit: int) -> FastAPI:
    """An app whose /echo endpoint reads the whole body and returns its length, behind a fixed limit."""
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request) -> dict:
        """Read the full body and report its size."""
        return {"size": len(await request.body())}

    app.add_middleware(BodySizeLimitMiddleware, limit_for=lambda path: limit)
    return app


def run_streamed(app, chunks: list[bytes]) -> tuple[int, int]:
    """Send a POST /echo whose body arrives in chunks with no Content-Length; return (status, chunks read)."""
    messages = [{"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
                for index, chunk in enumerate(chunks)]
    sent, read = [], 0

    async def receive():
        """Hand out the next body chunk."""
        nonlocal read
        read += 1
        return messages[read - 1]

    async def send(message):
        """Collect response messages."""
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/echo", "raw_path": b"/echo", "query_string": b"",
             "headers": [(b"content-type", b"application/octet-stream")], "http_version": "1.1",
             "scheme": "http", "server": ("testserver", 80), "client": ("testclient", 1), "root_path": ""}
    asyncio.run(app(scope, receive, send))
    return next(m["status"] for m in sent if m["type"] == "http.response.start"), read


# ----- Limits by path -----
@pytest.mark.parametrize(("path", "expected"), [
    ("/contracts/upload", MAX_FILE_BYTES + MB),
    ("/contracts/text", 6 * MAX_TEXT_CHARS + MB),
    ("/weights", MB),
    ("/weights/preview", MB),
    ("/ui", None),
    ("/ui/gradio_api/upload", None),
    ("/uix", MB),
])
def test_default_limit_for_paths(path, expected):
    """Hand-computed limits: uploads 26 MB, pasted text 13 MB, other API paths 1 MB, the Gradio mount none."""
    assert default_limit_for("/ui")(path) == expected


def test_limits_are_the_documented_sizes():
    """The constants are the sizes docs/api.md states."""
    assert (UPLOAD_BODY_LIMIT, TEXT_BODY_LIMIT, JSON_BODY_LIMIT) == (26 * MB, 12_000_000 + MB, MB)


def test_no_ui_path_limits_everything():
    """Without a Gradio mount, /ui paths get the JSON limit like any other path."""
    assert default_limit_for(None)("/ui/upload") == MB


# ----- Declared lengths -----
def test_body_within_limit_passes():
    """A body at the limit reaches the handler whole."""
    with TestClient(echo_app(10)) as client:
        assert client.post("/echo", content=b"x" * 10).json() == {"size": 10}


def test_declared_length_over_limit_is_413():
    """A Content-Length over the limit is refused with a JSON 413."""
    with TestClient(echo_app(10)) as client:
        response = client.post("/echo", content=b"x" * 11)
    assert (response.status_code, response.json()) == (413, {"detail": "The request body is too large."})


# ----- Streamed bodies -----
def test_streamed_body_over_limit_is_413_before_the_rest_is_read():
    """Without a Content-Length, reading stops with 413 at the first chunk past the limit."""
    status, read = run_streamed(echo_app(10), [b"x" * 6, b"x" * 6, b"x" * 6, b"x" * 6])
    assert (status, read) == (413, 2)


def test_streamed_body_within_limit_passes():
    """A chunked body under the limit is read in full."""
    status, read = run_streamed(echo_app(10), [b"x" * 5, b"x" * 5])
    assert (status, read) == (200, 2)
