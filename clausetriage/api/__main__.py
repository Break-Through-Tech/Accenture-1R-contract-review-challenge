"""
Run the ClauseTriage API and the Gradio page in one local process.

From the repo root:
    python -m clausetriage.api                                   # http://127.0.0.1:8000 (API docs at /docs, page at /ui/)
    python -m clausetriage.api --port 8080 --database outputs/api/try.sqlite
    python -m clausetriage.api --weights path/to/weights.json --bundle path/to/bundle.joblib

There is no authentication. The server binds to 127.0.0.1 and must not be exposed on a network;
--host with any other address prints a warning. See docs/api.md.
"""
from __future__ import annotations

import argparse
import ipaddress
import logging
import sys
from collections.abc import Sequence

from clausetriage.api.runtime import add_service_arguments

DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 8000
EXPOSED_WARNING = ("WARNING: the ClauseTriage API has no authentication. Binding to {host} may expose it to other "
                   "machines; use 127.0.0.1 unless you are sure the network is private.")


def is_loopback(host: str) -> bool:
    """Whether host only accepts connections from this machine (localhost or a loopback address)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options (see the module docstring)."""
    parser = argparse.ArgumentParser(prog="python -m clausetriage.api", description="Run the ClauseTriage API and page")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to bind (default: 127.0.0.1, this machine only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-ui", action="store_true", help="don't mount the Gradio page at /ui")
    add_service_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load the model and store, build the app and serve it with uvicorn."""
    import uvicorn

    from clausetriage.api.app import UI_PATH, create_app
    from clausetriage.api.routers import review
    from clausetriage.api.runtime import build_service

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not is_loopback(args.host):
        print(EXPOSED_WARNING.format(host=args.host), file=sys.stderr)
    service = build_service(args)
    ui_path = None if args.no_ui else UI_PATH
    # The review queue and feedback endpoints come in as an extra router
    app = create_app(service, routers=[review.router], ui_path=ui_path, host=args.host, port=args.port)
    page = f"  page: http://{args.host}:{args.port}{ui_path}/" if ui_path else ""
    print(f"ClauseTriage API: http://{args.host}:{args.port}/docs{page}  model {service.model_version}, "
          f"weights {service.default_weight_set.weight_version_id}, database {args.database}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
