from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from loopgraph_supervisor.api.app import create_app
from loopgraph_supervisor.config import Settings, build_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopgraph-supervisor")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="Start the HTTP control plane")
    serve.add_argument("--config", help="Path to a JSON settings file")
    serve.add_argument("--host", help="Override the configured bind host")
    serve.add_argument("--port", type=int, help="Override the configured bind port")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        settings = Settings.from_json(args.config) if args.config else Settings()
        runtime = build_runtime(settings)
        uvicorn.run(
            create_app(runtime),
            host=args.host or settings.host,
            port=args.port or settings.port,
        )
