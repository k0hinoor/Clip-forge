"""CLIPFORGE AI command line.

    python -m clipforge                 # start the app (API + worker + UI) and open the browser
    python -m clipforge serve --no-browser
    python -m clipforge worker          # run only the job worker (separate process/GPU box)
    python -m clipforge analyze <url>   # headless: analyse a video and print the discovered clips
    python -m clipforge doctor          # print the hardware / dependency report

The same entry point is installed as the ``clipforge`` console script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .config import Env, ensure_dirs, get_settings
from .logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def _print_banner(url: str) -> None:
    settings = get_settings()
    lines = [
        "",
        "  ██████╗██╗     ██╗██████╗ ███████╗ ██████╗ ██████╗  ██████╗ ███████╗",
        " ██╔════╝██║     ██║██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝",
        " ██║     ██║     ██║██████╔╝█████╗  ██║   ██║██████╔╝██║  ███╗█████╗",
        " ██║     ██║     ██║██╔═══╝ ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝",
        " ╚██████╗███████╗██║██║     ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗",
        "  ╚═════╝╚══════╝╚═╝╚═╝     ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
        "",
        f"  Local AI clipping studio · {url}",
        f"  Data:    {Env.DATA_DIR}",
        f"  Exports: {settings.resolved_export_dir()}",
        f"  Logs:    {Env.LOG_DIR}",
        "",
        "  Press Ctrl+C to stop.",
        "",
    ]
    print("\n".join(lines))


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.app import app

    host = args.host or Env.HOST
    port = args.port or Env.PORT
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}"

    if args.open_browser and Env.OPEN_BROWSER:
        def _open() -> None:
            time.sleep(2.5)
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - headless machines
                pass

        threading.Thread(target=_open, daemon=True).start()

    _print_banner(url)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=args.log_level.lower(),
        access_log=False,
        ws="none",
        timeout_keep_alive=120,
    )
    return 0


def command_worker(args: argparse.Namespace) -> int:
    from .jobs.manager import manager

    from .db import init_db

    init_db()
    state = manager().start(workers=args.workers or None)
    log.info("worker process running: %s", state)
    print(json.dumps(state, indent=2))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        manager().stop()
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    from .db import init_db
    from .jobs import queue as job_queue
    from .pipeline.analyze import analyze
    from .pipeline.context import NullReporter
    from .services.projects import create_project, project_clips, project_status

    init_db()
    reporter = NullReporter()
    project = create_project(url=args.url)
    project_id = project["id"]
    print(f"project {project_id} created for {args.url}")

    class Printer(NullReporter):
        def stage(self, key, message, *, fraction=0.0):  # noqa: D102
            super().stage(key, message, fraction=fraction)
            print(f"[{key:>11}] {message}")

        def sub(self, fraction, message=""):  # noqa: D102
            if message and (fraction * 100) % 10 < 2:
                print(f"             {int(fraction * 100):3d}%  {message}")

    outcome = analyze(project_id, report=Printer())
    clips = project_clips(project_id, sort="score")
    print("\n" + "=" * 72)
    print(json.dumps(outcome.to_dict(), indent=1, default=str))
    print("=" * 72)
    for clip in clips["clips"]:
        print(f"{clip['score']:5.1f}  {clip['duration']:5.1f}s  {clip['category']:<14} {clip['title'][:60]}")
    print(f"\n{len(clips['clips'])} clips stored for project {project_id}")
    return 0


def command_doctor(_args: argparse.Namespace) -> int:
    from .db import init_db
    from .system import diagnostics

    init_db()
    report = diagnostics()
    print(json.dumps(report, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipforge", description="CLIPFORGE AI - local AI clipping studio")
    parser.add_argument("--log-level", default=Env.LOG_LEVEL, help="DEBUG, INFO, WARNING, ERROR")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the API, worker and web UI")
    serve.add_argument("--host", default="")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--no-browser", dest="open_browser", action="store_false", default=True)
    serve.set_defaults(func=command_serve)

    worker = sub.add_parser("worker", help="run only the job worker")
    worker.add_argument("--workers", type=int, default=0)
    worker.set_defaults(func=command_worker)

    analyze_cmd = sub.add_parser("analyze", help="headless analysis of a YouTube URL")
    analyze_cmd.add_argument("url")
    analyze_cmd.set_defaults(func=command_analyze)

    doctor = sub.add_parser("doctor", help="print the hardware/dependency report")
    doctor.set_defaults(func=command_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level)
    ensure_dirs()

    if not getattr(args, "command", None):
        return command_serve(argparse.Namespace(host="", port=0, open_browser=True, log_level=args.log_level))
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
