"""Worker entry point: ``python -m clipforge.worker [--once]``."""

from __future__ import annotations

import argparse

from clipforge.core.config import get_settings
from clipforge.core.logging import configure_logging
from clipforge.db.session import make_session_factory
from clipforge.queue import get_queue
from clipforge.services.bootstrap import bootstrap
from clipforge.storage import get_storage
from clipforge.worker.runner import Worker


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ClipForge processing worker")
    parser.add_argument("--once", action="store_true", help="process runnable work and exit")
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--cleanup", action="store_true", help="run one cleanup pass and exit")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("worker", settings.LOG_LEVEL, settings.LOG_JSON)
    factory = make_session_factory(settings)
    bootstrap(settings, factory)
    storage = get_storage(settings)
    worker = Worker(settings, factory, storage, get_queue(settings, factory), worker_id=args.worker_id)
    if args.cleanup:
        worker.cleanup.run_once()
        return
    if args.once:
        worker.run_until_idle()
        return
    worker.run_forever()


if __name__ == "__main__":
    main()
