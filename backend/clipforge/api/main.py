"""ASGI entry point: ``uvicorn clipforge.api.main:app`` or ``python -m clipforge.api``."""

from __future__ import annotations

from clipforge.api.app import create_app


def __getattr__(name: str):  # lazily build the app so importing this module has no side effects
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)


def run() -> None:
    import uvicorn

    from clipforge.core.config import get_settings

    s = get_settings()
    uvicorn.run(create_app(s), host=s.API_HOST, port=s.API_PORT, proxy_headers=s.TRUST_PROXY_HEADERS,
                forwarded_allow_ips="*" if s.TRUST_PROXY_HEADERS else None, log_config=None)


if __name__ == "__main__":
    run()
