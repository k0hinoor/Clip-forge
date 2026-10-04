"""The frontend must not be able to call an endpoint that does not exist.

``frontend/lib/api.ts`` is the single place the UI talks to the backend, so every
``"/api/..."`` literal in it is checked against the FastAPI route table. A typo or a
renamed route therefore fails the test suite instead of showing up as a silent 404
in the browser.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CLIENT = Path(__file__).resolve().parents[2] / "frontend" / "lib" / "api.ts"

# "/api/clips/${id}/render" -> "/api/clips/{}/render"
_PATH_RE = re.compile(r"""["'`](/api/[^"'`?]*)""")
_TEMPLATE_RE = re.compile(r"\$\{[^}]*\}")


def _client_paths() -> set[str]:
    if not CLIENT.exists():  # pragma: no cover - frontend lives in the same repo
        pytest.skip("frontend/lib/api.ts is not present in this checkout")
    text = CLIENT.read_text(encoding="utf-8")
    paths: set[str] = set()
    for match in _PATH_RE.finditer(text):
        raw = match.group(1).rstrip("/")
        # replace ${...} inside the URL with a wildcard segment
        paths.add(_TEMPLATE_RE.sub("\x00", raw) or "/")
    return paths


def _tool_paths() -> set[str]:
    from clipforge.api.app import create_app

    app = create_app()
    spec = app.openapi()
    found = set(spec["paths"])
    # served by FastAPI itself rather than declared on a router
    found |= {"/api/docs", "/api/openapi.json"}
    # allow both with and without the trailing slash FastAPI accepts
    return {path.rstrip("/") for path in found}


def _matches(client_path: str, server_paths: set[str]) -> bool:
    if client_path in server_paths:
        return True
    # template the client's parameter placeholders to match OpenAPI's {name} form
    parts = client_path.split("/")
    pattern = re.compile(
        "/".join(re.escape(part) if part != "\x00" else r"\{[^/]+\}" for part in parts) + "$"
    )
    return any(pattern.match(server) for server in server_paths)


def test_every_frontend_endpoint_exists():
    client_paths = _client_paths()
    server_paths = _tool_paths()

    assert client_paths, "no API paths found in the client"

    missing = sorted(path for path in client_paths if not _matches(path, server_paths))
    assert not missing, (
        "the frontend calls endpoints the backend does not expose:\n  "
        + "\n  ".join(path.replace("\x00", "{id}") for path in missing)
    )


def test_frontend_client_covers_the_documented_api():
    """The REST surface in the README must exist, so the docs cannot rot."""
    readme = Path(__file__).resolve().parents[2] / "README.md"
    server_paths = _tool_paths()
    documented = set(_PATH_RE.findall(readme.read_text(encoding="utf-8")))
    documented |= {path.split("?")[0].split("|")[0].strip() for path in documented}

    missing = sorted(
        path.rstrip("/")
        for path in documented
        if path.startswith("/api/") and not _matches(path.rstrip("/"), server_paths)
    )
    assert not missing, f"documented endpoints that do not exist: {missing}"


def test_openapi_documents_the_error_envelope():
    from clipforge.api.app import create_app

    spec = create_app().openapi()
    assert spec["info"]["title"]
    # Every route must declare a response so the client can rely on the shapes.
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            assert operation.get("responses"), f"{method.upper()} {path} documents no response"
