"""LLM providers (TRD §15, §53).

The application works with every remote provider disabled. Only transcript
excerpts and structured metadata are ever sent — never media. Remote endpoints
are refused unless ``ALLOW_REMOTE_PROVIDERS=true``.
"""

from __future__ import annotations

import ipaddress
import json
import re
import threading
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from clipforge.core.config import Settings
from clipforge.core.errors import AppError, ErrorCode
from clipforge.core.logging import get_logger

log = get_logger(__name__)


class LLMProvider(Protocol):
    name: str
    model: str
    is_remote: bool

    def generate_json(self, system: str, prompt: str) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]: ...


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if host in ("localhost", "ollama", "host.docker.internal") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("LLM did not return JSON") from None
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("LLM JSON must be an object")
    return data


class HeuristicProvider:
    """No model at all — metadata is produced by deterministic heuristics."""

    name = "heuristic"
    model = "heuristic-v1"
    is_remote = False

    def generate_json(self, system: str, prompt: str) -> dict[str, Any]:
        raise AppError(ErrorCode.MODEL_UNAVAILABLE, internal="heuristic provider has no generative model")

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "ok": True}


class OllamaProvider:
    name = "ollama"
    is_remote = False

    def __init__(self, base_url: str, model: str, timeout: float, allow_remote: bool) -> None:
        if not is_local_url(base_url) and not allow_remote:
            raise AppError(ErrorCode.MODEL_UNAVAILABLE, retryable=False,
                           internal="OLLAMA_BASE_URL is not local and ALLOW_REMOTE_PROVIDERS=false")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.is_remote = not is_local_url(base_url)

    def generate_json(self, system: str, prompt: str) -> dict[str, Any]:
        try:
            resp = httpx.post(f"{self.base_url}/api/chat", timeout=self.timeout, json={
                "model": self.model, "stream": False, "format": "json",
                "options": {"temperature": 0.2, "seed": 7},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            })
            resp.raise_for_status()
            return _extract_json(resp.json().get("message", {}).get("content", ""))
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            raise AppError(ErrorCode.MODEL_UNAVAILABLE, internal=f"ollama: {exc}") from exc

    def health(self) -> dict[str, Any]:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {"provider": self.name, "ok": True, "model": self.model,
                    "model_installed": any(self.model == m or m.startswith(self.model + ":") for m in models)}
        except Exception as exc:
            return {"provider": self.name, "ok": False, "error": type(exc).__name__}


class TransformersProvider:
    """Local Hugging Face transformers text-generation (optional dependency)."""

    name = "transformers"
    is_remote = False

    def __init__(self, model: str) -> None:
        self.model = model
        self._pipe: Any = None
        self._lock = threading.Lock()

    def _pipeline(self) -> Any:
        with self._lock:
            if self._pipe is None:
                try:
                    from transformers import pipeline  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise AppError(ErrorCode.MODEL_UNAVAILABLE, retryable=False,
                                   internal="transformers not installed") from exc
                try:
                    self._pipe = pipeline("text-generation", model=self.model)
                except Exception as exc:
                    raise AppError(ErrorCode.MODEL_UNAVAILABLE, internal=str(exc)[:300]) from exc
            return self._pipe

    def generate_json(self, system: str, prompt: str) -> dict[str, Any]:
        pipe = self._pipeline()
        try:
            out = pipe(f"{system}\n\n{prompt}\n\nJSON:", max_new_tokens=300, do_sample=False,
                       return_full_text=False)
            return _extract_json(out[0]["generated_text"])
        except (ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise AppError(ErrorCode.MODEL_UNAVAILABLE, internal=f"transformers: {exc}") from exc

    def health(self) -> dict[str, Any]:
        try:
            import transformers  # type: ignore[import-not-found]  # noqa: F401
            return {"provider": self.name, "ok": True, "model": self.model}
        except ImportError:
            return {"provider": self.name, "ok": False, "error": "not installed"}


class OpenAICompatibleProvider:
    """Optional remote provider. Disabled unless explicitly permitted (TRD §53)."""

    name = "openai_compatible"
    is_remote = True

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float, allow_remote: bool) -> None:
        if not allow_remote and not is_local_url(base_url):
            raise AppError(ErrorCode.MODEL_UNAVAILABLE, retryable=False,
                           internal="remote LLM provider requires ALLOW_REMOTE_PROVIDERS=true")
        if not base_url:
            raise AppError(ErrorCode.MODEL_UNAVAILABLE, retryable=False, internal="REMOTE_LLM_BASE_URL not set")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.is_remote = not is_local_url(base_url)

    def generate_json(self, system: str, prompt: str) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = httpx.post(f"{self.base_url}/chat/completions", timeout=self.timeout, headers=headers, json={
                "model": self.model, "temperature": 0.2, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            })
            resp.raise_for_status()
            return _extract_json(resp.json()["choices"][0]["message"]["content"])
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise AppError(ErrorCode.MODEL_UNAVAILABLE, internal=f"remote llm: {type(exc).__name__}") from exc

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "ok": True, "model": self.model, "remote": self.is_remote}


def build_llm_provider(settings: Settings) -> LLMProvider:
    try:
        if settings.LLM_PROVIDER == "ollama":
            return OllamaProvider(settings.OLLAMA_BASE_URL, settings.LLM_MODEL, settings.LLM_TIMEOUT_SECONDS,
                                  settings.ALLOW_REMOTE_PROVIDERS)
        if settings.LLM_PROVIDER == "transformers":
            return TransformersProvider(settings.LLM_MODEL)
        if settings.LLM_PROVIDER == "openai_compatible":
            return OpenAICompatibleProvider(settings.REMOTE_LLM_BASE_URL, settings.REMOTE_LLM_API_KEY,
                                            settings.LLM_MODEL, settings.LLM_TIMEOUT_SECONDS,
                                            settings.ALLOW_REMOTE_PROVIDERS)
    except AppError as err:
        log.warning("LLM provider unavailable, using heuristics", extra={"reason": err.internal})
    return HeuristicProvider()
