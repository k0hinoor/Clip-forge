"""Local LLM integration (Ollama).

The model is used exactly where language understanding beats arithmetic:

* reading transcript chunks and pointing at moments worth clipping,
* describing what a candidate moment actually is (title / hook / category),
* explaining *why* a moment works, in the speaker's own language,
* suggesting an editing plan (zoom points, emphasis words, b-roll hint).

It never renders video, never produces timestamps that are not grounded in the
transcript (candidate times are snapped back to sentence boundaries), and if
Ollama is not running the whole pipeline still works - the analytical engine in
:mod:`clipforge.ai.features` is the base layer and the LLM only blends into it.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import httpx

from ..config import AppSettings, get_settings
from ..errors import ClipForgeError, ErrorCode
from ..logging_setup import get_logger

log = get_logger("clipforge.ai")

SYSTEM_PROMPT = (
    "You are the analysis engine inside CLIPFORGE AI, a local short-form video clipping studio.\n"
    "You read raw speech transcripts and identify the moments that would genuinely work as a "
    "30-75 second vertical short: strong hooks, complete stories, emotional peaks, sharp opinions, "
    "funny exchanges, useful advice, surprising statements and clear payoffs.\n"
    "Rules you never break:\n"
    "1. Only use moments that exist in the transcript you are given - never invent content.\n"
    "2. Use the [hh:mm:ss] timestamps that appear in the transcript; never invent timestamps.\n"
    "3. Never translate. Titles and hooks stay in the language the person actually spoke "
    "(Hinglish stays Hinglish, Hindi stays Hindi).\n"
    "4. A clip must be understandable on its own, with a beginning, a middle and a payoff.\n"
    "5. Be selective: a weak moment is worse than no moment. Do not pad a list to a fixed size.\n"
    "6. Reply with JSON only - no commentary, no markdown fences."
)

DISCOVERY_PROMPT = """Transcript chunk {chunk_index} of {chunk_count} from a {duration_label} video.
Primary language: {language}. Speakers: {speakers}.
Every line starts with a [hh:mm:ss] timestamp. Blank lines separate topic blocks.

Find every moment in THIS CHUNK that would work as a standalone short. Judge honestly - a
{chunk_minutes:.0f}-minute chunk usually contains between 0 and 8 real moments. If there are none,
return an empty list.

Return JSON with this exact shape:
{{
  "moments": [
    {{
      "start": "hh:mm:ss",
      "end": "hh:mm:ss",
      "title": "short punchy title in the spoken language",
      "hook": "the actual sentence that opens the moment",
      "category": "story|emotional|funny|controversial|advice|lesson|information|surprising|curiosity|inspirational|quote|argument|revelation|experience|qa|opinion|punchline|conclusion|hook",
      "score": 0-100,
      "reason": "why this works as a short, referring to the actual words",
      "why": ["short bullet", "short bullet"],
      "context_needed": "none|some|lots"
    }}
  ]
}}

Transcript:
{transcript}"""

ENRICH_PROMPT = """A candidate clip has been found. Write its presentation copy.

Clip duration: {duration:.0f} seconds. Language: {language}.
Transcript of the clip (timestamps are relative to the source video):
{transcript}

Context spoken just before (for understanding only - do not include it):
{context_before}

Return JSON:
{{
  "title": "max 8 words, punchy, in the spoken language",
  "hook": "the single strongest opening line, verbatim if possible",
  "summary": "one sentence describing what happens in this clip",
  "category": "one of the categories listed above",
  "why": ["3-6 short reasons grounded in the transcript"],
  "emphasis_words": ["2-5 words that deserve caption emphasis"],
  "zoom_points": [{{"time": 12.5, "reason": "emotional peak"}}],
  "b_roll_hint": "gameplay|broll|none"
}}"""


@dataclass
class LLMStatus:
    available: bool = False
    base_url: str = ""
    model: str = ""
    models: list[str] = field(default_factory=list)
    error: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "base_url": self.base_url,
            "model": self.model,
            "models": self.models[:20],
            "error": self.error,
            "version": self.version,
        }


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class OllamaClient:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = (self.settings.ollama_base_url or "http://127.0.0.1:11434").rstrip("/")
        self.model = self.settings.ollama_model

    # ------------------------------------------------------------- plumbing
    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout or self.settings.ollama_timeout_seconds, connect=5.0),
        )

    def list_models(self) -> list[str]:
        try:
            with self._client(timeout=6.0) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            log.debug("ollama tags failed: %s", exc)
            return []
        return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]

    def health(self) -> LLMStatus:
        status = LLMStatus(base_url=self.base_url, model=self.model)
        try:
            with self._client(timeout=6.0) as client:
                response = client.get("/api/version")
                if response.status_code == 200:
                    status.version = str(response.json().get("version", ""))
                elif response.status_code == 404:
                    status.version = "unknown"
                else:
                    response.raise_for_status()
        except httpx.HTTPError as exc:
            status.error = _friendly_connection_error(exc, self.base_url)
            return status

        status.models = self.list_models()
        status.available = bool(status.models) or True
        if status.models and self.model and not any(name.startswith(self.model.split(":")[0]) for name in status.models):
            status.error = f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
            status.available = False
        return status

    def ensure_ready(self) -> LLMStatus:
        status = self.health()
        if not status.available:
            raise ClipForgeError(
                code=ErrorCode.OLLAMA_UNAVAILABLE,
                message=f"Ollama is not reachable at {self.base_url}.",
                hint=(
                    status.error
                    or "Start Ollama (ollama serve) and pull a model (ollama pull qwen3:4b), "
                    "or switch Settings -> AI to analytical mode."
                ),
                status_code=503,
            )
        return status

    # ------------------------------------------------------------ inference
    def generate_json(self, prompt: str, *, system: str = SYSTEM_PROMPT, retries: int = 2) -> Any:
        last_error = ""
        for attempt in range(retries + 1):
            try:
                text = self._generate(prompt, system=system, json_mode=True)
            except ClipForgeError:
                raise
            parsed = parse_json_block(text)
            if parsed is not None:
                return parsed
            last_error = text[:500]
            log.warning("ollama returned non-JSON output (attempt %d)", attempt + 1)
            prompt = prompt + "\n\nReturn valid JSON only. No prose, no markdown."
        raise ClipForgeError(
            code=ErrorCode.OLLAMA_ERROR,
            message="The local model did not return usable JSON.",
            hint="Try a larger model (qwen3:4b or better) in Settings -> AI, or use analytical mode.",
            detail=last_error,
            status_code=502,
        )

    def _generate(self, prompt: str, *, system: str, json_mode: bool) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "format": "json" if json_mode else None,
            "options": {
                "temperature": self.settings.ollama_temperature,
                "num_ctx": self.settings.ollama_num_ctx,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
            },
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        try:
            with self._client() as client:
                response = client.post("/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:600] if exc.response is not None else str(exc)
            if exc.response is not None and exc.response.status_code == 404:
                raise ClipForgeError(
                    code=ErrorCode.OLLAMA_ERROR,
                    message=f"The model '{self.model}' is not installed in Ollama.",
                    hint=f"Run:  ollama pull {self.model}",
                    detail=detail,
                    status_code=503,
                ) from exc
            raise ClipForgeError(
                code=ErrorCode.OLLAMA_ERROR,
                message="Ollama rejected the request.",
                hint="Check the model name and that Ollama is up to date.",
                detail=detail,
                status_code=502,
            ) from exc
        except httpx.HTTPError as exc:
            raise ClipForgeError(
                code=ErrorCode.OLLAMA_UNAVAILABLE,
                message=f"Lost connection to Ollama at {self.base_url}.",
                hint=_friendly_connection_error(exc, self.base_url),
                detail=str(exc),
                status_code=503,
            ) from exc
        return str(data.get("response") or "")


def _friendly_connection_error(exc: Exception, base_url: str) -> str:
    if isinstance(exc, httpx.ConnectError):
        return (
            f"Could not connect to Ollama at {base_url}. Start it with 'ollama serve' "
            "(it usually runs as a background service) and confirm the URL in Settings -> AI."
        )
    if isinstance(exc, httpx.TimeoutException):
        return "Ollama did not answer in time. Try a smaller model or raise the timeout in Settings -> AI."
    return str(exc)


def parse_json_block(text: str) -> Any | None:
    """Pull the first JSON object/array out of a model response."""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for position in range(start, len(cleaned)):
            char = cleaned[position]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start: position + 1])
                    except json.JSONDecodeError:
                        break
    return None


# --------------------------------------------------------------------------- #
# Transcript chunking for the discovery prompt
# --------------------------------------------------------------------------- #


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_timestamp(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    match = re.match(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$", text)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?$", text)
    if match:
        return float(match.group(1))
    return None


@dataclass
class TranscriptChunk:
    index: int
    start: float
    end: float
    text: str


def build_chunks(
    sentences: Sequence[Any],
    *,
    max_chars: int = 9000,
    max_chunks: int = 40,
    overlap_sentences: int = 2,
) -> list[TranscriptChunk]:
    """Pack timestamped sentences into LLM-sized chunks (with a small overlap)."""
    if not sentences:
        return []
    total_chars = sum(len(sentence.text) for sentence in sentences)
    char_budget = max(max_chars, total_chars // max(1, max_chunks))

    chunks: list[TranscriptChunk] = []
    buffer: list[str] = []
    size = 0
    chunk_start = sentences[0].start
    previous_block = sentences[0].block

    for sentence in sentences:
        line = f"[{format_timestamp(sentence.start)}] {sentence.text.strip()}"
        block_break = sentence.block != previous_block
        if buffer and (size + len(line) > char_budget or block_break and size > char_budget * 0.55):
            chunks.append(
                TranscriptChunk(
                    index=len(chunks),
                    start=chunk_start,
                    end=_last_timestamp(buffer),
                    text="\n".join(buffer),
                )
            )
            buffer = buffer[-overlap_sentences:]
            size = sum(len(item) for item in buffer)
            chunk_start = sentence.start
        buffer.append(line)
        size += len(line) + 1
        previous_block = sentence.block

    if buffer:
        chunks.append(
            TranscriptChunk(index=len(chunks), start=chunk_start, end=_last_timestamp(buffer), text="\n".join(buffer))
        )

    if len(chunks) > max_chunks:
        step = len(chunks) / max_chunks
        chunks = [chunks[int(index * step)] for index in range(max_chunks)]
        for index, chunk in enumerate(chunks):
            chunk.index = index
    return chunks


def _last_timestamp(lines: Sequence[str]) -> float:
    for line in reversed(lines):
        match = re.match(r"^\[(\d{2}):(\d{2}):(\d{2})\]", line)
        if match:
            return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
    return 0.0


# --------------------------------------------------------------------------- #
# High-level operations
# --------------------------------------------------------------------------- #


def discover_moments(
    sentences: Sequence[Any],
    *,
    language: str = "English",
    speakers: int = 1,
    duration: float = 0.0,
    settings: AppSettings | None = None,
    progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Ask the model to point at candidate moments, chunk by chunk (in parallel)."""
    settings = settings or get_settings()
    client = OllamaClient(settings)
    chunks = build_chunks(sentences, max_chars=settings.llm_chunk_chars, max_chunks=settings.llm_max_chunks)
    if not chunks:
        return []

    log.info("LLM discovery over %d transcript chunks with %s", len(chunks), settings.ollama_model)
    results: list[dict[str, Any]] = []
    completed = 0

    def work(chunk: TranscriptChunk) -> list[dict[str, Any]]:
        prompt = DISCOVERY_PROMPT.format(
            chunk_index=chunk.index + 1,
            chunk_count=len(chunks),
            duration_label=f"{duration / 60:.0f} minute" if duration else "long",
            language=language,
            speakers=speakers,
            chunk_minutes=max((chunk.end - chunk.start) / 60.0, 0.5),
            transcript=chunk.text,
        )
        try:
            payload = client.generate_json(prompt)
        except ClipForgeError as exc:
            log.warning("chunk %d failed: %s", chunk.index, exc.message)
            return []
        moments = payload.get("moments") if isinstance(payload, dict) else payload
        if not isinstance(moments, list):
            return []
        out: list[dict[str, Any]] = []
        for moment in moments:
            if not isinstance(moment, dict):
                continue
            start = parse_timestamp(moment.get("start"))
            end = parse_timestamp(moment.get("end"))
            if start is None or end is None or end <= start:
                continue
            if end - start < 15 or end - start > 180:  # keep obviously broken ranges out
                continue
            out.append({**moment, "start": start, "end": end, "chunk": chunk.index})
        return out

    workers = max(1, min(settings.llm_max_workers, len(chunks)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="clipforge-llm") as pool:
        futures = {pool.submit(work, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            completed += 1
            if should_cancel and should_cancel():
                for pending in futures:
                    pending.cancel()
                break
            try:
                results.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill analysis
                log.warning("LLM chunk failed: %s", exc)
            if progress:
                progress(completed / len(chunks), f"model analysed {completed}/{len(chunks)} transcript chunks")

    log.info("LLM proposed %d raw moments", len(results))
    return results


def enrich_candidate(
    *,
    transcript: str,
    context_before: str,
    duration: float,
    language: str,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """Get presentation copy (title, hook, why, emphasis) for one candidate."""
    settings = settings or get_settings()
    client = OllamaClient(settings)
    prompt = ENRICH_PROMPT.format(
        duration=duration,
        language=language,
        transcript=transcript[:6000],
        context_before=(context_before or "(nothing)")[-1500:],
    )
    payload = client.generate_json(prompt)
    if not isinstance(payload, dict):
        return {}
    return payload


def describe_moment(
    *,
    transcript: str,
    duration: float,
    language: str,
    settings: AppSettings | None = None,
) -> str:
    """Short natural-language explanation of a moment (used by the UI's explain button)."""
    settings = settings or get_settings()
    client = OllamaClient(settings)
    prompt = (
        f"Explain in 2-3 sentences why this {duration:.0f}-second moment from a {language} video "
        f"works (or does not work) as a short-form clip. Be concrete, refer to the actual words, "
        f"and do not invent anything.\n\nTranscript:\n{transcript[:4000]}\n\n"
        'Return JSON: {"explanation": "..."}'
    )
    payload = client.generate_json(prompt)
    if isinstance(payload, dict):
        return str(payload.get("explanation") or payload.get("text") or "").strip()
    return ""


def available_models(settings: AppSettings | None = None) -> list[str]:
    return OllamaClient(settings).list_models()


def status(settings: AppSettings | None = None) -> LLMStatus:
    settings = settings or get_settings()
    if not settings.llm_enabled:
        return LLMStatus(available=False, base_url=settings.ollama_base_url, model=settings.ollama_model, error="Disabled in Settings")
    return OllamaClient(settings).health()


__all__ = [
    "DISCOVERY_PROMPT",
    "ENRICH_PROMPT",
    "LLMStatus",
    "OllamaClient",
    "SYSTEM_PROMPT",
    "TranscriptChunk",
    "available_models",
    "build_chunks",
    "describe_moment",
    "discover_moments",
    "enrich_candidate",
    "format_timestamp",
    "parse_json_block",
    "parse_timestamp",
    "status",
]
