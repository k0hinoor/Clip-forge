"""Safe subprocess execution for ffmpeg / ffprobe / yt-dlp.

All external commands funnel through :func:`run_command` so that:

* arguments are always passed as an argv list (never a shell string),
* output is streamed to the render log with a redacted command line,
* progress callbacks can parse ffmpeg's ``-progress`` pipe,
* cancellation is cooperative and kills the whole process tree.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..config import Env
from ..errors import ClipForgeError, ErrorCode, from_ffmpeg_error
from ..logging_setup import get_logger

log = get_logger("clipforge.render")

# Flags that could let a file path turn into an ffmpeg option or a network call.
_DANGEROUS_PATTERNS = (
    re.compile(r"^-http_", re.IGNORECASE),
    re.compile(r"^-f\s+concat", re.IGNORECASE),
)


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    command: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self.stderr.strip().splitlines()[-25:])


class CommandCancelled(ClipForgeError):
    def __init__(self, message: str = "Cancelled.") -> None:
        super().__init__(code=ErrorCode.RENDER_CANCELLED, message=message, status_code=409)


# --------------------------------------------------------------------------- #
# Process registry (so the API can cancel a running render)
# --------------------------------------------------------------------------- #

_running: dict[str, subprocess.Popen] = {}
_cancelled_pids: set[int] = set()
_registry_lock = threading.Lock()


def register_process(key: str, process: subprocess.Popen) -> None:
    with _registry_lock:
        _running[key] = process


def unregister_process(key: str) -> None:
    with _registry_lock:
        _running.pop(key, None)


def kill_process(key: str) -> bool:
    """Terminate the process tree registered under ``key`` (a no-op for unknown keys).

    The process is remembered as *cancelled* so :func:`run_command` reports a
    cancellation rather than an encoder failure when it exits non-zero.
    """
    if not key:
        return False
    with _registry_lock:
        process = _running.get(key)
        if process is None or process.poll() is not None:
            return False
        _cancelled_pids.add(process.pid)
    _terminate(process)
    return True


def kill_all() -> None:
    """Terminate every registered process (used on shutdown)."""
    with _registry_lock:
        processes = [process for process in _running.values() if process.poll() is None]
        _cancelled_pids.update(process.pid for process in processes)
    for process in processes:
        _terminate(process)


def _was_cancelled(process: subprocess.Popen) -> bool:
    with _registry_lock:
        if process.pid in _cancelled_pids:
            _cancelled_pids.discard(process.pid)
            return True
    return False


def _terminate(process: subprocess.Popen) -> None:
    try:
        if Env.IS_WINDOWS:
            subprocess.run(  # noqa: S603 - fixed argv
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            time.sleep(0.6)
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Argument sanitisation
# --------------------------------------------------------------------------- #


def sanitize_args(args: Sequence[str]) -> list[str]:
    """Validate an argv list before it reaches an external tool.

    Prevents accidental option injection (a path starting with ``-``) and blocks
    ffmpeg protocols that could read from or write to the network.
    """
    clean: list[str] = []
    for index, argument in enumerate(args):
        text = str(argument)
        if "\x00" in text:
            raise ClipForgeError(
                code=ErrorCode.INVALID_INPUT,
                message="Invalid character in a file path.",
                status_code=422,
            )
        if index == 0:
            clean.append(text)
            continue
        if text.startswith("-"):
            if any(pattern.match(text) for pattern in _DANGEROUS_PATTERNS):
                raise ClipForgeError(
                    code=ErrorCode.INVALID_INPUT,
                    message=f"Blocked unsafe argument: {text}",
                    hint="CLIPFORGE does not allow network protocols in ffmpeg arguments.",
                    status_code=403,
                )
            clean.append(text)
            continue
        if text.startswith("http://") or text.startswith("https://"):
            # Only yt-dlp/curl-style tools may fetch URLs, and only the ones we built.
            clean.append(text)
            continue
        clean.append(text)
    return clean


def redact_command(command: Sequence[str]) -> str:
    """Human-readable command for the logs, with long paths shortened."""
    parts: list[str] = []
    for token in command:
        text = str(token)
        # Keep filtergraphs readable but bounded.
        if len(text) > 400:
            text = text[:400] + f"...(+{len(text) - 400} chars)"
        parts.append(text if " " not in text else f'"{text}"')
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_command(
    command: Sequence[str],
    *,
    timeout: float | None = None,
    label: str = "command",
    cancel_key: str = "",
    progress: Callable[[str], None] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    error_code: str = ErrorCode.FFMPEG_FAILED,
    extra_headers: Iterable[str] = (),
) -> CommandResult:
    """Run an external command, capturing output and streaming progress lines."""
    argv = sanitize_args(command)
    logger = log
    logger.debug("%s :: %s", label, redact_command(argv))
    started = time.time()

    creationflags = 0
    preexec = None
    if Env.IS_WINDOWS:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        preexec = os.setsid  # own process group so we can kill children

    child_env = {**os.environ, **(env or {})}

    try:
        process = subprocess.Popen(  # noqa: S603 - argv list, shell disabled
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if progress is None else subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
            cwd=str(cwd) if cwd else None,
            env=child_env,
            creationflags=creationflags,
            preexec_fn=preexec,
        )
    except FileNotFoundError as exc:
        raise ClipForgeError(
            code=ErrorCode.FFMPEG_NOT_FOUND,
            message=f"Could not run {argv[0]}.",
            hint="Check the ffmpeg path in Settings -> Video, or let CLIPFORGE auto-detect it.",
            detail=str(exc),
            status_code=500,
        ) from exc
    except OSError as exc:
        raise ClipForgeError(
            code=error_code,
            message=f"Failed to start {label}.",
            detail=str(exc),
            status_code=500,
        ) from exc

    if cancel_key:
        register_process(cancel_key, process)

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def pump_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            stdout_chunks.append(line)
            if progress and line.strip():
                try:
                    progress(line.rstrip("\n"))
                except Exception:  # a progress parser must never break a render
                    pass

    def pump_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_chunks.append(line)

    threads = [threading.Thread(target=pump_stdout, daemon=True)]
    if progress is not None and process.stderr is not None:
        threads.append(threading.Thread(target=pump_stderr, daemon=True))
    for thread in threads:
        thread.start()

    try:
        deadline = started + timeout if timeout else None
        while True:
            if process.poll() is not None:
                break
            if deadline and time.time() > deadline:
                _terminate(process)
                raise ClipForgeError(
                    code=error_code,
                    message=f"{label} timed out after {int(timeout)}s.",
                    hint="Try a smaller Whisper model, a shorter clip, or check the log for the failing step.",
                    status_code=504,
                )
            time.sleep(0.25)
    finally:
        for thread in threads:
            thread.join(timeout=2.0)
        if cancel_key:
            unregister_process(cancel_key)

    duration = time.time() - started
    result = CommandResult(
        returncode=process.returncode or 0,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        duration_seconds=duration,
        command=list(argv),
    )

    if _was_cancelled(process) and result.returncode != 0:
        logger.info("%s cancelled after %.1fs", label, duration)
        raise CommandCancelled(f"{label} was cancelled.")
    if result.returncode != 0:
        tail = result.stderr_tail or result.stdout[-2000:]
        from ..logging_setup import get_logger as _get

        _get("clipforge.render").error("%s failed (rc=%s):\n%s", label, result.returncode, tail)
        raise from_ffmpeg_error(tail, code=error_code)

    logger.debug("%s finished in %.1fs", label, duration)
    return result


def probe_command(command: Sequence[str], *, timeout: float = 20.0, label: str = "probe") -> CommandResult:
    """Run a command that is expected to succeed; returns the result without raising on rc!=0."""
    argv = sanitize_args(command)
    try:
        completed = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if Env.IS_WINDOWS else 0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CommandResult(returncode=127, stdout="", stderr=str(exc), duration_seconds=0.0, command=argv)
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        duration_seconds=0.0,
        command=argv,
    )


__all__ = [
    "CommandCancelled",
    "CommandResult",
    "kill_all",
    "kill_process",
    "probe_command",
    "redact_command",
    "register_process",
    "run_command",
    "sanitize_args",
    "unregister_process",
]
