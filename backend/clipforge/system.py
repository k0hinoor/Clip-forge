"""Hardware, dependency and capability detection.

Everything here is *measured*, never assumed: CLIPFORGE probes for the ffmpeg
build, the installed AI stack, the GPU and the disk before it decides how to
process a video, and the dashboard shows exactly what it found.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Env, get_settings
from .logging_setup import get_logger

log = get_logger(__name__)

_CACHE_TTL = 120.0
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _cached(key: str, producer) -> Any:
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL:
            return hit[1]
    value = producer()
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def invalidate_cache() -> None:
    with _cache_lock:
        _cache.clear()


# --------------------------------------------------------------------------- #
# Subprocess helper (used only for short probes)
# --------------------------------------------------------------------------- #


def _probe(command: list[str], timeout: float = 8.0) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if Env.IS_WINDOWS else 0,
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)


# --------------------------------------------------------------------------- #
# FFmpeg
# --------------------------------------------------------------------------- #


@dataclass
class FfmpegInfo:
    available: bool = False
    ffmpeg: str = ""
    ffprobe: str = ""
    version: str = ""
    source: str = "missing"
    has_libass: bool = False
    has_drawtext: bool = False
    has_overlay: bool = True
    has_libx264: bool = True
    encoders: list[str] = field(default_factory=list)
    hwaccels: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def encoder_for(self, accel: str) -> str:
        """Pick a video encoder for the requested acceleration mode."""
        if accel == "nvenc" and "h264_nvenc" in self.encoders:
            return "h264_nvenc"
        if accel == "qsv" and "h264_qsv" in self.encoders:
            return "h264_qsv"
        if accel == "amf" and "h264_amf" in self.encoders:
            return "h264_amf"
        if accel == "videotoolbox" and "h264_videotoolbox" in self.encoders:
            return "h264_videotoolbox"
        return "libx264" if self.has_libx264 else "h264"

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "ffmpeg": self.ffmpeg,
            "ffprobe": self.ffprobe,
            "version": self.version,
            "source": self.source,
            "libass": self.has_libass,
            "drawtext": self.has_drawtext,
            "overlay": self.has_overlay,
            "libx264": self.has_libx264,
            "encoders": self.encoders,
            "hwaccels": self.hwaccels,
            "problems": self.problems,
        }


_FFMPEG_CANDIDATES = (
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    "/usr/local/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/bin/ffmpeg",
)


def _find_ffmpeg() -> tuple[str, str, str]:
    """Return ``(ffmpeg, ffprobe, source)`` using settings, PATH and bundles."""
    settings = get_settings()

    if settings.ffmpeg_path and Path(settings.ffmpeg_path).exists():
        ffmpeg = settings.ffmpeg_path
        ffprobe = settings.ffprobe_path or _sibling_probe(ffmpeg)
        return ffmpeg, ffprobe, "configured"

    env_path = os.environ.get("CLIPFORGE_FFMPEG") or os.environ.get("FFMPEG_BINARY")
    if env_path and Path(env_path).exists():
        return env_path, _sibling_probe(env_path), "env"

    try:  # optional bundled binary (imageio-ffmpeg)
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and Path(bundled).exists():
            return bundled, os.environ.get("CLIPFORGE_FFPROBE", ""), "bundled"
    except Exception:  # pragma: no cover - optional dependency
        pass

    found = shutil.which("ffmpeg")
    if found:
        return found, shutil.which("ffprobe") or _sibling_probe(found), "path"

    for candidate in _FFMPEG_CANDIDATES:
        if Path(candidate).exists():
            return candidate, _sibling_probe(candidate), "common-path"

    return "", "", "missing"


def _sibling_probe(ffmpeg_path: str | None) -> str:
    if not ffmpeg_path:
        return ""
    path = Path(ffmpeg_path)
    stem = "ffprobe.exe" if path.suffix.lower() == ".exe" else "ffprobe"
    sibling = path.with_name(stem)
    return str(sibling) if sibling.exists() else ""


def ffmpeg_info(refresh: bool = False) -> FfmpegInfo:
    if refresh:
        invalidate_cache()
    return _cached("ffmpeg", _detect_ffmpeg)


def _detect_ffmpeg() -> FfmpegInfo:
    info = FfmpegInfo()
    ffmpeg, ffprobe, source = _find_ffmpeg()
    if not ffmpeg:
        info.problems.append(
            "ffmpeg was not found. Install it (winget install Gyan.FFmpeg on Windows, "
            "brew install ffmpeg on macOS, apt install ffmpeg on Linux) or run "
            'pip install "clipforge[bundled-ffmpeg]" and set the path in Settings.'
        )
        return info

    info.available = True
    info.ffmpeg = ffmpeg
    info.ffprobe = ffprobe
    info.source = source

    code, out, err = _probe([ffmpeg, "-hide_banner", "-version"])
    banner = (out or "") + (err or "")
    match = re.search(r"ffmpeg version (\S+)", banner)
    info.version = match.group(1) if match else "unknown"
    info.has_libass = "enable-libass" in banner or "--enable-libass" in banner

    code, out, _ = _probe([ffmpeg, "-hide_banner", "-filters"])
    filters = out or ""
    info.has_libass = info.has_libass or bool(re.search(r"\bass\b\s+V->V", filters)) or "subtitles" in filters
    info.has_drawtext = bool(re.search(r"\bdrawtext\b", filters))
    info.has_overlay = bool(re.search(r"\boverlay\b", filters))

    code, out, _ = _probe([ffmpeg, "-hide_banner", "-encoders"])
    encoders = out or ""
    info.encoders = [
        name
        for name in (
            "libx264", "libx265", "h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv",
            "h264_amf", "hevc_amf", "h264_videotoolbox", "aac",
        )
        if re.search(rf"\b{name}\b", encoders)
    ]
    info.has_libx264 = "libx264" in info.encoders

    code, out, _ = _probe([ffmpeg, "-hide_banner", "-hwaccels"])
    info.hwaccels = [line.strip() for line in (out or "").splitlines()[1:] if line.strip()]

    if not ffprobe:
        info.problems.append(
            "ffprobe was not found next to ffmpeg; CLIPFORGE will fall back to parsing ffmpeg output, "
            "which is slower. Installing a full ffmpeg build is recommended."
        )
    if not info.has_libass:
        info.problems.append(
            "This ffmpeg build has no libass; burned-in captions need it. "
            "Install a full build or use the bundled binary in Settings -> Video."
        )
    if not info.has_overlay:
        info.problems.append("This ffmpeg build has no overlay filter; split-screen layouts will not render.")
    return info


# --------------------------------------------------------------------------- #
# Python AI stack
# --------------------------------------------------------------------------- #


def _detect_ai_stack() -> dict[str, Any]:
    stack: dict[str, Any] = {"faster_whisper": False, "whisperx": False, "opencv": False, "torch": False, "torch_cuda": False, "faster_whisper_version": ""}
    try:
        import faster_whisper  # type: ignore

        stack["faster_whisper"] = True
        stack["faster_whisper_version"] = getattr(faster_whisper, "__version__", "unknown")
    except Exception:
        pass
    try:
        import whisperx  # type: ignore  # noqa: F401

        stack["whisperx"] = True
    except Exception:
        pass
    try:
        import cv2  # type: ignore  # noqa: F401

        stack["opencv"] = True
    except Exception:
        pass
    try:
        import torch  # type: ignore

        stack["torch"] = True
        stack["torch_cuda"] = bool(torch.cuda.is_available())
        stack["torch_device"] = torch.cuda.get_device_name(0) if stack["torch_cuda"] else ""
    except Exception:
        pass
    stack["transcription_available"] = bool(stack["faster_whisper"] or stack["whisperx"])
    return stack


def ai_stack(refresh: bool = False) -> dict[str, Any]:
    if refresh:
        invalidate_cache()
    return _cached("ai_stack", _detect_ai_stack)


# --------------------------------------------------------------------------- #
# GPU / CPU / RAM
# --------------------------------------------------------------------------- #


def _detect_nvidia() -> dict[str, Any] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    code, out, _ = _probe(
        [nvidia_smi, "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap", "--format=csv,noheader,nounits"],
        timeout=6,
    )
    if code != 0 or not out.strip():
        return None
    gpus: list[dict[str, Any]] = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "name": parts[0],
                "vram_total_mb": _to_int(parts[1]),
                "vram_free_mb": _to_int(parts[2]),
                "driver": parts[3],
                "compute_capability": parts[4] if len(parts) > 4 else "",
            }
        )
    if not gpus:
        return None
    primary = gpus[0]
    return {
        "available": True,
        "vendor": "NVIDIA",
        "cuda": True,
        "devices": gpus,
        "name": primary["name"],
        "vram_total_mb": primary["vram_total_mb"],
        "vram_free_mb": primary["vram_free_mb"],
        "driver": primary["driver"],
    }


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _detect_generic_gpu() -> dict[str, Any] | None:
    """Best-effort GPU name on Windows/macOS/Linux without vendor tools."""
    if Env.IS_WINDOWS:
        code, out, _ = _probe(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
            timeout=10,
        )
        names = [line.strip() for line in (out or "").splitlines() if line.strip()]
        if names:
            return {"available": True, "vendor": _vendor_from_name(names[0]), "cuda": False, "devices": [{"name": n} for n in names], "name": names[0]}
        return None
    if platform.system() == "Darwin":
        return {"available": True, "vendor": "Apple", "cuda": False, "devices": [], "name": "Apple Silicon GPU"}
    lspci = shutil.which("lspci")
    if lspci:
        code, out, _ = _probe([lspci], timeout=6)
        names = [line.split(": ", 1)[-1] for line in (out or "").splitlines() if re.search(r"VGA|3D|Display", line)]
        if names:
            return {"available": True, "vendor": _vendor_from_name(names[0]), "cuda": False, "devices": [{"name": n} for n in names], "name": names[0]}
    return None


def _vendor_from_name(name: str) -> str:
    low = name.lower()
    for vendor in ("nvidia", "amd", "radeon", "intel", "apple", "qualcomm"):
        if vendor in low:
            return vendor.upper()
    return "unknown"


def cpu_name() -> str:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        code, out, _ = _probe(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=4)
        if out.strip():
            return out.strip()
    return platform.processor() or platform.machine() or "Unknown CPU"


def hardware(refresh: bool = False) -> dict[str, Any]:
    """Full hardware report powering the dashboard + processing decisions."""
    if refresh:
        invalidate_cache()
    return _cached("hardware", _detect_hardware)


def _detect_hardware() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory()
        ram_total = memory.total
        ram_available = memory.available
        cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        cpu_physical = psutil.cpu_count(logical=False) or cpu_count
        cpu_percent = psutil.cpu_percent(interval=None)
    except Exception:  # pragma: no cover - psutil is a hard dependency, but be safe
        ram_total = ram_available = 0
        cpu_count = os.cpu_count() or 1
        cpu_physical = cpu_count
        cpu_percent = 0.0

    gpu = _detect_nvidia() or _detect_generic_gpu()

    disk: dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        usage = psutil.disk_usage(str(Env.DATA_DIR))
        disk = {"total_gb": round(usage.total / 1e9, 1), "free_gb": round(usage.free / 1e9, 1), "used_percent": usage.percent}
    except Exception:
        pass

    ffmpeg = ffmpeg_info()
    stack = ai_stack()

    # Which whisper device will actually be used.
    preferred_device = "cpu"
    if gpu and gpu.get("cuda") and stack.get("torch_cuda"):
        preferred_device = "cuda"
    elif gpu and gpu.get("cuda") and stack.get("faster_whisper"):
        # faster-whisper (CTranslate2) can use CUDA without torch.
        preferred_device = "cuda"

    return {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": {
            "name": cpu_name(),
            "logical_cores": cpu_count,
            "physical_cores": cpu_physical,
            "usage_percent": cpu_percent,
        },
        "memory": {
            "total_gb": round(ram_total / 1e9, 1),
            "available_gb": round(ram_available / 1e9, 1),
            "used_percent": round((1 - (ram_available / ram_total)) * 100, 1) if ram_total else 0.0,
        },
        "gpu": gpu or {"available": False, "vendor": "none", "cuda": False, "devices": []},
        "disk": disk,
        "ffmpeg": ffmpeg.to_dict(),
        "ai": stack,
        "recommended": {
            "whisper_device": preferred_device,
            "whisper_model": recommend_whisper_model(ram_total, gpu),
            "hw_accel": "nvenc" if preferred_device == "cuda" and "h264_nvenc" in ffmpeg.encoders else "none",
            "concurrency": max(1, min(2, cpu_count // 4)),
        },
    }


def recommend_whisper_model(ram_bytes: int, gpu: dict[str, Any] | None) -> str:
    ram_gb = ram_bytes / 1e9
    vram_gb = ((gpu or {}).get("vram_total_mb") or 0) / 1024
    if vram_gb >= 10 or ram_gb >= 32:
        return "large-v3"
    if vram_gb >= 6 or ram_gb >= 16:
        return "medium"
    if ram_gb >= 8:
        return "small"
    return "base"


def disk_report(path: Path | None = None) -> dict[str, Any]:
    target = Path(path or Env.DATA_DIR)
    try:
        import psutil  # type: ignore

        usage = psutil.disk_usage(str(target))
        return {
            "path": str(target),
            "total_gb": round(usage.total / 1e9, 2),
            "free_gb": round(usage.free / 1e9, 2),
            "used_percent": usage.percent,
            "ok": usage.free > 2 * 1024**3,
        }
    except Exception:
        return {"path": str(target), "total_gb": 0, "free_gb": 0, "used_percent": 0, "ok": True}


def ensure_disk_space(required_gb: float = 2.0) -> None:
    from .errors import ClipForgeError, ErrorCode

    report = disk_report()
    if report["free_gb"] and report["free_gb"] < required_gb:
        raise ClipForgeError(
            code=ErrorCode.LOW_DISK,
            message=f"Only {report['free_gb']} GB free where CLIPFORGE stores data.",
            hint="Free up disk space or move the data directory to a larger drive in Settings -> Storage.",
            status_code=507,
        )


def process_snapshot() -> dict[str, Any]:
    """Lightweight live usage numbers for the dashboard header."""
    try:
        import psutil  # type: ignore

        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
            "process_cpu_percent": psutil.Process().cpu_percent(interval=None),
        }
    except Exception:
        return {"cpu_percent": 0.0, "memory_percent": 0.0, "process_memory_mb": 0.0, "process_cpu_percent": 0.0}


def diagnostics() -> dict[str, Any]:
    """Everything the Diagnostics panel shows."""
    from .logging_setup import log_file_paths, recent_errors

    ffmpeg = ffmpeg_info()
    stack = ai_stack()
    return {
        "paths": {
            "data_dir": str(Env.DATA_DIR),
            "database": str(Env.DB_PATH),
            "logs": str(Env.LOG_DIR),
            "exports": str(get_settings().resolved_export_dir()),
        },
        "logs": log_file_paths(),
        "ffmpeg": ffmpeg.to_dict(),
        "ai": stack,
        "errors": recent_errors(30),
        "python": sys.version,
        "platform": platform.platform(),
    }


__all__ = [
    "FfmpegInfo",
    "ai_stack",
    "diagnostics",
    "disk_report",
    "ensure_disk_space",
    "ffmpeg_info",
    "hardware",
    "invalidate_cache",
    "process_snapshot",
    "recommend_whisper_model",
]
