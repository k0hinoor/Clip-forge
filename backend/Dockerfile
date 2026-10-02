# ClipForge backend image (API and worker share it; the command selects the role).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STORAGE_ROOT=/data \
    WHISPER_MODEL_DIR=/models

# ffmpeg/ffprobe (with libass for caption burn-in) + fonts for deterministic captions.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core tini curl \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 clipforge
WORKDIR /app

# Dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY clipforge/__init__.py clipforge/__init__.py
ARG EXTRAS="ai,postgres,s3,redis"
RUN if [ -n "$EXTRAS" ]; then pip install -e ".[${EXTRAS}]"; else pip install -e .; fi

COPY . .
RUN mkdir -p /data /models && chown -R clipforge:clipforge /data /models /app
USER clipforge

EXPOSE 8000
VOLUME ["/data", "/models"]
ENTRYPOINT ["/usr/bin/tini", "--"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1
CMD ["python", "-m", "clipforge.api"]
