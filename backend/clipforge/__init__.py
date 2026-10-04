"""CLIPFORGE AI - local-first long-form video to short-form clipping studio.

The package is organised as:

``clipforge.config``      - configuration + persisted user settings
``clipforge.db``          - SQLite persistence (projects, clips, transcripts, jobs)
``clipforge.media``       - ffmpeg / yt-dlp / reframing / captions / composition
``clipforge.ai``          - transcript understanding, candidate discovery, scoring
``clipforge.pipeline``    - the staged analysis pipeline
``clipforge.jobs``        - background job queue + worker pool
``clipforge.api``         - FastAPI application (REST + SSE + media streaming)
"""

__version__ = "1.0.0"
