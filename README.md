# ClipForge

ClipForge is a local-first AI video clipping tool. You upload a long video, and ClipForge finds the strongest moments, frames them for vertical platforms, burns in captions, and exports ready-to-post clips. By default everything runs on your own hardware.

| Directory | Status |
|---|---|
| [`backend/`](backend/README.md) | API, worker and AI pipeline (FastAPI, SQLAlchemy, faster-whisper, FFmpeg) |
| `frontend/` | Next.js app (planned) |

Specifications: [TRD](ClipForge_TRD.md), [PRD](ClipForge_PRD.md), [combined](ClipForge_PRD_TRD.md).

```bash
cp backend/.env.example backend/.env
docker compose up --build        # http://localhost:8000/api/v1/docs
```
