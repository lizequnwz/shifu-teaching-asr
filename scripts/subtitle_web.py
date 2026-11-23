"""Private loopback entry point for the shared subtitle workflow."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import logging
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from typing import Literal
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile

import subtitle_workflow as engine

ROOT = engine.ROOT
JOBS = engine.JOBS
DIST = ROOT / "web" / "dist"
HOST = "127.0.0.1"
PORT = 8765
TOKEN = secrets.token_urlsafe(32)
PAGE_SIZE = 50
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
CLIP_ID = re.compile(r"audio-[0-9]+-[0-9]+\.wav")
RELEASE_ID = re.compile(r"[a-f0-9]{32}")
Selection = Literal["flagged", "all", "wording", "coverage", "timing", "seam"]
log = logging.getLogger(__name__)


def _read(path, default=None):
    try:
        return engine.read_json(path)
    except FileNotFoundError:
        return default


def _job(job_id: str) -> Path:
    if not SAFE_ID.fullmatch(job_id):
        raise HTTPException(404, "Recording not found.")
    job = JOBS / job_id
    manifest = job / "job.json"
    if job.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        raise HTTPException(404, "Recording not found.")
    if _read(manifest, {}).get("version") != engine.VERSION:
        raise HTTPException(404, "Recording not found.")
    return job


def _files():
    if JOBS.is_dir():
        for job in JOBS.iterdir():
            try:
                yield _job(job.name)
            except (HTTPException, OSError, ValueError, AttributeError):
                continue


class LocalWorker:
    def __init__(self):
        self.guard = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="subtitle-local")
        self.scheduled: set[Path] = set()
        self.stopping = threading.Event()

    def submit(self, job: Path, retry_request: str | None = None):
        with self.guard:
            if job in self.scheduled:
                raise HTTPException(409, "This recording is already queued or processing.")
            operation = {
                "status": "queued", "stage": "queued", "retry_request": retry_request,
                "queued_at": time.time(),
            }
            engine.write_json(job / "operation.json", operation)
            self.scheduled.add(job)
            self.executor.submit(self._run, job, retry_request)

    def _run(self, job, retry_request):
        try:
            while engine.processing_busy():
                if self.stopping.wait(0.5):
                    return
            result, _ = engine.execute(job, engine.read_json(job / "job.json"), kind="resume", retry_request=retry_request)
            operation = _read(job / "operation.json", {})
            if operation.get("status") == "queued":
                engine.write_json(job / "operation.json", {
                    "status": result["status"], "stage": "queued", "detail": result.get("detail"),
                    "retry_request": retry_request,
                })
        except Exception:
            log.exception("Local job processing failed")
            engine.write_json(job / "operation.json", {
                "status": "failed", "stage": "processing", "detail": "Processing failed. See the local server log.",
            })
        finally:
            with self.guard:
                self.scheduled.discard(job)

    def recover(self):
        for job in _files():
            operation = _read(job / "operation.json", {})
            if operation.get("status") in {"queued", "running", "interrupted"}:
                self.submit(job, operation.get("retry_request"))


@asynccontextmanager
async def lifespan(app):
    JOBS.mkdir(parents=True, exist_ok=True, mode=0o700)
    worker = LocalWorker()
    app.state.worker = worker
    worker.recover()
    try:
        yield
    finally:
        worker.stopping.set()
        worker.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Subtitle Studio", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@app.middleware("http")
async def local_security(request: Request, call_next):
    host = request.headers.get("host", "").lower()
    if host not in {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}:
        return JSONResponse({"detail": "Invalid host."}, status_code=403)
    origin = request.headers.get("origin")
    if (origin and origin != f"http://{host}") or request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
        return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
    if request.url.path.startswith("/api/") and request.method not in {"GET", "HEAD", "OPTIONS"}:
        if not hmac.compare_digest(request.headers.get("x-subtitle-token", ""), TOKEN):
            return JSONResponse({"detail": "Invalid request token."}, status_code=403)
        if request.method != "DELETE":
            length = request.headers.get("content-length", "")
            limit = engine.MAX_BYTES + 1024 * 1024 if request.url.path == "/api/jobs" else 2 * 1024 * 1024
            if not length.isascii() or not length.isdigit():
                return JSONResponse({"detail": "A valid content length is required."}, status_code=411)
            if len(length) > 12 or int(length) > limit:
                return JSONResponse({"detail": "The request is too large."}, status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


def _detail(job, draft=None):
    manifest = engine.read_json(job / "job.json")
    name = _read(job / "studio.json", {}).get("name") or Path(manifest["source"]).name
    operation = _read(job / "operation.json", {})
    stage = operation.get("status")
    result = {"id": job.name, "name": name}
    if draft is None:
        draft = engine.current_draft(job)
    publication = engine.published_directory(job)
    if draft is not None:
        summary = engine.read_json(draft / "summary.json")
        result["summary"] = summary
        result["revision"] = engine.identity(engine.read_json(draft / "decisions.json"))
        result["status"] = summary["status"]
        if publication and engine.identity(engine.read_json(publication / "decisions.json")) == result["revision"]:
            result["status"] = "published"
    else:
        result["status"] = {
            "queued": "queued", "running": "processing", "paused": "paused",
            "interrupted": "paused", "failed": "failed",
        }.get(stage, "paused")
        if operation:
            result["operation"] = {key: operation[key] for key in ("stage", "completed", "total") if key in operation}
        if operation.get("detail"):
            result["error"] = operation["detail"]
        if operation.get("retry_request"):
            result["retry_request"] = operation["retry_request"]
    if publication:
        result["published_revision"] = publication.name
    return result


def _page(items, cursor):
    if cursor is not None and (not re.fullmatch(r"[0-9]{1,8}", cursor)):
        raise HTTPException(400, "Invalid page cursor.")
    start = int(cursor or 0)
    end = start + PAGE_SIZE
    return items[start:end], str(end) if end < len(items) else None


@app.get("/api/config")
def config():
    return {"mode": "local", "token": TOKEN}


@app.get("/api/jobs")
def list_jobs(cursor: str | None = None):
    jobs = sorted(_files(), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    page, next_cursor = _page(jobs, cursor)
    return {"jobs": [_detail(job) for job in page], "next_cursor": next_cursor}


@app.get("/api/jobs/{job_id}")
def detail(job_id: str):
    return _detail(_job(job_id))


async def _save_upload(upload, destination):
    written = 0
    with destination.open("xb") as output:
        destination.chmod(0o600)
        while part := await upload.read(1024 * 1024):
            written += len(part)
            if written > engine.MAX_BYTES:
                raise HTTPException(413, "The recording exceeds 512 MiB.")
            output.write(part)
    if not written:
        raise HTTPException(400, "The recording is empty.")


@app.post("/api/jobs", status_code=201)
async def create_job(request: Request):
    async with request.form(max_files=1, max_fields=1, max_part_size=100_001) as form:
        upload, terms = form.get("file"), form.get("terms", "")
        if not isinstance(upload, UploadFile) or not isinstance(terms, str):
            raise HTTPException(400, "Choose a recording.")
        name = Path((upload.filename or "").replace("\\", "/")).name
        suffix = Path(name).suffix.lower()
        if suffix not in engine.AUDIO_SUFFIXES or len(name) > 255:
            raise HTTPException(400, "Choose a supported audio or video file.")
        try:
            vocabulary = engine.normalize_vocabulary(terms.splitlines())
            await asyncio.to_thread(engine.preflight, True)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (RuntimeError, OSError) as exc:
            raise HTTPException(503, str(exc)) from exc
        job = JOBS / f"studio-{uuid4().hex}"
        job.mkdir(mode=0o700)
        source = job / f"recording{suffix}"
        try:
            await _save_upload(upload, source)
            manifest = await asyncio.to_thread(engine.make_manifest, source, vocabulary=vocabulary)
            engine.write_json(job / "studio.json", {"name": name})
            engine.write_json(job / "job.json", manifest)
            request.app.state.worker.submit(job)
        except BaseException:
            shutil.rmtree(job, ignore_errors=True)
            raise
    return _detail(job)


def _passages(job):
    draft = engine.current_draft(job)
    return engine.read_json(draft / "passages.json") if draft else []


def _passage_view(passage):
    from workflow_pipeline import identify_issues

    return {**passage, "issues": identify_issues(passage["issues"]), "clips": [
        {"id": Path(clip["path"]).name, "start_seconds": clip["start_seconds"], "end_seconds": clip["end_seconds"]}
        for clip in passage["clips"]
    ]}


@app.get("/api/jobs/{job_id}/passages")
def passages(job_id: str, filter: Selection = "flagged", cursor: str | None = None):
    if cursor is not None and not re.fullmatch(r"[0-9]{1,8}", cursor):
        raise HTTPException(400, "Invalid page cursor.")
    draft = engine.current_draft(_job(job_id))
    items = engine.read_json(draft / "passages.json") if draft else []
    revision = engine.identity(engine.read_json(draft / "decisions.json")) if draft else None
    page, next_cursor = engine.page_passages(items, filter, int(cursor or 0), PAGE_SIZE)
    return {"passages": [_passage_view(passage) for passage in page], "next_cursor": next_cursor, "revision": revision}


def _clip_path(job, clip_id):
    if not CLIP_ID.fullmatch(clip_id):
        raise HTTPException(404, "Clip not found.")
    relative = f"clips/{clip_id}"
    assigned = any(clip["path"] == relative for item in _passages(job) for clip in item["clips"])
    path = job / relative
    if not assigned or path.parent.is_symlink() or path.is_symlink() or not path.is_file():
        raise HTTPException(404, "Clip not found.")
    return path


@app.get("/api/jobs/{job_id}/clips/{clip_id}")
def sign_clip(job_id: str, clip_id: str):
    _clip_path(_job(job_id), clip_id)
    return {"url": f"/api/jobs/{job_id}/clips/{clip_id}/file"}


@app.get("/api/jobs/{job_id}/clips/{clip_id}/file")
def play_clip(job_id: str, clip_id: str):
    return FileResponse(_clip_path(_job(job_id), clip_id), media_type="audio/wav")


async def _body(request, allowed):
    try:
        body = await request.json()
    except (ValueError, UnicodeError) as exc:
        raise HTTPException(400, "Invalid request.") from exc
    if not isinstance(body, dict) or not set(body) <= allowed:
        raise HTTPException(400, "Invalid request.")
    return body


@app.put("/api/jobs/{job_id}/decisions/{passage_id}")
async def decide(job_id: str, passage_id: str, request: Request):
    job = _job(job_id)
    body = await _body(request, {"choice", "text", "timings", "revision", "resolved_issues"})
    try:
        draft = await asyncio.to_thread(
            engine.save_review_decision, job, passage_id, body.get("choice"), body.get("text"),
            body.get("revision"), body.get("timings"), resolved_issues=body.get("resolved_issues"),
        )
    except (engine.DecisionConflict, engine.WorkflowBusy) as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    passage = next(passage for passage in engine.read_json(draft / "passages.json") if passage["id"] == passage_id)
    return {"job": _detail(job, draft), "passage": _passage_view(passage)}


@app.post("/api/jobs/{job_id}/publish")
async def publish(job_id: str, request: Request):
    job = _job(job_id)
    body = await _body(request, {"revision"})
    try:
        await asyncio.to_thread(engine.publish, job, body.get("revision"))
    except (engine.DecisionConflict, engine.WorkflowBusy) as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _detail(job)


@app.post("/api/jobs/{job_id}/resume")
async def resume(job_id: str, request: Request):
    job = _job(job_id)
    body = await _body(request, {"retry_request"})
    detail = _detail(job)
    if detail["status"] != "paused":
        raise HTTPException(409, "This recording is not paused.")
    retry_request = body.get("retry_request")
    if retry_request != detail.get("retry_request"):
        raise HTTPException(409, "Confirm the outstanding provider request before retrying.")
    request.app.state.worker.submit(job, retry_request)
    return _detail(job)


def _export_path(job, revision, format):
    if not RELEASE_ID.fullmatch(revision):
        raise HTTPException(404, "Published revision not found.")
    directory = job / ".published" / revision
    asset = directory / f"zh.{format}"
    if directory.parent.is_symlink() or directory.is_symlink() or asset.is_symlink() or not asset.is_file():
        raise HTTPException(404, "Published revision not found.")
    return asset


@app.get("/api/jobs/{job_id}/exports/{revision}/{format}")
def export_url(job_id: str, revision: str, format: Literal["txt", "srt"]):
    _export_path(_job(job_id), revision, format)
    return {"url": f"/api/jobs/{job_id}/exports/{revision}/{format}/file"}


@app.get("/api/jobs/{job_id}/exports/{revision}/{format}/file")
def download(job_id: str, revision: str, format: Literal["txt", "srt"]):
    job = _job(job_id)
    path = _export_path(job, revision, format)
    name = Path(_detail(job)["name"]).stem
    media_type = "application/x-subrip" if format == "srt" else "text/plain"
    return FileResponse(path, media_type=f"{media_type}; charset=utf-8",
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}.zh.{format}"})


@app.delete("/api/jobs/{job_id}", status_code=204)
def remove_job(job_id: str, request: Request):
    job = _job(job_id)
    worker = request.app.state.worker
    with worker.guard:
        if job in worker.scheduled:
            raise HTTPException(409, "Wait for this recording to finish processing.")
        try:
            with engine.job_lock(job):
                retired = JOBS / f".deleted-{uuid4().hex}"
                job.replace(retired)
        except engine.WorkflowBusy as exc:
            raise HTTPException(409, str(exc)) from exc
    shutil.rmtree(retired)
    return Response(status_code=204)


@app.get("/{path:path}")
def site(path: str):
    if path == "api" or path.startswith("api/"):
        raise HTTPException(404, "Not found.")
    root = DIST.resolve()
    asset = (root / path).resolve()
    if not asset.is_relative_to(root):
        raise HTTPException(404, "Not found.")
    if path and asset.is_file():
        return FileResponse(asset)
    if (root / "index.html").is_file():
        return FileResponse(root / "index.html")
    return Response("Build the web application with npm --prefix web run build.", status_code=503)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
