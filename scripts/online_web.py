"""Hosted browser API. All recording access passes through an owner check."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import json
import re
from tempfile import TemporaryDirectory
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
import psycopg

from online_storage import MAX_UPLOAD_BYTES, MEDIA_SUFFIXES, ObjectMissing, OnlineSettings, StorageError, Supabase
from online_store import JobError, OnlineStore, object_prefix


DIST = Path(__file__).resolve().parents[1] / "web" / "dist"
CLIP_NAME = re.compile(r"[A-Za-z0-9._-]{1,120}\.wav\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
BLOCK_ID = re.compile(r"[0-9]+-[0-9]+\Z")
FILTERS = Literal["flagged", "all", "wording", "coverage", "timing", "seam"]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UploadRequest(Input):
    name: str = Field(min_length=1, max_length=220)
    bytes: int = Field(strict=True, ge=1, le=MAX_UPLOAD_BYTES)
    mime_type: str = Field(max_length=100)
    terms: list[str] = Field(default_factory=list, max_length=100)


class RevisionRequest(Input):
    revision: UUID


class TimingInput(Input):
    start_ms: int = Field(strict=True)
    end_ms: int = Field(strict=True)
    text: str = Field(max_length=400)


class DecisionRequest(RevisionRequest):
    choice: Literal["gemini", "openai", "suggestion", "edited", "inaudible", "no_speech"] | None = None
    text: str | None = Field(default=None, max_length=24_000)
    timings: list[TimingInput] | None = Field(default=None, max_length=100)
    resolved_issues: list[str] | None = Field(default=None, max_length=100)


class ResumeRequest(Input):
    retry_request: str | None = Field(default=None, max_length=64)


def _job_view(job):
    operation = {"stage": job["stage"]}
    for key in ("completed", "total"):
        if job.get(key) is not None:
            operation[key] = job[key]
    result = {"id": str(job["id"]), "name": job["name"], "status": job["status"], "operation": operation,
              "summary": job["summary"], "revision": str(job["revision"]),
              "published_revision": str(job["published_revision"]) if job["published_revision"] else None}
    if job["status"] == "uploading":
        result["upload_bytes"] = job["source_bytes"]
    for key in ("error", "retry_request"):
        if job.get(key):
            result[key] = job[key]
    return result


def _clip_id(clip):
    name = Path(clip["path"]).name
    if not CLIP_NAME.fullmatch(name) or clip["path"] != f"clips/{name}":
        raise ValueError("A review clip has an invalid path.")
    return name


def _passage_view(passage):
    from workflow_pipeline import identify_issues

    clips = [{"id": _clip_id(clip), "start_seconds": clip["start_seconds"], "end_seconds": clip["end_seconds"]}
             for clip in passage["clips"]]
    return {**passage, "clips": clips, "issues": identify_issues(passage["issues"])}


def _upload_fields(body):
    from subtitle_workflow import normalize_vocabulary

    name = body.name.strip()
    if not name or not name.isprintable() or "/" in name or "\\" in name or Path(name).suffix.lower() not in MEDIA_SUFFIXES:
        raise JobError(400, "Select a supported audio or video file.")
    mime_type = body.mime_type or "application/octet-stream"
    if not re.fullmatch(r"(?:audio|video|application)/[a-zA-Z0-9._+-]+", mime_type):
        raise JobError(400, "Select a supported audio or video file.")
    try:
        terms = normalize_vocabulary(body.terms)
    except ValueError as exc:
        raise JobError(400, str(exc)) from None
    return name, mime_type, terms


def _render(job, decisions, directory):
    from workflow_pipeline import build_review

    summary = build_review(job["data"], decisions, directory)
    passages = json.loads((directory / "passages.json").read_text("utf-8"))
    return summary, passages


def create_app(settings=None, *, store=None, supabase=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.settings = settings or OnlineSettings.from_environment()
        app.state.store = store or OnlineStore(app.state.settings.database_url)
        app.state.supabase = supabase or Supabase(app.state.settings)
        try:
            yield
        finally:
            if supabase is None:
                app.state.supabase.client.close()

    app = FastAPI(title="Subtitle Studio", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    def owner(request: Request, authorization: str | None = Header(default=None)):
        return request.app.state.supabase.authorize(authorization)

    @app.exception_handler(JobError)
    async def job_error(_request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(StorageError)
    async def storage_error(_request, exc):
        code = 404 if isinstance(exc, ObjectMissing) else 503
        return JSONResponse({"detail": str(exc)}, status_code=code)

    @app.exception_handler(psycopg.Error)
    async def database_error(_request, _exc):
        return JSONResponse({"detail": "Studio's data store is temporarily unavailable."}, status_code=503)

    @app.middleware("http")
    async def origin_and_headers(request: Request, call_next):
        settings = request.app.state.settings
        origin = request.headers.get("origin")
        if request.url.path.startswith("/api/") and origin and origin != settings.origin:
            return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
        if request.url.path.startswith("/api/") and request.method in {"POST", "PUT"}:
            length = request.headers.get("content-length", "0")
            if request.headers.get("transfer-encoding") or not re.fullmatch(r"[0-9]{1,8}", length) or int(length) > 512 * 1024:
                return JSONResponse({"detail": "Request body is too large."}, status_code=413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        tus_origin = urlsplit(settings.tus_url)
        network = f"{settings.supabase_url} {tus_origin.scheme}://{tus_origin.netloc}"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            f"connect-src 'self' {network}; media-src 'self' blob: {network}; worker-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/health")
    def health(request: Request):
        request.app.state.store.health()
        return {"healthy": True}

    @app.get("/api/config")
    def config(request: Request):
        settings = request.app.state.settings
        return {"mode": "online", "supabase_url": settings.supabase_url,
                "supabase_publishable_key": settings.publishable_key, "allowed_regions": settings.allowed_regions,
                "max_upload_bytes": MAX_UPLOAD_BYTES}

    @app.get("/api/jobs")
    def list_jobs(request: Request, owner_id=Depends(owner), cursor: UUID | None = None):
        jobs = request.app.state.store.list_jobs(owner_id, cursor)
        return {"jobs": [_job_view(job) for job in jobs[:20]],
                "next_cursor": str(jobs[19]["id"]) if len(jobs) > 20 else None}

    @app.post("/api/jobs", status_code=201)
    def create_job(body: UploadRequest, request: Request, owner_id=Depends(owner)):
        name, mime_type, terms = _upload_fields(body)
        store = request.app.state.store
        job = store.create_job(owner_id, name, body.bytes, mime_type, terms)
        key = store.upload_grant(owner_id, job["id"])
        return {"job": _job_view(job), "upload": request.app.state.supabase.upload_token(key)}

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: UUID, request: Request, owner_id=Depends(owner)):
        return _job_view(request.app.state.store.get_job(owner_id, job_id))

    @app.post("/api/jobs/{job_id}/upload-token")
    def refresh_upload(job_id: UUID, request: Request, owner_id=Depends(owner)):
        key = request.app.state.store.upload_grant(owner_id, job_id)
        return request.app.state.supabase.upload_token(key)

    @app.post("/api/jobs/{job_id}/upload-complete")
    def complete_upload(job_id: UUID, request: Request, owner_id=Depends(owner)):
        store = request.app.state.store
        job = store.get_job(owner_id, job_id)
        if job["status"] != "uploading":
            return _job_view(job)
        try:
            size = request.app.state.supabase.size(f"{object_prefix(job_id)}/source")
        except ObjectMissing:
            raise JobError(409, "Finish the recording upload before starting transcription.") from None
        if size > MAX_UPLOAD_BYTES:
            raise JobError(413, "The recording exceeds the online upload limit.")
        return _job_view(store.complete_upload(owner_id, job_id, size))

    @app.post("/api/jobs/{job_id}/resume")
    def resume(job_id: UUID, body: ResumeRequest, request: Request, owner_id=Depends(owner)):
        if body.retry_request is not None and not SHA256.fullmatch(body.retry_request):
            raise JobError(400, "Invalid request identifier.")
        return _job_view(request.app.state.store.resume(owner_id, job_id, body.retry_request))

    @app.get("/api/jobs/{job_id}/passages")
    def passages(job_id: UUID, request: Request, owner_id=Depends(owner), filter: FILTERS = "flagged",
                 cursor: int = Query(default=0, ge=0, le=1_000_000)):
        from subtitle_workflow import page_passages

        job = request.app.state.store.get_job(owner_id, job_id)
        page, next_cursor = page_passages(job["passages"] or [], filter, cursor, 30)
        return {"passages": [_passage_view(row) for row in page],
                "next_cursor": next_cursor, "revision": str(job["revision"])}

    @app.put("/api/jobs/{job_id}/decisions/{passage_id}")
    def decide(job_id: UUID, passage_id: str, body: DecisionRequest, request: Request, owner_id=Depends(owner)):
        from workflow_pipeline import validate_decision

        store = request.app.state.store
        job = store.get_job(owner_id, job_id)
        if job["revision"] != body.revision or job["data"] is None:
            raise JobError(409, "The review changed. Reload before saving.")
        if not BLOCK_ID.fullmatch(passage_id) or not any(row["id"] == passage_id for row in job["passages"]):
            raise JobError(404, "Passage not found.")
        timings = [cue.model_dump() for cue in body.timings] if body.timings is not None else None
        try:
            decision = validate_decision(
                job["data"], passage_id, body.choice, body.text, timings,
                previous=job["decisions"].get(passage_id), resolved_issues=body.resolved_issues,
            )
        except ValueError as exc:
            raise JobError(400, str(exc)) from None
        decisions = {**job["decisions"], passage_id: decision}
        with TemporaryDirectory(prefix="studio-review-") as temp:
            summary, passages = _render(job, decisions, Path(temp))
        updated = store.save_review(owner_id, job_id, body.revision, decisions, summary, passages)
        passage = next(row for row in updated["passages"] if row["id"] == passage_id)
        return {"job": _job_view(updated), "passage": _passage_view(passage)}

    @app.post("/api/jobs/{job_id}/publish")
    def publish(job_id: UUID, body: RevisionRequest, request: Request, owner_id=Depends(owner)):
        store = request.app.state.store
        supabase = request.app.state.supabase
        job = store.get_job(owner_id, job_id)
        if job["revision"] == body.revision and job["published_revision"] == body.revision:
            return _job_view(job)
        if job["status"] != "ready_to_publish" or job["revision"] != body.revision:
            raise JobError(409, "Review the current draft before publishing.")
        prefix = f"{object_prefix(job_id)}/publications/{uuid4()}"
        with TemporaryDirectory(prefix="studio-publish-") as temp:
            directory = Path(temp)
            summary, _ = _render(job, job["decisions"], directory)
            if summary["status"] != "ready_to_publish":
                raise JobError(409, "Resolve the outstanding review issues before publishing.")
            for filename, content_type in (("zh.txt", "text/plain; charset=utf-8"), ("zh.srt", "application/x-subrip")):
                supabase.put(f"{prefix}/{filename}", directory / filename, content_type)
        try:
            current, stored = store.publish(owner_id, job_id, body.revision, prefix)
        except JobError:
            supabase.remove_tree(prefix)
            raise
        if not stored:
            supabase.remove_tree(prefix)
        return _job_view(current)

    @app.get("/api/jobs/{job_id}/clips/{clip_id}")
    def clip(job_id: UUID, clip_id: str, request: Request, owner_id=Depends(owner)):
        store = request.app.state.store
        job = store.get_job(owner_id, job_id)
        if not CLIP_NAME.fullmatch(clip_id) or not job["clip_prefix"] or not any(
            _clip_id(clip) == clip_id for passage in job["passages"] or [] for clip in passage["clips"]
        ):
            raise JobError(404, "Clip not found.")
        url = request.app.state.supabase.signed_url(f"{job['clip_prefix']}/{clip_id}")
        store.get_job(owner_id, job_id)
        return {"url": url}

    @app.get("/api/jobs/{job_id}/exports/{revision}/{format}")
    def export(job_id: UUID, revision: UUID, format: Literal["txt", "srt"], request: Request, owner_id=Depends(owner)):
        store = request.app.state.store
        name, prefix = store.get_export(owner_id, job_id, revision)
        filename = f"{Path(name).stem}.zh.{format}"
        url = request.app.state.supabase.signed_url(f"{prefix}/zh.{format}", filename=filename)
        store.get_job(owner_id, job_id)
        return {"url": url}

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete(job_id: UUID, request: Request, owner_id=Depends(owner)):
        request.app.state.store.delete(owner_id, job_id)
        return Response(status_code=204)

    @app.get("/{path:path}", include_in_schema=False)
    def site(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not found.")
        target = (DIST / path).resolve()
        if path and target.is_relative_to(DIST) and target.is_file():
            return FileResponse(target)
        index = DIST / "index.html"
        if not index.is_file():
            raise HTTPException(503, "The frontend build is unavailable.")
        return FileResponse(index)

    return app


app = create_app()
