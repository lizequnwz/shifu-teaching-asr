"""PostgreSQL storage for private jobs, worker leases, and provider reservations."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import math
import os
from uuid import UUID, uuid4

import psycopg
from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from workflow_cloud import BudgetExceeded, RequestLedger, RequestPending


JOB_LIMIT_SECONDS = 120 * 60
LEASE_SECONDS = 90
SUMMARY_STATUSES = {"needs_review", "ready_to_publish", "no_speech"}


class JobError(ValueError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class LeaseLost(RuntimeError):
    """The worker may no longer write to this job."""


def object_prefix(job_id: UUID) -> str:
    return f"jobs/{job_id}"


def _source_key(job_id: UUID) -> str:
    return f"{object_prefix(job_id)}/source"


def _summary_status(summary: dict) -> str:
    status = summary["status"]
    if status not in SUMMARY_STATUSES:
        raise ValueError("Invalid status for a review preview")
    return status


def _redact(db, job_id):
    db.execute("DELETE FROM studio.windows WHERE job_id = %s", (job_id,))
    db.execute("DELETE FROM studio.publications WHERE job_id = %s", (job_id,))
    db.execute("""UPDATE studio.requests SET fingerprint = NULL, request = NULL, raw = NULL,
        result = NULL, usage = NULL, provider_request_id = NULL WHERE job_id = %s""", (job_id,))


class OnlineStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    @contextmanager
    def connection(self):
        with psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=5, prepare_threshold=None) as db:
            yield db

    def health(self):
        with self.connection() as db:
            db.execute("SELECT 1 FROM studio.jobs LIMIT 1")

    def _owned(self, db, owner_id, job_id, *, lock=False):
        query = "SELECT * FROM studio.jobs WHERE owner_id = %s AND id = %s AND status <> 'deleted' AND expires_at > now()"
        if lock:
            query += " FOR UPDATE"
        job = db.execute(query, (owner_id, job_id)).fetchone()
        if job is None:
            raise JobError(404, "Job not found.")
        return job

    def get_job(self, owner_id, job_id):
        with self.connection() as db:
            return self._owned(db, owner_id, job_id)

    def list_jobs(self, owner_id, cursor, limit=20):
        columns = "id, name, status, stage, completed, total, error, retry_request, revision, published_revision, summary, source_bytes"
        with self.connection() as db:
            if cursor is None:
                return db.execute(f"""SELECT {columns} FROM studio.jobs
                    WHERE owner_id = %s AND status <> 'deleted' AND expires_at > now()
                    ORDER BY created_at DESC, id DESC LIMIT %s""", (owner_id, limit + 1)).fetchall()
            anchor = db.execute("SELECT created_at FROM studio.jobs WHERE id = %s AND owner_id = %s",
                                (cursor, owner_id)).fetchone()
            if anchor is None:
                raise JobError(400, "Invalid job cursor.")
            return db.execute(f"""SELECT {columns} FROM studio.jobs
                WHERE owner_id = %s AND status <> 'deleted' AND expires_at > now()
                  AND (created_at, id) < (%s, %s)
                ORDER BY created_at DESC, id DESC LIMIT %s""",
                              (owner_id, anchor["created_at"], cursor, limit + 1)).fetchall()

    def create_job(self, owner_id, name, byte_count, mime_type, terms):
        try:
            with self.connection() as db:
                return db.execute("""INSERT INTO studio.jobs (
                    id, owner_id, name, source_bytes, mime_type, terms, status, stage, revision
                ) VALUES (%s, %s, %s, %s, %s, %s, 'uploading', 'uploading', %s) RETURNING *""",
                                  (uuid4(), owner_id, name, byte_count, mime_type, Jsonb(terms), uuid4())).fetchone()
        except UniqueViolation:
            raise JobError(409, "Finish or delete your active job before starting another.") from None

    def upload_grant(self, owner_id, job_id):
        with self.connection() as db:
            row = self._owned(db, owner_id, job_id, lock=True)
            if row["status"] != "uploading":
                raise JobError(409, "This job is past the upload stage.")
            # A Supabase signed token can create a TUS upload for two hours, and the
            # resulting upload URL can remain usable for a further 24 hours.
            db.execute("UPDATE studio.jobs SET upload_grant_until = now() + interval '26 hours' WHERE id = %s",
                       (job_id,))
            return _source_key(job_id)

    def complete_upload(self, owner_id, job_id, byte_count):
        with self.connection() as db:
            job = self._owned(db, owner_id, job_id, lock=True)
            if job["status"] != "uploading":
                return job
            if byte_count != job["source_bytes"]:
                raise JobError(409, "The uploaded file size differs from the selected file.")
            return db.execute("""UPDATE studio.jobs SET status = 'queued', stage = 'queued', available_at = now()
                WHERE id = %s RETURNING *""", (job_id,)).fetchone()

    def resume(self, owner_id, job_id, request_id):
        try:
            with self.connection() as db:
                job = self._owned(db, owner_id, job_id, lock=True)
                if job["status"] != "paused":
                    raise JobError(409, "Only a paused job can be resumed.")
                if job["stage"] == "rate_limited" and db.execute("SELECT %s > now() AS cooling_down",
                                                                  (job["available_at"],)).fetchone()["cooling_down"]:
                    raise JobError(409, "The speech provider is still rate-limiting requests. Try again later.")
                if job["retry_request"] != request_id:
                    raise JobError(409, "Confirm the current uncertain request before retrying it.")
                return db.execute("""UPDATE studio.jobs SET status = 'queued', stage = 'queued',
                    available_at = now(), authorized_retry = %s, retry_request = NULL, error = NULL,
                    rate_limit_count = 0 WHERE id = %s RETURNING *""", (request_id, job_id)).fetchone()
        except UniqueViolation:
            raise JobError(409, "Finish or delete your active job before resuming another.") from None

    def save_review(self, owner_id, job_id, revision, decisions, summary, passages):
        with self.connection() as db:
            job = self._owned(db, owner_id, job_id, lock=True)
            if job["revision"] != revision or job["data"] is None:
                raise JobError(409, "The review changed. Reload before saving.")
            if job["decisions"] == decisions:
                return job
            return db.execute("""UPDATE studio.jobs SET decisions = %s, summary = %s, passages = %s,
                status = %s, revision = %s WHERE id = %s RETURNING *""",
                              (Jsonb(decisions), Jsonb(summary), Jsonb(passages), _summary_status(summary),
                               uuid4(), job_id)).fetchone()

    def publish(self, owner_id, job_id, revision, object_directory):
        with self.connection() as db:
            job = self._owned(db, owner_id, job_id, lock=True)
            if job["revision"] == revision and job["published_revision"] == revision:
                return job, False
            if job["status"] != "ready_to_publish" or job["revision"] != revision:
                raise JobError(409, "The current revision is not ready to publish.")
            db.execute("INSERT INTO studio.publications (job_id, revision, object_prefix) VALUES (%s, %s, %s)",
                       (job_id, revision, object_directory))
            job = db.execute("""UPDATE studio.jobs SET published_revision = %s, status = 'published'
                WHERE id = %s RETURNING *""", (revision, job_id)).fetchone()
            return job, True

    def get_export(self, owner_id, job_id, revision):
        with self.connection() as db:
            job = self._owned(db, owner_id, job_id, lock=True)
            snapshot = db.execute("SELECT object_prefix FROM studio.publications WHERE job_id = %s AND revision = %s",
                                  (job_id, revision)).fetchone()
            if snapshot is None:
                raise JobError(404, "Publication not found.")
            return job["name"], snapshot["object_prefix"]

    def delete(self, owner_id, job_id):
        with self.connection() as db:
            self._owned(db, owner_id, job_id, lock=True)
            self._tombstone(db, job_id)

    def _tombstone(self, db, job_id):
        db.execute("""UPDATE studio.jobs SET status = 'deleted', stage = 'deleted', lease = NULL,
            lease_until = NULL, cleanup_at = now(), name = '', terms = '[]'::jsonb, error = NULL,
            upload_grant_until = GREATEST(upload_grant_until, now() + interval '5 minutes'),
            retry_request = NULL, authorized_retry = NULL, manifest = NULL, data = NULL, summary = NULL, passages = NULL,
            decisions = '{}'::jsonb, clip_prefix = NULL, published_revision = NULL WHERE id = %s""", (job_id,))
        _redact(db, job_id)

    def expire_and_next_cleanup(self):
        with self.connection() as db:
            expired = db.execute("""SELECT id FROM studio.jobs WHERE status <> 'deleted' AND
                (expires_at <= now() OR (status = 'uploading' AND created_at <= now() - interval '24 hours'))
                ORDER BY created_at LIMIT 30 FOR UPDATE SKIP LOCKED""").fetchall()
            for row in expired:
                self._tombstone(db, row["id"])
            cleanup = db.execute("""SELECT id FROM studio.jobs WHERE status = 'deleted' AND cleanup_at <= now()
                ORDER BY cleanup_at, id LIMIT 1""").fetchone()
            return cleanup["id"] if cleanup else None

    def cleanup_done(self, job_id):
        with self.connection() as db:
            db.execute("""UPDATE studio.jobs SET cleanup_at = CASE WHEN upload_grant_until > now()
                THEN upload_grant_until ELSE NULL END WHERE id = %s AND status = 'deleted'""", (job_id,))
            db.execute("""DELETE FROM studio.jobs WHERE status = 'deleted' AND cleanup_at IS NULL
                AND created_at < now() - interval '30 days'
                AND (duration_reserved_at IS NULL OR duration_reserved_at < now() - interval '30 days')""")

    def claim(self):
        with self.connection() as db:
            return db.execute("""WITH next_job AS (
                    SELECT id FROM studio.jobs WHERE expires_at > now() AND
                        ((status = 'queued' AND available_at <= now()) OR
                         (status = 'processing' AND lease_until <= now()))
                    ORDER BY available_at, id LIMIT 1 FOR UPDATE SKIP LOCKED
                ) UPDATE studio.jobs AS job SET status = 'processing', stage = 'audio_preflight', completed = NULL, total = NULL,
                    lease = %s, lease_until = now() + make_interval(secs => %s), error = NULL
                FROM next_job WHERE job.id = next_job.id RETURNING job.*""", (uuid4(), LEASE_SECONDS)).fetchone()

    def renew(self, job_id, lease):
        with self.connection() as db:
            return bool(db.execute("""UPDATE studio.jobs SET lease_until = now() + make_interval(secs => %s)
                WHERE id = %s AND lease = %s AND status = 'processing'
                AND lease_until > now() AND expires_at > now()""", (LEASE_SECONDS, job_id, lease)).rowcount)

    def assert_lease(self, db, job_id, lease):
        job = db.execute("""SELECT * FROM studio.jobs WHERE id = %s AND lease = %s AND status = 'processing'
            AND lease_until > now() AND expires_at > now() FOR UPDATE""", (job_id, lease)).fetchone()
        if job is None:
            raise LeaseLost("The job lease is no longer active.")
        return job

    def progress(self, job_id, lease, stage, completed=None, total=None):
        with self.connection() as db:
            self.assert_lease(db, job_id, lease)
            db.execute("UPDATE studio.jobs SET stage = %s, completed = %s, total = %s WHERE id = %s",
                       (stage, completed, total, job_id))

    def authorize_duration(self, job_id, lease, duration):
        if not math.isfinite(duration) or not 0 < duration <= JOB_LIMIT_SECONDS:
            raise ValueError("Recordings must be at most 120 minutes.")
        duration = math.ceil(duration)
        with self.connection() as db:
            job = self.assert_lease(db, job_id, lease)
            if job["duration_seconds"] is not None:
                if job["duration_seconds"] != duration:
                    raise ValueError("The recording changed after processing started.")
                return
            db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 1))", (str(job["owner_id"]),))
            usage = db.execute("""SELECT COALESCE(sum(duration_seconds), 0) AS used FROM studio.jobs
                WHERE owner_id = %s AND duration_reserved_at >= now() - interval '30 days'""",
                               (job["owner_id"],)).fetchone()["used"]
            if usage + duration > JOB_LIMIT_SECONDS:
                raise BudgetExceeded("This account has reached its rolling 120-minute allowance.")
            db.execute("UPDATE studio.jobs SET duration_seconds = %s, duration_reserved_at = now() WHERE id = %s",
                       (duration, job_id))

    def windows(self, job_id, lease):
        with self.connection() as db:
            self.assert_lease(db, job_id, lease)
            return db.execute("SELECT name, result FROM studio.windows WHERE job_id = %s ORDER BY name", (job_id,)).fetchall()

    def pin_manifest(self, job_id, lease, manifest):
        with self.connection() as db:
            job = self.assert_lease(db, job_id, lease)
            if job["manifest"] is not None:
                if job["manifest"] != manifest:
                    raise ValueError("The recording or processing version changed. Start a new job.")
                return
            db.execute("UPDATE studio.jobs SET manifest = %s WHERE id = %s", (Jsonb(manifest), job_id))

    def save_window(self, job_id, lease, name, result):
        with self.connection() as db:
            self.assert_lease(db, job_id, lease)
            db.execute("""INSERT INTO studio.windows (job_id, name, result) VALUES (%s, %s, %s)
                ON CONFLICT (job_id, name) DO UPDATE SET result = EXCLUDED.result""", (job_id, name, Jsonb(result)))

    def finish(self, job_id, lease, data, summary, passages, clip_prefix):
        with self.connection() as db:
            self.assert_lease(db, job_id, lease)
            db.execute("""UPDATE studio.jobs SET data = %s, summary = %s, passages = %s, clip_prefix = %s,
                status = %s, stage = 'complete', lease = NULL, lease_until = NULL, authorized_retry = NULL
                WHERE id = %s""", (Jsonb(data), Jsonb(summary), Jsonb(passages), clip_prefix, _summary_status(summary), job_id))

    def stop(self, job_id, lease, status, reason, retry_request=None):
        with self.connection() as db:
            self.assert_lease(db, job_id, lease)
            db.execute("""UPDATE studio.jobs SET status = %s, error = %s, retry_request = %s, authorized_retry = NULL,
                lease = NULL, lease_until = NULL WHERE id = %s""", (status, reason, retry_request, job_id))

    def backoff(self, job_id, lease, delay):
        with self.connection() as db:
            job = self.assert_lease(db, job_id, lease)
            attempts = job["rate_limit_count"] + 1
            status = "paused" if attempts >= 5 else "queued"
            db.execute("""UPDATE studio.jobs SET status = %s, stage = 'rate_limited', rate_limit_count = %s,
                available_at = now() + make_interval(secs => %s), lease = NULL, lease_until = NULL,
                error = 'A speech provider is rate-limiting requests.' WHERE id = %s""",
                       (status, attempts, max(delay, 15 * 2 ** (attempts - 1)), job_id))


class OnlineLedger(RequestLedger):
    inline_retries = False

    def __init__(self, store, job, daily_limit):
        self.store = store
        self.job_id = job["id"]
        self.lease = job["lease"]
        self.daily_limit = daily_limit
        self.retry_request = job["authorized_retry"]

    def _latest(self, fingerprint, db=None):
        if db is None:
            with self.store.connection() as connection:
                return self._latest(fingerprint, connection)
        return db.execute("""SELECT state, result FROM studio.requests WHERE job_id = %s AND fingerprint = %s
            ORDER BY id DESC LIMIT 1""", (self.job_id, fingerprint)).fetchone()

    def _reserve(self, fingerprint, provider, model, purpose, request, amount):
        with self.store.connection() as db:
            db.execute("SELECT pg_advisory_xact_lock(hashtextextended('subtitle-studio-daily-budget', 0))")
            job = self.store.assert_lease(db, self.job_id, self.lease)
            latest = self._latest(fingerprint, db)
            if latest and latest["state"] == "completed":
                return None, latest["result"]
            if latest and latest["state"] in {"pending", "invalid_response"} and job["authorized_retry"] != fingerprint:
                raise RequestPending(fingerprint, provider, "has an unresolved prior attempt")
            usage = db.execute("""SELECT COALESCE(sum(reserved_usd), 0) AS spent FROM studio.requests
                WHERE created_at >= (date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')""").fetchone()
            if usage["spent"] + amount > self.daily_limit:
                raise BudgetExceeded("The site's daily transcription budget has been reached. Try again after 00:00 UTC.")
            attempt = db.execute("""INSERT INTO studio.requests
                (job_id, fingerprint, provider, model, purpose, request, state, reserved_usd)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s) RETURNING id""",
                                 (self.job_id, fingerprint, provider, model, purpose, Jsonb(json.loads(request)), amount)).fetchone()
            if job["authorized_retry"] == fingerprint:
                db.execute("UPDATE studio.jobs SET authorized_retry = NULL WHERE id = %s", (self.job_id,))
                self.retry_request = None
            return attempt["id"], None

    def _store(self, attempt, state, *, raw=None, result=None, usage=None, estimate=None, error=None,
               committed=None, provider_request_id=None):
        with self.store.connection() as db:
            self.store.assert_lease(db, self.job_id, self.lease)
            db.execute("""UPDATE studio.requests SET state = %s, raw = %s, result = %s, usage = %s,
                estimated_usd = %s, error_kind = %s, reserved_usd = COALESCE(%s, reserved_usd), provider_request_id = %s
                WHERE id = %s AND job_id = %s""",
                       (state, Jsonb(raw), Jsonb(result), Jsonb(usage), estimate, error, committed,
                        provider_request_id, attempt, self.job_id))


if __name__ == "__main__":
    store = OnlineStore(os.environ["DATABASE_URL"])
    directory = Path(__file__).resolve().parents[1] / "migrations"
    with store.connection() as database:
        database.execute("SELECT pg_advisory_xact_lock(hashtextextended('subtitle-studio-migrations', 0))")
        for migration in sorted(directory.glob("*.sql")):
            database.execute(migration.read_text("utf-8"))
