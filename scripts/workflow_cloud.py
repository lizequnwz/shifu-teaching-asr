"""Shared request cache and estimated spending for a subtitle job."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import wave


class CloudError(RuntimeError):
    """A provider could not complete a request."""


class BudgetExceeded(CloudError):
    """The next request exceeds the job's estimated allowance."""


class RateLimited(CloudError):
    """The provider declined the request and its cooldown exceeded the retry."""

    def __init__(self, message: str, retry_after: float = 30):
        super().__init__(message)
        self.retry_after = retry_after


class RequestPending(CloudError):
    def __init__(self, request_id: str, provider: str, detail: str):
        self.request_id = request_id
        super().__init__(f"{provider} {detail}. Its reservation is retained. Retry request ID: {request_id}")


def encode_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audio_clip(path: Path, limit: int) -> tuple[bytes, float]:
    with Path(path).open("rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError("The WAV clip exceeds the provider upload limit")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        raise ValueError("Expected a valid WAV clip") from None
    if not 0 < duration <= 3600:
        raise ValueError("Clip duration must be positive and at most one hour")
    return data, duration


def api_key(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.is_file():
        try:
            lines = env.read_text("utf-8-sig").splitlines()
        except (OSError, UnicodeError):
            raise CloudError("The project .env could not be read as UTF-8") from None
        for line in lines:
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and key.strip() in names:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if value:
                    return value
    raise CloudError(f"Set {' or '.join(names)} in the environment or project .env")


def response_data(response) -> dict:
    if isinstance(response, dict):
        return response
    return response.model_dump(mode="json", exclude_none=True)


def token_cost(usage: dict, input_rate: float, output_rate: float, *,
               input_key: str, output_key: str, thought_key: str | None = None) -> float | None:
    inputs = usage.get(input_key)
    outputs = usage.get(output_key)
    thoughts = (usage.get(thought_key) or 0) if thought_key else 0
    counts = (inputs, outputs, thoughts)
    if not all(isinstance(n, (int, float)) and math.isfinite(n) and n >= 0 for n in counts) or inputs == 0:
        return None
    return (inputs * input_rate + (outputs + thoughts) * output_rate) / 1_000_000


def _retry_delay(error):
    headers = getattr(getattr(error, "response", None), "headers", {})
    delay = headers.get("Retry-After") if headers else None
    try:
        seconds = float(delay)
    except (TypeError, ValueError):
        return 30
    return max(12, seconds) if math.isfinite(seconds) else 30


class RequestLedger:
    inline_retries = True

    def request(self, provider: str, model: str, purpose: str, details: dict, reservation: float,
                prepare, send, parse, cost):
        request = encode_json({"provider": provider, "model": model, "purpose": purpose, "details": details})
        fingerprint = digest(request.encode("utf-8"))
        latest = self._latest(fingerprint)
        if latest and latest["state"] == "completed":
            return latest["result"]
        if latest and latest["state"] in {"pending", "invalid_response"} and self.retry_request != fingerprint:
            raise RequestPending(fingerprint, provider, "has an unresolved prior attempt")
        client = prepare()
        for retry in range(2 if self.inline_retries else 1):
            attempt, cached = self._reserve(fingerprint, provider, model, purpose, request, reservation)
            if attempt is None:
                return cached
            try:
                response = send(client)
            except Exception as exc:
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                code = status if isinstance(status, int) else None
                error = type(exc).__name__ + (f" HTTP {code}" if code else "")
                request_id = getattr(exc, "request_id", None)
                if code == 429:
                    self._store(attempt, "rate_limited", error=error, committed=0, provider_request_id=request_id)
                    delay = _retry_delay(exc)
                    if self.inline_retries and retry == 0 and delay <= 60:
                        time.sleep(delay)
                        continue
                    raise RateLimited(f"{provider} rate-limited {purpose}; resume in about {math.ceil(delay)} seconds", delay) from None
                if code is not None and 400 <= code < 500 and code not in {408, 409, 425}:
                    self._store(attempt, "rejected", error=error, committed=0, provider_request_id=request_id)
                    raise CloudError(f"{provider} rejected {purpose} (HTTP {code})") from None
                self._store(attempt, "pending", error=error, provider_request_id=request_id)
                raise RequestPending(fingerprint, provider, f"could not complete {purpose} ({error})") from None
            raw = usage = estimate = None
            request_id = getattr(response, "_request_id", None)
            try:
                raw = response_data(response)
                request_id = request_id or raw.get("id")
                usage = raw.get("usage") or {}
                estimate = cost(usage)
                result = parse(response, raw)
            except (AttributeError, TypeError, ValueError, KeyError):
                self._store(attempt, "invalid_response", raw=raw, usage=usage, estimate=estimate,
                            error="invalid_or_incomplete_response", provider_request_id=request_id)
                raise RequestPending(fingerprint, provider, "returned an invalid response for " + purpose) from None
            committed = estimate * 1.25 if estimate is not None else None
            self._store(attempt, "completed", raw=raw, result=result, usage=usage, estimate=estimate,
                        committed=committed, provider_request_id=request_id)
            return result


class CloudLedger(RequestLedger):
    def __init__(self, job: Path, budget_usd: float, *, retry_request: str | None = None):
        if not math.isfinite(budget_usd) or budget_usd < 0:
            raise ValueError("Budget must be a finite, nonnegative USD amount")
        self.budget_usd = float(budget_usd)
        self.retry_request = retry_request
        job = Path(job)
        job.mkdir(parents=True, exist_ok=True)
        self.path = job / "cloud-requests.sqlite3"
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        with self._connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL,
                provider TEXT NOT NULL, model TEXT NOT NULL, purpose TEXT NOT NULL,
                request_json TEXT NOT NULL, state TEXT NOT NULL, reserved_usd REAL NOT NULL,
                response_json TEXT, result_json TEXT, usage_json TEXT, estimated_usd REAL,
                provider_request_id TEXT, error_kind TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            db.execute("CREATE INDEX IF NOT EXISTS attempts_fingerprint ON attempts(fingerprint, id)")

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _reserve(self, fingerprint: str, provider: str, model: str, purpose: str,
                 request: str, amount: float):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            latest = db.execute("SELECT state,result_json FROM attempts WHERE fingerprint=? ORDER BY id DESC LIMIT 1",
                                (fingerprint,)).fetchone()
            if latest and latest["state"] == "completed":
                return None, json.loads(latest["result_json"])
            uncertain = latest and latest["state"] in {"pending", "invalid_response"}
            if uncertain and self.retry_request != fingerprint:
                raise RequestPending(fingerprint, provider, "has an unresolved prior attempt")
            committed = db.execute("SELECT COALESCE(SUM(reserved_usd), 0) FROM attempts").fetchone()[0]
            remaining = max(0.0, self.budget_usd - committed)
            if committed + amount > self.budget_usd + 1e-12:
                raise BudgetExceeded(f"{provider} needs an estimated ${amount:.4f} for {purpose}; ${remaining:.4f} remains")
            attempt = db.execute("""INSERT INTO attempts
                (fingerprint,provider,model,purpose,request_json,state,reserved_usd)
                VALUES (?,?,?,?,?,'pending',?)""",
                (fingerprint, provider, model, purpose, request, amount)).lastrowid
        if self.retry_request == fingerprint:
            self.retry_request = None
        return attempt, None

    def _store(self, attempt: int, state: str, *, raw=None, result=None, usage=None,
               estimate=None, error=None, committed=None, provider_request_id=None):
        with self._connection() as db:
            db.execute("""UPDATE attempts SET state=?, response_json=?, result_json=?, usage_json=?,
                estimated_usd=?, error_kind=?, reserved_usd=COALESCE(?,reserved_usd),
                provider_request_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                (state, None if raw is None else encode_json(raw), None if result is None else encode_json(result),
                 None if usage is None else encode_json(usage), estimate, error, committed, provider_request_id, attempt))

    def usage(self) -> dict:
        with self._connection() as db:
            db.execute("BEGIN")
            rows = db.execute("SELECT state,reserved_usd,estimated_usd FROM attempts").fetchall()
            unresolved = db.execute("""SELECT fingerprint AS request_id,provider,model,purpose,state,error_kind AS error
                FROM attempts WHERE id IN (SELECT MAX(id) FROM attempts GROUP BY fingerprint)
                AND (state='pending' OR state='invalid_response') ORDER BY id""").fetchall()
        committed = sum(row["reserved_usd"] for row in rows)
        estimated = sum(row["estimated_usd"] or 0 for row in rows)
        by_status = {}
        for row in rows:
            state = row["state"]
            by_status[state] = by_status.get(state, 0) + 1
        return {"budget_usd": self.budget_usd, "estimated_spent_usd": estimated,
                "reserved_usd": max(0.0, committed - estimated), "exposure_usd": committed,
                "remaining_usd": max(0.0, self.budget_usd - committed), "by_status": by_status,
                "unresolved_requests": [dict(row) for row in unresolved],
                "note": "Local estimates can differ from provider billing."}

    def _latest(self, fingerprint):
        with self._connection() as db:
            row = db.execute("SELECT state,result_json FROM attempts WHERE fingerprint=? ORDER BY id DESC LIMIT 1",
                             (fingerprint,)).fetchone()
        if row:
            return {"state": row["state"], "result": json.loads(row["result_json"]) if row["result_json"] is not None else None}
