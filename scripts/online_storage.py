"""Supabase Auth and Storage calls for the hosted Studio."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from uuid import UUID

import httpx

from online_store import JobError


MAX_UPLOAD_BYTES = 50_000_000
MEDIA_SUFFIXES = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".mp4", ".mov", ".webm",
                  ".ogg", ".aiff", ".aif", ".opus"}


class StorageError(RuntimeError):
    """The private object store could not complete an operation."""


class ObjectMissing(StorageError):
    """An object does not exist or an upload is not complete."""


def _legacy_role(key):
    try:
        parts = key.split(".")
        return json.loads(base64.urlsafe_b64decode(parts[1] + "==")).get("role") if len(parts) == 3 else None
    except (ValueError, AttributeError, UnicodeError):
        return None


@dataclass(frozen=True)
class OnlineSettings:
    database_url: str
    supabase_url: str
    publishable_key: str
    service_key: str
    bucket: str
    tus_url: str
    allowed_regions: tuple[str, ...]
    origin: str
    daily_budget: float

    @classmethod
    def from_environment(cls):
        project_url = os.environ["SUPABASE_URL"].rstrip("/")
        origin = os.environ["STUDIO_PUBLIC_ORIGIN"].rstrip("/")
        for value in (project_url, origin):
            parsed = urlsplit(value)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query or parsed.fragment
                    or parsed.username is not None or parsed.password is not None):
                raise ValueError("Use HTTPS origins for the Supabase project and Studio site.")
        hostname = urlsplit(project_url).hostname
        storage_host = project_url
        if re.fullmatch(r"[a-z0-9]+\.supabase\.co", hostname):
            project = hostname.split(".")[0]
            storage_host = f"https://{project}.storage.supabase.co"
        bucket = os.environ.get("STUDIO_BUCKET", "recordings")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,63}", bucket):
            raise ValueError("Invalid private storage bucket name.")
        allowed_regions = tuple(sorted(set(os.environ["ONLINE_ALLOWED_REGIONS"].upper().replace(",", " ").split())))
        if not allowed_regions or not all(re.fullmatch(r"[A-Z]{2}", region) for region in allowed_regions):
            raise ValueError("Set ONLINE_ALLOWED_REGIONS to supported ISO country codes.")
        daily_budget = float(os.environ["ONLINE_DAILY_BUDGET_USD"])
        if not math.isfinite(daily_budget) or daily_budget <= 0:
            raise ValueError("Set a positive ONLINE_DAILY_BUDGET_USD.")
        publishable_key = os.environ["SUPABASE_PUBLISHABLE_KEY"]
        service_key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        if ((not publishable_key.startswith("sb_publishable_") and _legacy_role(publishable_key) != "anon") or
                (not service_key.startswith("sb_secret_") and _legacy_role(service_key) != "service_role")):
            raise ValueError("Use distinct publishable and service keys for the Supabase project.")
        return cls(os.environ["DATABASE_URL"], project_url, publishable_key, service_key, bucket,
                   f"{storage_host}/storage/v1/upload/resumable/sign", allowed_regions, origin, daily_budget)


class Supabase:
    def __init__(self, settings: OnlineSettings, client=None):
        self.settings = settings
        self.client = client or httpx.Client(timeout=httpx.Timeout(60, connect=5), trust_env=False)
        self.storage_root = settings.supabase_url + "/storage/v1"
        self.storage_headers = {"apikey": settings.service_key}
        if settings.service_key.startswith("eyJ"):
            self.storage_headers["Authorization"] = f"Bearer {settings.service_key}"

    def authorize(self, authorization: str | None):
        header = authorization or ""
        scheme, space, token = header.partition(" ")
        if scheme.lower() != "bearer" or not space or not 1 <= len(token) <= 16_384 or re.search(r"\s", token):
            raise JobError(401, "Sign in to access this job.")
        try:
            response = self.client.get(f"{self.settings.supabase_url}/auth/v1/user",
                                       headers={"apikey": self.settings.publishable_key, "Authorization": f"Bearer {token}"},
                                       timeout=8)
            if response.status_code in {400, 401, 403, 422}:
                raise JobError(401, "Your sign-in session expired. Sign in again.")
            response.raise_for_status()
            user = response.json()
            owner_id = UUID(user["id"])
        except JobError:
            raise
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            raise JobError(503, "Sign-in verification is temporarily unavailable.") from None
        if not user.get("email_confirmed_at") or not user.get("email") or user.get("is_anonymous"):
            raise JobError(403, "Verify your email to access Studio.")
        metadata = user.get("user_metadata")
        if not isinstance(metadata, dict):
            raise JobError(403, "Confirm your age and a supported region to access Studio.")
        country = metadata.get("country")
        if metadata.get("adult") is not True or not isinstance(country, str) or country.upper() not in self.settings.allowed_regions:
            raise JobError(403, "Confirm your age and a supported region to access Studio.")
        return owner_id

    def _object(self, key: str):
        return f"{self.settings.bucket}/{quote(key, safe='/')}"

    def _storage(self, method, path, *, headers=None, missing=False, **kwargs):
        try:
            response = self.client.request(method, self.storage_root + path,
                                           headers={**self.storage_headers, **(headers or {})}, **kwargs)
            if missing and response.status_code in (400, 404):
                raise ObjectMissing("The file is unavailable in private storage.")
            response.raise_for_status()
        except httpx.HTTPError:
            raise StorageError("Private storage is temporarily unavailable.") from None
        return response

    def upload_token(self, key):
        response = self._storage("POST", f"/object/upload/sign/{self._object(key)}", json={})
        try:
            url = response.json()["url"]
            token = parse_qs(urlsplit(url).query)["token"][0]
        except (KeyError, ValueError, TypeError, IndexError):
            raise StorageError("Private storage returned an invalid upload grant.") from None
        return {"endpoint": self.settings.tus_url, "bucket": self.settings.bucket, "object_key": key, "token": token}

    def size(self, key):
        response = self._storage("HEAD", f"/object/{self._object(key)}", headers={"Accept-Encoding": "identity"}, missing=True)
        try:
            return int(response.headers["content-length"])
        except (KeyError, ValueError):
            raise StorageError("Private storage omitted the file size.") from None

    def download(self, key, destination: Path, size: int):
        path = f"/object/{self._object(key)}"
        total = 0
        try:
            with self.client.stream("GET", self.storage_root + path,
                                    headers={**self.storage_headers, "Accept-Encoding": "identity"}) as response:
                if response.status_code in (400, 404):
                    raise ObjectMissing("The recording is unavailable in private storage.")
                response.raise_for_status()
                with destination.open("xb") as stream:
                    for block in response.iter_raw():
                        total += len(block)
                        if total > size:
                            raise ValueError("The recording exceeds its confirmed size.")
                        stream.write(block)
        except httpx.HTTPError:
            raise StorageError("Could not read the recording from private storage.") from None
        if total != size:
            raise ValueError("The recording differs from its confirmed size.")

    def put(self, key, path: Path, content_type: str):
        headers = {"Content-Type": content_type, "Content-Length": str(path.stat().st_size),
                   "Cache-Control": "max-age=0", "x-upsert": "false"}
        with path.open("rb") as stream:
            self._storage("POST", f"/object/{self._object(key)}", headers=headers, content=stream)

    def signed_url(self, key, *, filename: str | None = None):
        response = self._storage("POST", f"/object/sign/{self._object(key)}", json={"expiresIn": 300}, missing=True)
        try:
            path = response.json()["signedURL"]
            if not isinstance(path, str) or not path.startswith("/object/sign/") or not urlsplit(path).query:
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise StorageError("Private storage returned an invalid download link.") from None
        if filename:
            path += "&" + urlencode({"download": filename}, quote_via=quote)
        return self.storage_root + path

    def remove_tree(self, prefix):
        directories = [prefix]
        objects = []
        while directories:
            directory = directories.pop()
            offset = 0
            while True:
                response = self._storage("POST", f"/object/list/{self.settings.bucket}", json={
                    "prefix": directory, "limit": 100, "offset": offset, "sortBy": {"column": "name", "order": "asc"},
                })
                entries = response.json()
                for entry in entries:
                    name = entry.get("name")
                    if not isinstance(name, str) or "/" in name or name in {"", ".", ".."}:
                        raise StorageError("Private storage returned an invalid directory entry.")
                    key = f"{directory}/{name}"
                    if entry.get("id") is None:
                        directories.append(key)
                    else:
                        objects.append(key)
                if len(entries) < 100:
                    break
                offset += len(entries)
        for offset in range(0, len(objects), 100):
            self._storage("DELETE", f"/object/{self.settings.bucket}", json={"prefixes": objects[offset:offset + 100]})
