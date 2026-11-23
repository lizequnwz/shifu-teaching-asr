import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from unittest.mock import patch
from uuid import uuid4
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    import httpx
    import psycopg
except ImportError:
    httpx = psycopg = None

if httpx is not None:
    from fastapi.testclient import TestClient
    from online_storage import ObjectMissing, OnlineSettings, Supabase
    from online_store import JobError, LeaseLost, OnlineLedger, OnlineStore
    from online_web import create_app
    from online_worker import process
    from workflow_cloud import BudgetExceeded, RequestPending, digest


def settings(database="unused"):
    return OnlineSettings(database, "https://project.example", "sb_publishable_studio", "sb_secret_studio",
                          "recordings", "https://storage.example/storage/v1/upload/resumable", ("US",),
                          "http://testserver", 2.0)


@skipUnless(httpx is not None, "Install requirements-online.txt for hosted tests")
class HostedAccessTests(TestCase):
    def test_auth_checks_verified_user_eligibility(self):
        owner = uuid4()

        def response(request):
            self.assertEqual(request.url.path, "/auth/v1/user")
            self.assertEqual(request.headers.get("apikey"), "sb_publishable_studio")
            token = request.headers["authorization"]
            if token == "Bearer expired":
                return httpx.Response(401)
            country = "XX" if token == "Bearer ineligible" else "US"
            return httpx.Response(200, json={"id": str(owner), "email": "editor@example.test",
                "email_confirmed_at": "2026-01-01", "user_metadata": {"adult": True, "country": country}})

        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            service = Supabase(settings(), client)
            self.assertEqual(service.authorize("Bearer valid"), owner)
            for header, status in ((None, 401), ("Bearer expired", 401), ("Bearer ineligible", 403)):
                with self.assertRaises(JobError) as raised:
                    service.authorize(header)
                self.assertEqual(raised.exception.status, status)

    def test_storage_signs_only_the_requested_path_and_sets_download_name(self):
        def response(request):
            self.assertEqual(request.headers["apikey"], "sb_secret_studio")
            self.assertNotIn("authorization", request.headers)
            if request.url.path.startswith("/storage/v1/object/upload/sign/"):
                self.assertEqual(request.url.path, "/storage/v1/object/upload/sign/recordings/jobs/test/source")
                self.assertEqual(json.loads(request.content), {})
                return httpx.Response(200, json={"url": "/object/upload/sign/path?token=signed-upload"})
            if request.url.path.startswith("/storage/v1/object/sign/"):
                self.assertEqual(json.loads(request.content), {"expiresIn": 300})
                return httpx.Response(200, json={"signedURL": "/object/sign/path?token=signed-read"})
            self.fail(f"Unexpected storage path {request.url.path}")

        with httpx.Client(transport=httpx.MockTransport(response)) as client:
            service = Supabase(settings(), client)
            grant = service.upload_token("jobs/test/source")
            self.assertEqual(grant["object_key"], "jobs/test/source")
            self.assertEqual(grant["token"], "signed-upload")
            link = httpx.URL(service.signed_url("jobs/test/zh.srt", filename="师父.zh.srt"))
            self.assertEqual(link.host, "project.example")
            self.assertEqual(link.params["download"], "师父.zh.srt")


class MemoryStorage:
    def __init__(self):
        self.objects = {}

    def authorize(self, authorization):
        if not authorization or not authorization.startswith("Bearer "):
            raise JobError(401, "Sign in to access this job.")
        from uuid import UUID
        return UUID(authorization.removeprefix("Bearer "))

    def upload_token(self, key):
        return {"endpoint": "https://storage.example/resumable", "bucket": "recordings",
                "object_key": key, "token": "fake"}

    def size(self, key):
        if key not in self.objects:
            raise ObjectMissing("Object missing.")
        return len(self.objects[key])

    def download(self, key, destination, size):
        assert self.size(key) == size
        destination.write_bytes(self.objects[key])

    def put(self, key, path, _content_type):
        if key in self.objects:
            raise ValueError("Refused to replace an immutable object")
        self.objects[key] = path.read_bytes()

    def signed_url(self, key, *, filename=None):
        return f"https://storage.example/{key}?download={filename or ''}"

    def remove_tree(self, prefix):
        for key in tuple(self.objects):
            if key.startswith(prefix + "/"):
                del self.objects[key]


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 32_000)
    return output.getvalue()


class FakeGemini:
    def __init__(self, _directory, _daily_budget, *, ledger):
        self.ledger = ledger

    def transcribe_timed(self, clip):
        result = {"text": "师父慈悲。", "words": [
            {"text": "师父", "text_start": 0, "text_end": 2, "start_seconds": 0.1, "end_seconds": 0.7},
            {"text": "慈悲", "text_start": 2, "text_end": 4, "start_seconds": 0.75, "end_seconds": 1.5},
        ]}
        return self.ledger.request("Gemini", "test-gemini", "timed", {"audio": digest(clip.read_bytes())},
                                   0.01, lambda: None, lambda _: result, lambda _response, raw: raw, lambda _: 0.001)


class FakeOpenAI:
    def __init__(self, _directory, _daily_budget, *, vocabulary, ledger):
        self.ledger = ledger
        self.vocabulary = vocabulary

    def transcribe(self, clip, *, purpose):
        return self.ledger.request("OpenAI", "test-openai", purpose,
                                   {"audio": digest(clip.read_bytes()), "terms": self.vocabulary},
                                   0.01, lambda: None, lambda _: {"text": "师父慈悲。"},
                                   lambda _response, raw: raw["text"], lambda _: 0.001)


@skipUnless(httpx is not None and os.environ.get("STUDIO_TEST_POSTGRES_BIN"),
            "Set STUDIO_TEST_POSTGRES_BIN to the bin directory of a PostgreSQL test installation")
class HostedPostgresTests(TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory(prefix="studio-pg-", dir="/tmp")
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        data = root / "data"
        binary = Path(os.environ["STUDIO_TEST_POSTGRES_BIN"])
        subprocess.run([binary / "initdb", "-D", data, "-A", "trust", "-U", "studio_test",
                        "--no-locale", "-E", "UTF8"], check=True, capture_output=True)
        subprocess.run([binary / "pg_ctl", "-D", data, "-l", root / "postgres.log", "-o",
                        f"-h '' -k {root} -p 55423", "-w", "start"], check=True, capture_output=True)
        cls.addClassCleanup(subprocess.run, [binary / "pg_ctl", "-D", data, "-m", "fast", "-w", "stop"],
                            check=True, capture_output=True)
        cls.base_connection = f"host={root} user=studio_test port=55423"

    def setUp(self):
        name = "studio_test_" + uuid4().hex
        with psycopg.connect(f"{self.base_connection} dbname=postgres", autocommit=True) as connection:
            connection.execute(psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(name)))
        self.store = OnlineStore(f"{self.base_connection} dbname={name}")
        migration = Path(__file__).resolve().parents[1] / "migrations" / "001_online.sql"
        with self.store.connection() as database:
            database.execute(migration.read_text("utf-8"))

    def test_queued_worker_review_publish_delete_and_tenant_boundaries(self):
        owner = {"Authorization": f"Bearer {uuid4()}"}
        other = {"Authorization": f"Bearer {uuid4()}"}
        bucket = MemoryStorage()
        recording = wav_bytes()
        with TestClient(create_app(settings(), store=self.store, supabase=bucket)) as client:
            config = client.get("/api/config").json()
            self.assertEqual(config["allowed_regions"], ["US"])
            self.assertEqual(config["max_upload_bytes"], 50_000_000)
            self.assertNotIn("sb_secret", json.dumps(config))
            create = {"name": "师父.wav", "bytes": len(recording), "mime_type": "audio/wav", "terms": ["师父"]}
            self.assertEqual(client.get("/api/jobs").status_code, 401)
            self.assertEqual(client.post("/api/jobs", json={**create, "bytes": 50_000_001}, headers=owner).status_code, 422)
            response = client.post("/api/jobs", json=create, headers=owner)
            self.assertEqual(response.status_code, 201, response.text)
            result = response.json()
            job = result["job"]
            self.assertEqual(job["upload_bytes"], len(recording))
            url = f"/api/jobs/{job['id']}"
            self.assertEqual(client.post("/api/jobs", json=create, headers=owner).status_code, 409)
            self.assertEqual(client.get("/api/jobs", headers=other).json()["jobs"], [])
            self.assertEqual(client.get(url, headers=other).status_code, 404)
            self.assertEqual(client.post(url + "/upload-token", headers=other).status_code, 404)
            self.assertEqual(client.post(url + "/upload-complete", headers=owner).status_code, 409)
            with patch.object(bucket, "size", return_value=50_000_001):
                self.assertEqual(client.post(url + "/upload-complete", headers=owner).status_code, 413)
            source = result["upload"]["object_key"]
            bucket.objects[source] = recording
            self.assertEqual(client.post(url + "/upload-complete", headers=owner).json()["status"], "queued")
            self.assertEqual(client.post(url + "/upload-complete", headers=owner).json()["status"], "queued")

            lease = self.store.claim()
            with patch("online_worker.GeminiClient", FakeGemini), patch("online_worker.OpenAIClient", FakeOpenAI):
                process(self.store, bucket, lease, 2.0)
            ready = client.get(url, headers=owner).json()
            self.assertEqual(ready["status"], "ready_to_publish", ready)
            review_page = client.get(url + "/passages?filter=all", headers=owner).json()
            self.assertEqual(review_page["revision"], ready["revision"])
            review = review_page["passages"][0]
            self.assertEqual(review["selected_text"], "师父慈悲。")
            clip = url + "/clips/" + review["clips"][0]["id"]
            self.assertEqual(client.get(clip, headers=owner).status_code, 200)
            self.assertEqual(client.get(clip, headers=other).status_code, 404)

            old_revision = ready["revision"]
            edit = {"revision": old_revision, "choice": "edited", "text": "师父，慈悲。"}
            review_url = url + "/decisions/" + review["id"]
            self.assertEqual(client.put(review_url, json=edit, headers=other).status_code, 404)
            edited = client.put(review_url, json=edit, headers=owner).json()
            self.assertEqual(edited["job"]["status"], "ready_to_publish", edited)
            self.assertEqual(edited["passage"]["selected_text"], "师父，慈悲。")
            self.assertEqual(client.put(review_url, json=edit, headers=owner).status_code, 409)
            self.assertEqual(client.post(url + "/publish", json={"revision": old_revision}, headers=owner).status_code, 409)
            revision = edited["job"]["revision"]
            self.assertEqual(client.get(url + f"/exports/{revision}/srt", headers=owner).status_code, 404)
            published = client.post(url + "/publish", json={"revision": revision}, headers=owner).json()
            self.assertEqual(published["published_revision"], revision)
            self.assertTrue(any("师父，慈悲。" in blob.decode() for key, blob in bucket.objects.items() if key.endswith("zh.srt")))
            self.assertEqual(client.get(url + f"/exports/{revision}/srt", headers=other).status_code, 404)
            self.assertIn("师父.zh.srt", client.get(url + f"/exports/{revision}/srt", headers=owner).json()["url"])

            self.assertEqual(client.delete(url, headers=other).status_code, 404)
            self.assertEqual(client.delete(url, headers=owner).status_code, 204)
            self.assertEqual(client.get(clip, headers=owner).status_code, 404)
            self.assertEqual(client.get(url + f"/exports/{revision}/srt", headers=owner).status_code, 404)
            with self.assertRaises(LeaseLost):
                self.store.progress(lease["id"], lease["lease"], "late")
            expired = self.store.expire_and_next_cleanup()
            self.assertEqual(str(expired), job["id"])
            bucket.remove_tree("jobs/" + job["id"])
            self.store.cleanup_done(expired)
            self.assertEqual(bucket.objects, {})

    def test_lease_reclaims_and_uncertain_request_keeps_daily_reservation(self):
        owner = uuid4()
        job = self.store.create_job(owner, "a.wav", 4, "audio/wav", [])
        self.store.complete_upload(owner, job["id"], 4)
        lease = self.store.claim()
        ledger = OnlineLedger(self.store, lease, 1.0)

        def timeout(_client):
            raise TimeoutError

        with self.assertRaises(RequestPending) as pending:
            ledger.request("Gemini", "model", "timed", {}, 0.6, lambda: None, timeout, lambda *_: {}, lambda _: None)
        old_lease = lease["lease"]
        with self.store.connection() as database:
            database.execute("UPDATE studio.jobs SET lease_until = now() - interval '1 minute' WHERE id = %s", (job["id"],))
        reclaimed = self.store.claim()
        self.assertNotEqual(reclaimed["lease"], old_lease)
        self.assertFalse(self.store.renew(job["id"], old_lease))
        self.store.stop(job["id"], reclaimed["lease"], "paused", "Unknown charge", pending.exception.request_id)
        with self.assertRaises(JobError):
            self.store.resume(owner, job["id"], None)
        self.store.resume(owner, job["id"], pending.exception.request_id)
        ledger = OnlineLedger(self.store, self.store.claim(), 1.0)
        with self.assertRaises(BudgetExceeded):
            ledger.request("Gemini", "model", "timed", {}, 0.6, lambda: None,
                           lambda _: self.fail("Spent beyond the daily reservation ceiling"), lambda *_: {}, lambda _: None)
