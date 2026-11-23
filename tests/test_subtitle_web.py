from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import subtitle_web as web
import subtitle_workflow as engine
from test_subtitle_workflow import Gemini, OpenAI, write_wav


class StudioTests(TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.jobs = root / "jobs"
        self.recording = root / "source.wav"
        write_wav(self.recording, 2)
        self.enterContext(patch.object(web, "JOBS", self.jobs))
        self.enterContext(patch.object(engine, "JOBS", self.jobs))
        self.enterContext(patch.object(engine, "preflight"))
        self.enterContext(patch("workflow_gemini.GeminiClient", Gemini))
        self.enterContext(patch("workflow_openai.OpenAIClient", OpenAI))
        self.client = self.enterContext(TestClient(web.app, base_url="http://127.0.0.1:8765"))
        self.token = {"X-Subtitle-Token": self.client.get("/api/config").json()["token"]}

    def _upload(self, headers=None):
        return self.client.post(
            "/api/jobs", headers=headers,
            files={"file": ("開示.wav", self.recording.read_bytes(), "audio/wav")},
            data={"terms": "师父\n"},
        )

    def _wait_for_review(self, job_id):
        deadline = monotonic() + 5
        while monotonic() < deadline:
            detail = self.client.get(f"/api/jobs/{job_id}").json()
            worker = self.client.app.state.worker
            with worker.guard:
                pending = self.jobs / job_id in worker.scheduled
            if detail["status"] == "needs_review" and not pending:
                return detail
            self.assertNotEqual(detail["status"], "failed", detail)
            sleep(.015)
        self.fail("The job did not reach review")

    def test_upload_review_publish_play_and_delete(self):
        response = self._upload(self.token)
        self.assertEqual(response.status_code, 201, response.text)
        job_id = response.json()["id"]
        endpoint = f"/api/jobs/{job_id}"
        original = self._wait_for_review(job_id)
        self.assertEqual(original["name"], "開示.wav")
        self.assertIsNone(original.get("published_revision"))
        listing = self.client.get("/api/jobs").json()
        self.assertEqual([job["id"] for job in listing["jobs"]], [job_id])

        flagged = self.client.get(f"{endpoint}/passages", params={"filter": "wording"}).json()
        self.assertEqual(flagged["revision"], original["revision"])
        passage, = flagged["passages"]
        self.assertEqual(passage["candidates"]["openai"], "我都看懂。")
        self.assertEqual(passage["timings"][0]["text"], "我都没看懂。")
        self.assertEqual(passage["issues"][0]["reason"], "negation")
        self.assertEqual(self.client.get(f"{endpoint}/passages", params={"filter": "timing"}).json()["passages"], [])

        clip_id = passage["clips"][0]["id"]
        clip_url = self.client.get(f"{endpoint}/clips/{clip_id}").json()["url"]
        clip = self.client.get(clip_url, headers={"Range": "bytes=0-43"})
        self.assertEqual(clip.status_code, 206)
        self.assertEqual(len(clip.content), 44)
        self.assertTrue(clip.content.startswith(b"RIFF"))
        self.assertTrue(clip.headers["content-range"].startswith("bytes 0-43/"))
        (self.jobs / job_id / "clips" / "audio-1-2.wav").write_bytes(b"unlisted")
        self.assertEqual(self.client.get(f"{endpoint}/clips/audio-1-2.wav").status_code, 404)

        publish = f"{endpoint}/publish"
        unavailable = self.client.post(publish, headers=self.token, json={"revision": original["revision"]})
        self.assertEqual(unavailable.status_code, 400)
        decision = {"choice": "suggestion", "revision": original["revision"]}
        edit = f"{endpoint}/decisions/{passage['id']}"
        saved = self.client.put(edit, headers=self.token, json=decision)
        self.assertEqual(saved.status_code, 200, saved.text)
        saved_review = saved.json()
        self.assertEqual(saved_review["job"]["status"], "ready_to_publish")
        self.assertEqual(saved_review["passage"]["id"], passage["id"])
        self.assertTrue(saved_review["passage"]["human_reviewed"])
        self.assertEqual(self.client.put(edit, headers=self.token, json=decision).status_code, 409)
        self.assertEqual(self.client.post(publish, headers=self.token, json={"revision": original["revision"]}).status_code, 409)

        ready = self.client.post(publish, headers=self.token, json={"revision": saved_review["job"]["revision"]})
        self.assertEqual(ready.status_code, 200, ready.text)
        self.assertEqual(ready.json()["status"], "published")
        release = ready.json()["published_revision"]
        text_export = self.client.get(f"{endpoint}/exports/{release}/txt").json()["url"]
        subtitles_export = self.client.get(f"{endpoint}/exports/{release}/srt").json()["url"]
        self.assertEqual(self.client.get(text_export).text, "我都没看懂。\n")
        self.assertIn("00:00:00,100 --> 00:00:01,200\n我都没看懂。", self.client.get(subtitles_export).text)

        self.assertEqual(self.client.delete(endpoint, headers=self.token).status_code, 204)
        for url in (endpoint, clip_url, text_export, subtitles_export):
            self.assertEqual(self.client.get(url).status_code, 404)
        self.assertFalse((self.jobs / job_id).exists())

    def test_rejects_cross_origin_and_unsanctioned_mutations(self):
        self.assertEqual(self.client.get("/api/config", headers={"Host": "testserver"}).status_code, 403)
        self.assertEqual(self._upload().status_code, 403)
        headers = {**self.token, "Origin": "https://unrelated.example"}
        self.assertEqual(self._upload(headers).status_code, 403)
        self.assertEqual(self.client.get("/api/jobs").json()["jobs"], [])
        self.assertEqual(self.client.get("/api/jobs", params={"cursor": "next"}).status_code, 400)
