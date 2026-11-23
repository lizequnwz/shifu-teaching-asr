"""Single-entry persistent job worker for the Online deployment."""
from __future__ import annotations

from pathlib import Path
import json
import logging
import re
import signal
from tempfile import TemporaryDirectory
from threading import Event, Thread

import psycopg

from online_storage import ObjectMissing, OnlineSettings, StorageError, Supabase
from online_store import LEASE_SECONDS, LeaseLost, OnlineLedger, OnlineStore, object_prefix
from subtitle_workflow import make_manifest, preflight
from workflow_cloud import BudgetExceeded, CloudError, RateLimited, RequestPending
from workflow_gemini import GeminiClient
from workflow_openai import OpenAIClient
from workflow_pipeline import build_review, transcribe


LOG = logging.getLogger("studio.worker")
WINDOW_NAME = re.compile(r"[0-9]{4}\.json\Z")


class Operation:
    def __init__(self, store, job):
        self.store = store
        self.job_id = job["id"]
        self.lease = job["lease"]

    def update(self, stage, completed=None, total=None):
        self.store.progress(self.job_id, self.lease, stage, completed, total)


def _heartbeat(store, job_id, lease, done):
    while not done.wait(LEASE_SECONDS / 4):
        try:
            if not store.renew(job_id, lease):
                return
        except Exception as exc:
            # A transient database error may be recovered on the next heartbeat.
            LOG.warning("Lease heartbeat failed for %s (%s)", job_id, type(exc).__name__)


def process(store, supabase, job, daily_budget):
    job_id, lease = job["id"], job["lease"]
    done = Event()
    keeper = Thread(target=_heartbeat, args=(store, job_id, lease, done), name="studio-job-lease", daemon=True)
    keeper.start()
    try:
        try:
            _process(store, supabase, job, daily_budget)
        except RateLimited as exc:
            store.backoff(job_id, lease, exc.retry_after)
        except RequestPending as exc:
            store.stop(job_id, lease, "paused", str(exc), exc.request_id)
        except BudgetExceeded as exc:
            store.stop(job_id, lease, "paused", str(exc))
        except ObjectMissing as exc:
            store.stop(job_id, lease, "failed", str(exc))
        except StorageError as exc:
            store.stop(job_id, lease, "paused", str(exc))
        except (ValueError, CloudError) as exc:
            store.stop(job_id, lease, "failed", str(exc))
        except (LeaseLost, psycopg.Error):
            raise
        except Exception as exc:
            LOG.error("Could not finish job %s (%s)", job_id, type(exc).__name__)
            store.stop(job_id, lease, "failed", "Studio could not finish processing this recording.")
    except LeaseLost:
        LOG.info("Stopped stale work for %s", job_id)
    finally:
        done.set()
        keeper.join(timeout=6)


def _process(store, supabase, job, daily_budget):
    job_id, lease = job["id"], job["lease"]
    prefix = object_prefix(job_id)
    operation = Operation(store, job)
    with TemporaryDirectory(prefix="studio-job-") as temp:
        directory = Path(temp)
        suffix = Path(job["name"]).suffix.lower()
        source = directory / f"source{suffix}"
        supabase.download(f"{prefix}/source", source, job["source_bytes"])
        manifest = make_manifest(source, vocabulary=job["terms"])
        manifest.pop("source")
        store.pin_manifest(job_id, lease, manifest)
        windows = directory / "windows"
        windows.mkdir()
        for window in store.windows(job_id, lease):
            if not WINDOW_NAME.fullmatch(window["name"]):
                raise ValueError("A stored checkpoint has an invalid name.")
            (windows / window["name"]).write_text(json.dumps(window["result"], ensure_ascii=False), "utf-8")

        def authorize(duration):
            store.authorize_duration(job_id, lease, duration)

        def checkpoint(path):
            store.save_window(job_id, lease, path.name, json.loads(path.read_text("utf-8")))

        ledger = OnlineLedger(store, job, daily_budget)
        gemini = GeminiClient(directory, daily_budget, ledger=ledger)
        openai = OpenAIClient(directory, daily_budget, vocabulary=manifest["vocabulary"], ledger=ledger)
        data = transcribe(source, directory, manifest, gemini, openai, operation,
                          authorize_duration=authorize, checkpoint=checkpoint)
        draft = directory / "draft"
        summary = build_review(data, {}, draft)
        passages = json.loads((draft / "passages.json").read_text("utf-8"))

        clip_prefix = f"{prefix}/clips/{lease}"
        clips = sorted((directory / "clips").glob("*.wav"))
        for index, filename in enumerate(clips, 1):
            operation.update("preparing_review", completed=index - 1, total=len(clips))
            supabase.put(f"{clip_prefix}/{filename.name}", filename, "audio/wav")
        store.finish(job_id, lease, data, summary, passages, clip_prefix)


def main():
    settings = OnlineSettings.from_environment()
    preflight(cloud=True)
    store = OnlineStore(settings.database_url)
    supabase = Supabase(settings)
    stopping = Event()
    signal.signal(signal.SIGTERM, lambda _signal, _frame: stopping.set())
    signal.signal(signal.SIGINT, lambda _signal, _frame: stopping.set())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        while not stopping.is_set():
            try:
                cleanup = store.expire_and_next_cleanup()
                if cleanup is not None:
                    try:
                        supabase.remove_tree(object_prefix(cleanup))
                        store.cleanup_done(cleanup)
                    except StorageError:
                        LOG.error("Object deletion pending for %s", cleanup)
                job = store.claim()
                if job is None:
                    stopping.wait(2)
                    continue
                LOG.info("Processing job %s", job["id"])
                process(store, supabase, job, settings.daily_budget)
            except Exception as exc:
                LOG.error("Worker iteration failed (%s)", type(exc).__name__)
                stopping.wait(5)
    finally:
        supabase.client.close()


if __name__ == "__main__":
    main()
