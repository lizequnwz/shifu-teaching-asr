"""Run, review, and explicitly publish Simplified Chinese subtitles."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from uuid import uuid4

from workflow_audio import AUDIO_SUFFIXES, MAX_BYTES, MAX_SECONDS, SAMPLE_RATE
from workflow_pipeline import CHOICES, PIPELINE_VERSION, build_review, transcribe, validate_decision


ROOT = Path(__file__).resolve().parents[1]
VERSION = 3
JOBS = ROOT / "output" / "jobs"
_SNAPSHOT_NAME = re.compile(r"[a-f0-9]{32}")


class WorkflowBusy(ValueError):
    """Another operation owns the required local lock."""


class DecisionConflict(ValueError):
    """The review has changed since it was opened."""


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def file_lock(path, message):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkflowBusy(message) from exc
        yield


def job_lock(job):
    return file_lock(job / ".lock", "Another subtitle process is using this job")


def processing_lock():
    return file_lock(JOBS / ".worker.lock", "Another subtitle job is processing")


def processing_busy():
    try:
        with processing_lock():
            return False
    except WorkflowBusy:
        return True


class Operation:
    def __init__(self, job, kind):
        self.path = job / "operation.json"
        self.id = os.environ.get("SUBTITLE_OPERATION_ID") or uuid4().hex
        self.kind = kind
        self.stage = "preflight"

    def update(self, stage, *, completed=None, total=None, detail=None, status="running", retry_request=None):
        self.stage = stage
        progress = {"id": self.id, "kind": self.kind, "status": status, "stage": stage}
        for field, value in (("completed", completed), ("total", total),
                             ("detail", detail), ("retry_request", retry_request)):
            if value is not None:
                progress[field] = value
        write_json(self.path, progress)


def preflight(cloud=False):
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Install {tool} before starting a subtitle job")
    try:
        import webrtcvad  # noqa: F401
        if cloud:
            from google import genai  # noqa: F401
            import openai  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install the speech dependencies from requirements-workflow.txt") from exc
    if cloud:
        from workflow_gemini import _api_key as gemini_key
        from workflow_openai import _api_key as openai_key
        gemini_key()
        openai_key()


def model_versions():
    from workflow_gemini import TRANSCRIBE_MODEL as gemini_model
    from workflow_openai import TRANSCRIBE_MODEL as openai_model
    return {"gemini": gemini_model, "openai": openai_model}


def normalize_vocabulary(vocabulary):
    if not isinstance(vocabulary, (list, tuple)) or any(not isinstance(term, str) for term in vocabulary):
        raise ValueError("The glossary must be a list of terms")
    result = list(dict.fromkeys(term.strip() for term in vocabulary if term.strip()))
    if len(result) > 100 or any(len(term) > 120 or any(char in term for char in "<>\n\r\x00") for term in result):
        raise ValueError("Provide at most 100 glossary terms of up to 120 characters each")
    return result


def make_manifest(source, *, seconds=None, terms=None, vocabulary=None):
    source = Path(source).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in AUDIO_SUFFIXES or not 0 < source.stat().st_size <= MAX_BYTES:
        raise ValueError("Choose a supported, nonempty recording up to 512 MiB")
    if seconds is not None and (
        type(seconds) not in {int, float} or not math.isfinite(seconds) or not 0 < seconds <= MAX_SECONDS
    ):
        raise ValueError("Pilot length must be between 0 and 120 minutes")
    if terms is not None:
        if vocabulary is not None:
            raise ValueError("Provide only one glossary")
        terms = Path(terms)
        if terms.stat().st_size > 100_000:
            raise ValueError("The glossary exceeds 100 KB")
        vocabulary = terms.read_text("utf-8-sig").splitlines()
    return {
        "version": VERSION, "mode": "audio", "source": str(source), "source_sha256": digest(source),
        "limit_seconds": seconds, "vocabulary": normalize_vocabulary(vocabulary if vocabulary is not None else []),
        "pipeline": PIPELINE_VERSION, "models": model_versions(),
    }


def _validate_job(manifest, job):
    if (manifest.get("version") != VERSION or manifest.get("pipeline") != PIPELINE_VERSION
            or manifest.get("models") != model_versions() or manifest.get("mode") != "audio"):
        raise ValueError("Start a new job with this version of Subtitle Studio")
    source = Path(manifest["source"])
    if digest(source) != manifest["source_sha256"]:
        raise ValueError("The source recording changed after the job started")
    path = job / "job.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("This job belongs to a different recording or configuration")
    else:
        write_json(path, manifest)


def _default_job(manifest):
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(manifest["source"]).stem).strip("-._")[:48]
    suffix = "-pilot" if manifest["limit_seconds"] else ""
    return JOBS / f"{label or 'recording'}{suffix}-{identity(manifest)[:12]}"


def _snapshot(job, pointer, folder):
    marker = job / pointer
    if not marker.is_file():
        return None
    name = read_json(marker)["directory"]
    root = job / folder
    if not isinstance(name, str) or not name.startswith(folder + "/") or not _SNAPSHOT_NAME.fullmatch(name[len(folder) + 1:]):
        raise ValueError("Invalid subtitle revision")
    path = job / name
    if root.is_symlink() or path.is_symlink() or not path.is_dir():
        raise ValueError("Invalid subtitle revision directory")
    return path


def current_draft(job):
    return _snapshot(job, "draft.json", ".drafts")


def published_directory(job):
    return _snapshot(job, "publication.json", ".published")


def decision_snapshot(job):
    draft = current_draft(job)
    decisions = read_json(draft / "decisions.json") if draft is not None else {}
    return decisions, identity(decisions)


def decision_revision(job):
    return decision_snapshot(job)[1]


def page_passages(passages, selection, cursor, size):
    """Page by position in the full recording so changing flags cannot move the cursor."""
    matches = [
        (position, passage)
        for position, passage in enumerate(passages[cursor:], cursor + 1)
        if selection == "all" or any(selection in {"flagged", issue["category"]} for issue in passage["issues"])
    ]
    return (
        [passage for _, passage in matches[:size]],
        str(matches[size - 1][0]) if len(matches) > size else None,
    )


def _new_draft(job, data, decisions):
    root = job / ".drafts"
    root.mkdir(exist_ok=True, mode=0o700)
    directory = root / uuid4().hex
    with tempfile.TemporaryDirectory(dir=root, prefix=".draft-") as temporary:
        staging = Path(temporary) / "review"
        build_review(data, decisions, staging)
        staging.replace(directory)
    write_json(job / "draft.json", {"directory": str(directory.relative_to(job)), "decision_revision": identity(decisions)})
    return directory


def save_review_decision(job, passage_id, choice, text, revision, timings=None):
    with job_lock(job):
        draft = current_draft(job)
        decisions = read_json(draft / "decisions.json") if draft is not None else {}
        if revision != identity(decisions):
            raise DecisionConflict("The review changed. Reload the job before saving.")
        data_path = job / "blocks.json"
        if not data_path.is_file() or draft is None:
            raise ValueError("The Chinese draft is not ready")
        data = read_json(data_path)
        changed = validate_decision(data, passage_id, choice, text, timings)
        if decisions.get(passage_id) != changed:
            decisions[passage_id] = changed
            draft = _new_draft(job, data, decisions)
        return draft


def publish(job, revision):
    with job_lock(job):
        if revision != decision_revision(job):
            raise DecisionConflict("The review changed. Reload the job before publishing.")
        draft = current_draft(job)
        if draft is None or read_json(draft / "summary.json")["status"] != "ready_to_publish" or not (draft / "zh.srt").is_file():
            raise ValueError("Resolve the remaining wording and timing issues before publishing")
        path = published_directory(job)
        if path is not None and read_json(job / "publication.json")["decision_revision"] == revision:
            return path
        root = job / ".published"
        root.mkdir(exist_ok=True, mode=0o700)
        destination = root / uuid4().hex
        with tempfile.TemporaryDirectory(dir=root, prefix=".publish-") as temporary:
            staging = Path(temporary) / "release"
            shutil.copytree(draft, staging)
            staging.replace(destination)
        write_json(job / "publication.json", {"directory": str(destination.relative_to(job)), "decision_revision": revision})
        return destination


def _result(job):
    draft = current_draft(job)
    if draft is None:
        raise ValueError("The Chinese draft is not ready")
    result = {**read_json(draft / "summary.json"), "job": str(job), "revision": decision_revision(job),
              "draft_transcript": str(draft / "zh.txt")}
    if (draft / "zh.srt").is_file():
        result["draft_subtitles"] = str(draft / "zh.srt")
    release = published_directory(job)
    if release is not None:
        if read_json(job / "publication.json")["decision_revision"] == result["revision"]:
            result["status"] = "published"
        result.update(published_transcript=str(release / "zh.txt"), published_subtitles=str(release / "zh.srt"))
    usage = job / "cloud-usage.json"
    if usage.is_file():
        result["cloud_usage"] = read_json(usage)
    return result


def _budget(job, override, pilot):
    path = job / "run-settings.json"
    value = override if override is not None else read_json(path)["budget_usd"] if path.exists() else 2.0 if pilot else 10.0
    write_json(path, {"budget_usd": value})
    return value


def _process_job(job, manifest, operation, budget, retry_request):
    from workflow_cloud import CloudLedger
    from workflow_gemini import GeminiClient
    from workflow_openai import OpenAIClient

    if current_draft(job) is not None:
        return _result(job)
    cached = job / "blocks.json"
    if not cached.is_file():
        preflight(cloud=True)
        ledger = CloudLedger(job, budget, retry_request=retry_request)
        try:
            gemini = GeminiClient(job, budget, ledger=ledger)
            openai = OpenAIClient(job, budget, vocabulary=manifest["vocabulary"], ledger=ledger)
            data = transcribe(Path(manifest["source"]), job, manifest, gemini, openai, operation)
            write_json(cached, data)
        finally:
            write_json(job / "cloud-usage.json", ledger.usage())
    else:
        data = read_json(cached)
    _new_draft(job, data, {})
    return _result(job)


def execute(job, manifest, *, kind, budget=None, retry_request=None):
    from workflow_cloud import BudgetExceeded, RateLimited, RequestPending

    try:
        with processing_lock(), job_lock(job):
            operation = Operation(job, kind)
            operation.update("preflight")
            try:
                _validate_job(manifest, job)
                allowance = _budget(job, budget, manifest["limit_seconds"] is not None)
                result = _process_job(job, manifest, operation, allowance, retry_request)
            except (BudgetExceeded, RateLimited, RequestPending) as exc:
                request = exc.request_id if isinstance(exc, RequestPending) else None
                operation.update(operation.stage, status="paused", detail=str(exc), retry_request=request)
                return {"job": str(job), "status": "paused", "detail": str(exc), "retry_request": request}, 1
            except (OSError, ValueError, RuntimeError) as exc:
                operation.update(operation.stage, status="failed", detail=str(exc))
                return {"job": str(job), "status": "failed", "detail": str(exc)}, 1
            except KeyboardInterrupt:
                operation.update(operation.stage, status="interrupted", detail="The job was interrupted. Resume to continue.")
                raise
            operation.update("complete", status="succeeded")
            return result, 0
    except WorkflowBusy as exc:
        return {"job": str(job), "status": "paused", "detail": str(exc)}, 1


def positive(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("expected a finite positive number") from None
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive number")
    return number


def batch(path):
    base = path.resolve(strict=True).parent
    results = []
    for line_number, line in enumerate(path.read_text("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) - {"audio", "terms", "seconds", "budget"} or "audio" not in row:
                raise ValueError("Each row needs an audio path and may include terms, seconds, and budget")
            manifest = make_manifest(base / row["audio"], terms=base / row["terms"] if "terms" in row else None,
                                     seconds=positive(row["seconds"]) if "seconds" in row else None)
            result, _ = execute(_default_job(manifest), manifest, kind="batch",
                                budget=positive(row["budget"]) if "budget" in row else None)
        except (OSError, ValueError, TypeError, argparse.ArgumentTypeError) as exc:
            result = {"status": "failed", "detail": str(exc)}
        results.append({"line": line_number, **result})
    if not results:
        raise ValueError("The batch manifest has no jobs")
    return {"jobs": results}, int(any(row["status"] in {"failed", "paused"} for row in results))


def _add_run_options(command):
    command.add_argument("--budget", type=positive, help="estimated API budget in USD")
    command.add_argument("--retry-request", help="retry this exact uncertain request with a new charge reservation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="transcribe a recording and prepare its Chinese review")
    run.add_argument("recording", type=Path)
    run.add_argument("--job", type=Path)
    run.add_argument("--seconds", type=positive, help="prepare a pilot from the first seconds of audio")
    run.add_argument("--terms", type=Path, help="UTF-8 glossary with one confirmed term per line")
    _add_run_options(run)
    resume = commands.add_parser("resume", help="continue an interrupted job")
    resume.add_argument("job", type=Path)
    _add_run_options(resume)
    batch_parser = commands.add_parser("batch", help="process a UTF-8 JSON Lines manifest")
    batch_parser.add_argument("manifest", type=Path)
    decide = commands.add_parser("decide", help="save one Chinese review decision")
    decide.add_argument("job", type=Path)
    decide.add_argument("passage_id")
    decide.add_argument("choice", choices=CHOICES)
    wording = decide.add_mutually_exclusive_group()
    wording.add_argument("--text", help="complete corrected wording")
    wording.add_argument("--text-stdin", action="store_true", help="read complete corrected wording from stdin")
    decide.add_argument("--timings", type=Path, help="JSON array of timed subtitles for the accepted passage")
    publish_parser = commands.add_parser("publish", help="publish the reviewed Chinese transcript and subtitles")
    publish_parser.add_argument("job", type=Path)
    probe = commands.add_parser("preflight", help="check prerequisites without uploading audio")
    probe.add_argument("--cloud", action="store_true", help="also check SDKs and credential setup")
    args = parser.parse_args(argv)
    status = 0
    if args.command == "preflight":
        preflight(args.cloud)
        result = {"ready": True}
    elif args.command == "batch":
        result, status = batch(args.manifest)
    elif args.command == "run":
        manifest = make_manifest(args.recording, seconds=args.seconds, terms=args.terms)
        job = (args.job or _default_job(manifest)).resolve()
        result, status = execute(job, manifest, kind="run", budget=args.budget, retry_request=args.retry_request)
    else:
        job = args.job.resolve(strict=True)
        if args.command == "resume":
            manifest = read_json(job / "job.json")
            result, status = execute(job, manifest, kind="resume", budget=args.budget, retry_request=args.retry_request)
        elif args.command == "decide":
            text = sys.stdin.read(24_001) if args.text_stdin else args.text
            timing = read_json(args.timings) if args.timings else None
            save_review_decision(job, args.passage_id, args.choice, text, decision_revision(job), timing)
            result = _result(job)
        else:
            publish(job, decision_revision(job))
            result = _result(job)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Subtitle workflow: {exc}", file=sys.stderr)
        sys.exit(1)
