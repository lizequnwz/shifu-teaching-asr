import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import subtitle_workflow as engine
from workflow_audio import Recording, SAMPLE_RATE, Window
from workflow_pipeline import _core_selection, _windows_passages, build_review


class Gemini:
    calls = 0

    def __init__(self, *args, **kwargs):
        pass

    def transcribe_timed(self, clip):
        type(self).calls += 1
        with wave.open(str(clip), "rb") as audio:
            assert audio.getframerate() == SAMPLE_RATE
        return {
            "text": "我都沒看懂。",
            "words": [
                {"text": "我都", "text_start": 0, "text_end": 2, "start_seconds": .1, "end_seconds": .5},
                {"text": "沒", "text_start": 2, "text_end": 3, "start_seconds": .5, "end_seconds": .7},
                {"text": "看懂", "text_start": 3, "text_end": 5, "start_seconds": .7, "end_seconds": 1.2},
            ],
        }


class OpenAI:
    calls = 0
    wording = "我都看懂。"

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, clip, *, purpose):
        type(self).calls += 1
        return self.wording


def write_wav(path, seconds):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * (seconds * SAMPLE_RATE))


class WorkflowTests(TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.recording = self.root / "recording.wav"
        write_wav(self.recording, 2)
        self.jobs = self.root / "jobs"
        self.job = self.jobs / "sample"
        Gemini.calls = OpenAI.calls = 0
        self.enterContext(patch.object(engine, "JOBS", self.jobs))
        self.enterContext(patch.object(engine, "preflight"))
        self.enterContext(patch.object(Recording, "speech", return_value=[(1600, 19_200)]))
        self.enterContext(patch("workflow_gemini.GeminiClient", Gemini))
        self.enterContext(patch("workflow_openai.OpenAIClient", OpenAI))

    def test_chinese_draft_and_explicit_publish_are_distinct_revisions(self):
        manifest = engine.make_manifest(self.recording)
        result, status = engine.execute(self.job, manifest, kind="run")
        self.assertEqual((status, result["status"]), (0, "needs_review"), result)
        self.assertNotIn("published_subtitles", result)
        initial = engine.current_draft(self.job)
        review = engine.read_json(initial / "review.json")
        passage = review[0]
        self.assertEqual(passage["timings"][0]["text"], "我都没看懂。")
        issue = passage["issues"][0]
        self.assertEqual(issue["reason"], "negation")
        self.assertEqual((issue["start_sample"], issue["end_sample"]), (8_000, 11_200))
        self.assertEqual(len(passage["clips"]), 1)
        with self.assertRaises(ValueError):
            engine.publish(self.job, result["revision"])

        draft = engine.save_review_decision(self.job, passage["id"], "suggestion", None, result["revision"])
        revision = engine.decision_revision(self.job)
        self.assertEqual(engine.read_json(draft / "summary.json")["status"], "ready_to_publish")
        with self.assertRaises(engine.DecisionConflict):
            engine.save_review_decision(self.job, passage["id"], "no_speech", None, result["revision"])
        release = engine.publish(self.job, revision)
        self.assertEqual(engine.publish(self.job, revision), release)
        self.assertEqual((release / "zh.txt").read_text(), "我都没看懂。\n")
        self.assertIn("00:00:00,100 --> 00:00:01,200", (release / "zh.srt").read_text())
        refreshed, _ = engine.execute(self.job, manifest, kind="resume")
        self.assertEqual(refreshed["status"], "published")
        self.assertEqual((Gemini.calls, OpenAI.calls), (1, 1))

    def test_lexical_replacement_requires_manual_times_and_preserves_the_prior_release(self):
        manifest = engine.make_manifest(self.recording)
        result, _ = engine.execute(self.job, manifest, kind="run")
        passage = engine.read_json(engine.current_draft(self.job) / "passages.json")[0]
        engine.save_review_decision(self.job, passage["id"], "gemini", None, result["revision"])
        initial = engine.publish(self.job, engine.decision_revision(self.job))
        draft = engine.save_review_decision(
            self.job, passage["id"], "edited", "我没理解。", engine.decision_revision(self.job),
        )
        review = engine.read_json(draft / "review.json")
        self.assertEqual(review[0]["issues"][0]["category"], "timing")
        self.assertFalse((draft / "zh.srt").exists())
        self.assertEqual(engine.published_directory(self.job), initial)

        cue = {"text": "我没理解。", "start_ms": 200, "end_ms": 1400}
        draft = engine.save_review_decision(
            self.job, passage["id"], "edited", "我没理解。", engine.decision_revision(self.job), [cue],
        )
        self.assertEqual(engine.read_json(draft / "passages.json")[0]["timings"], [cue])
        final = engine.publish(self.job, engine.decision_revision(self.job))
        self.assertNotEqual(final, initial)
        self.assertEqual((initial / "zh.txt").read_text(), "我都没看懂。\n")
        self.assertEqual((final / "zh.txt").read_text(), "我没理解。\n")

    def test_review_cursor_stays_in_recording_order_after_a_flag_is_cleared(self):
        passages = [{"id": index, "issues": [{"category": "wording"}]} for index in range(3)]
        page, cursor = engine.page_passages(passages, "flagged", 0, 2)
        self.assertEqual([item["id"] for item in page], [0, 1])
        passages[1]["issues"] = []
        page, cursor = engine.page_passages(passages, "flagged", int(cursor), 2)
        self.assertEqual([item["id"] for item in page], [2])
        self.assertIsNone(cursor)

    def test_checkpoint_is_reused_after_an_interrupted_review_build(self):
        manifest = engine.make_manifest(self.recording)
        with patch.object(engine, "build_review", side_effect=OSError("interrupted")):
            result, status = engine.execute(self.job, manifest, kind="run")
        self.assertEqual((status, result["status"]), (1, "failed"))
        (self.job / "blocks.json").unlink()
        result, status = engine.execute(self.job, manifest, kind="resume")
        self.assertEqual((status, result["status"]), (0, "needs_review"), result)
        self.assertTrue((self.job / "windows" / "0001.json").is_file())
        self.assertEqual((Gemini.calls, OpenAI.calls), (1, 1))

    def test_unanchored_text_is_not_dropped_and_speech_without_words_is_flagged(self):
        end = 3 * SAMPLE_RATE
        data, _ = _windows_passages({"text": "尊敬的师父", "words": []}, "尊敬的师父",
                                   Window(0, end, 0, end), [(0, end)], ["师父"], [])
        self.assertEqual(data[0]["text"], "尊敬的师父")
        data[0]["clips"] = []
        summary = build_review({"samples": end, "passages": data}, {}, self.root / "unanchored")
        self.assertEqual(summary["status"], "needs_review")
        issues = json.loads((self.root / "unanchored" / "review.json").read_text())[0]["issues"]
        self.assertEqual({issue["category"] for issue in issues}, {"coverage", "timing"})

    def test_candidate_insertion_replay_requires_words_on_both_sides(self):
        start = 2 * SAMPLE_RATE
        end = 5 * SAMPLE_RATE
        draft = {"text": "師父慈悲。", "words": [
            {"text": "師父", "text_start": 0, "text_end": 2, "start_seconds": .25, "end_seconds": .5},
            {"text": "慈悲", "text_start": 2, "text_end": 4, "start_seconds": .75, "end_seconds": 1},
        ]}

        passages, _ = _windows_passages(draft, "師父今天說法慈悲。", Window(start, end, start, end), [], [], [])
        issue = passages[0]["issues"][0]
        self.assertEqual((issue["category"], issue["reason"]), ("coverage", "candidate_addition"))
        self.assertEqual((issue["start_sample"], issue["end_sample"]), (36_000, 48_000))

        passages, _ = _windows_passages(draft, "今晚一定師父慈悲。", Window(start, end, start, end), [], [], [])
        issue = passages[0]["issues"][0]
        self.assertEqual(issue["reason"], "candidate_addition")
        self.assertNotIn("start_sample", issue)

    def test_repeated_disagreement_keeps_the_first_timed_instance(self):
        end = 3 * SAMPLE_RATE
        draft = {"text": "我沒看懂，也沒聽懂。", "words": [
            {"text": "看懂", "text_start": 2, "text_end": 4, "start_seconds": .25, "end_seconds": .5},
            {"text": "沒", "text_start": 6, "text_end": 7, "start_seconds": .75, "end_seconds": 1},
        ]}

        passages, _ = _windows_passages(draft, "我看懂，也聽懂。", Window(0, end, 0, end), [], [], [])
        issue, = passages[0]["issues"]
        self.assertEqual(issue["reason"], "negation")
        self.assertEqual((issue["start_sample"], issue["end_sample"]), (12_000, 16_000))

    def test_deduplication_requires_the_same_timestamp_as_well_as_the_same_word(self):
        previous = {"text": "师父", "start_sample": 17920, "end_sample": 18010}
        overlapping = {"text": "师父", "start_sample": 17940, "end_sample": 18080}
        repetition = {"text": "师父", "start_sample": 18100, "end_sample": 18400}
        self.assertEqual(
            _core_selection([overlapping, repetition], Window(18000, 20000, 17000, 20000), [previous]), (1, 2),
        )
