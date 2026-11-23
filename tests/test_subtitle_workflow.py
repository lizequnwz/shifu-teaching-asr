import json
from pathlib import Path
from threading import Barrier
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import wave


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import subtitle_workflow as engine
from workflow_audio import Recording, SAMPLE_RATE, Window
from workflow_pipeline import (
    _comparison_windows, _reconcile_seam, _reconciled_windows, _windows_passages, build_review, validate_decision,
)


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


def timed_draft(*entries):
    text, words = "", []
    for wording, start, end in entries:
        words.append({"text": wording, "text_start": len(text), "text_end": len(text) + len(wording),
                      "start_seconds": start, "end_seconds": end})
        text += wording
    return {"text": text, "words": words}


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

    def test_checkpoint_keeps_the_finished_provider_when_its_peer_fails(self):
        manifest = engine.make_manifest(self.recording)
        rendezvous = Barrier(2, timeout=4)
        original = Gemini.transcribe_timed

        def draft(client, clip):
            rendezvous.wait()
            return original(client, clip)

        def fail(_client, _clip, *, purpose):
            rendezvous.wait()
            raise RuntimeError("comparison failed")

        with patch.object(Gemini, "transcribe_timed", draft), patch.object(OpenAI, "transcribe", fail):
            result, status = engine.execute(self.job, manifest, kind="run")
        self.assertEqual((status, result["detail"]), (1, "comparison failed"))
        checkpoint = self.job / "windows" / "0001.json"
        self.assertEqual(set(engine.read_json(checkpoint)), {"identity", "gemini"})
        result, status = engine.execute(self.job, manifest, kind="resume")
        self.assertEqual((status, result["status"]), (0, "needs_review"), result)
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

    def test_repeated_disagreement_keeps_each_instance(self):
        end = 3 * SAMPLE_RATE
        draft = {"text": "我沒看懂，也沒聽懂。", "words": [
            {"text": "看懂", "text_start": 2, "text_end": 4, "start_seconds": .25, "end_seconds": .5},
            {"text": "沒", "text_start": 6, "text_end": 7, "start_seconds": .75, "end_seconds": 1},
        ]}

        passages, _ = _windows_passages(draft, "我看懂，也聽懂。", Window(0, end, 0, end), [], [], [])
        untimed, timed = passages[0]["issues"]
        self.assertEqual(untimed["reason"], "negation")
        self.assertNotIn("start_sample", untimed)
        self.assertEqual(timed["reason"], "negation")
        self.assertEqual((timed["start_sample"], timed["end_sample"]), (12_000, 16_000))

    def test_deduplication_requires_the_same_timestamp_as_well_as_the_same_word(self):
        previous = {"text": "师父", "start_sample": 17920, "end_sample": 18010}
        overlapping = {"text": "师父", "start_sample": 17940, "end_sample": 18080}
        repetition = {"text": "师父", "start_sample": 18100, "end_sample": 18400}
        self.assertEqual(
            _reconcile_seam([previous], [overlapping, repetition], 18000, 1, 0, []), (18010, 1, 1, False),
        )

    def test_preserves_a_word_when_overlapping_timestamps_disagree_about_ownership(self):
        r = SAMPLE_RATE
        windows = [Window(0, 10 * r, 0, 15 * r), Window(10 * r, 20 * r, 5 * r, 20 * r)]
        left = timed_draft(("我们修行讲", 8.0, 9.1), ("正法", 10.02, 10.25))
        right = timed_draft(("正法", 4.61, 4.91), ("特别重要", 5.45, 6.55))
        saved = [{"gemini": draft, "openai": draft["text"]} for draft in (left, right)]
        speech = [(int(9.61 * r), int(10.25 * r))]
        blocks = []
        previous = []
        for draft, (window, words, selection, issue) in zip(saved, _reconciled_windows(windows, saved, speech)):
            passages, previous = _windows_passages(
                draft["gemini"], draft["openai"], window, speech, [], previous,
                word_selection=(words, selection), seam_issue=issue,
            )
            blocks.extend({**passage, "clips": []} for passage in passages)
        summary = build_review({"samples": 20 * r, "passages": blocks}, {}, self.root / "joined")
        self.assertEqual(summary["status"], "ready_to_publish")
        self.assertEqual((self.root / "joined" / "zh.txt").read_text().replace("\n", ""), "我们修行讲正法特别重要")

    def test_disputed_seam_has_a_replay_range_across_the_cut(self):
        r = SAMPLE_RATE
        windows = [Window(0, 10 * r, 0, 15 * r), Window(10 * r, 20 * r, 5 * r, 20 * r)]
        left = timed_draft(("正法", 10.0, 10.3))
        right = timed_draft(("证法", 4.7, 5.0))
        _, (_, _, _, issue) = _reconciled_windows(windows, [{"gemini": left}, {"gemini": right}], [])
        self.assertEqual(issue["reason"], "ambiguous_window_overlap")
        self.assertLess(issue["start_sample"], 10 * r)
        self.assertGreater(issue["end_sample"], 10 * r)

    def test_targeted_comparison_covers_flags_and_samples_a_routine_window(self):
        r = SAMPLE_RATE
        windows = [Window(i * r, (i + 10) * r, i * r, (i + 10) * r) for i in range(0, 50, 10)]
        drafts = [timed_draft((text, 2.0, 3.0)) for text in ("大家好。", "念佛。", "往后看。", "先坐下。", "谢谢。")]
        saved = [{"gemini": draft} for draft in drafts]
        speech = [(26 * r, 28 * r)]
        aligned = list(_reconciled_windows(windows, saved, speech))
        first = _comparison_windows(aligned, saved, speech, ["念佛"], "same-source")
        self.assertTrue({1, 2} <= first)
        self.assertEqual(len(first), 3)
        self.assertEqual(first, _comparison_windows(aligned, saved, speech, ["念佛"], "same-source"))

    def test_targeted_run_calls_openai_for_the_audit_and_labels_the_remaining_passages(self):
        write_wav(self.recording, 120)
        manifest = engine.make_manifest(self.recording, core_seconds=30, comparison="targeted")
        with (patch.object(Recording, "speech", return_value=[]),
              patch.object(Gemini, "transcribe_timed", return_value=timed_draft(("我都看懂。", 7.0, 8.0)))):
            result, status = engine.execute(self.job, manifest, kind="run")
        self.assertEqual((status, result["status"]), (0, "ready_to_publish"), result)
        self.assertEqual(result["comparison"]["mode"], "targeted")
        self.assertEqual(result["comparison"]["windows"], 4)
        self.assertEqual(OpenAI.calls, 1)
        passages = engine.read_json(engine.current_draft(self.job) / "passages.json")
        self.assertEqual(sum(passage["comparison_skipped"] for passage in passages), 3)

    def test_glossary_promotes_a_single_character_difference(self):
        r = SAMPLE_RATE
        draft = timed_draft(("大家一起", .1, .8), ("念佛。", 1, 2))
        passages, _ = _windows_passages(draft, "大家一起练佛。", Window(0, 3 * r, 0, 3 * r), [], ["念佛"], [])
        issue, = passages[0]["issues"]
        self.assertEqual(issue["reason"], "glossary_difference")
        self.assertEqual((issue["start_sample"], issue["end_sample"]), (r, 2 * r))

    def test_accepting_wording_leaves_each_coverage_issue_open(self):
        r = SAMPLE_RATE
        draft = timed_draft(("我们没有念佛。", 1.0, 2.0))
        passages, _ = _windows_passages(draft, "我们念佛。", Window(0, 9 * r, 0, 9 * r),
                                       [(4 * r, 6 * r), (7 * r, 9 * r)], [], [])
        passage = {**passages[0], "clips": []}
        data = {"samples": 9 * r, "passages": [passage]}
        target = self.root / "coverage"
        decision = validate_decision(data, passage["id"], "gemini", None)
        build_review(data, {passage["id"]: decision}, target)
        issues = engine.read_json(target / "review.json")[0]["issues"]
        self.assertEqual([issue["reason"] for issue in issues], ["speech_without_words"] * 2)
        confirmed = validate_decision(data, passage["id"], None, None, previous=decision,
                                      resolved_issues=[issues[0]["id"]])
        state = build_review(data, {passage["id"]: confirmed}, target)
        self.assertEqual(state["status"], "needs_review")
        self.assertEqual(engine.read_json(target / "review.json")[0]["issues"], [issues[1]])
        with self.assertRaises(ValueError):
            validate_decision(data, passage["id"], None, None, previous=confirmed, resolved_issues=["issue-0"])
        resolved = validate_decision(data, passage["id"], None, None, previous=confirmed,
                                     resolved_issues=[issues[1]["id"]])
        self.assertEqual(build_review(data, {passage["id"]: resolved}, target)["status"], "ready_to_publish")

    def test_one_character_edit_keeps_surrounding_cue_times(self):
        r = SAMPLE_RATE
        draft = timed_draft(("现在开始。", 0.0, 1.0), ("一起练佛。", 1.3, 2.3), ("最后回向。", 2.6, 3.6))
        passages, _ = _windows_passages(draft, draft["text"], Window(0, 4 * r, 0, 4 * r), [], [], [])
        passage = {**passages[0], "clips": []}
        data = {"samples": 4 * r, "passages": [passage]}
        target = self.root / "corrected"
        build_review(data, {}, target)
        before = engine.read_json(target / "passages.json")[0]["timings"]
        wording = draft["text"].replace("练", "念")
        decision = validate_decision(data, passage["id"], "edited", wording)
        self.assertEqual(build_review(data, {passage["id"]: decision}, target)["status"], "needs_review")
        template = engine.read_json(target / "passages.json")[0]["timing_template"]
        self.assertEqual(template, [
            {**before[0], "needs_confirmation": False},
            {**before[1], "text": "一起念佛。", "needs_confirmation": True},
            {**before[2], "needs_confirmation": False},
        ])
        timings = [{key: item[key] for key in ("text", "start_ms", "end_ms")} for item in template]
        decision = validate_decision(data, passage["id"], "edited", wording, timings, previous=decision)
        self.assertEqual(build_review(data, {passage["id"]: decision}, target)["status"], "ready_to_publish")
