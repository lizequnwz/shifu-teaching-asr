from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_subtitles import evaluate, main, normalize_text
from subtitle_export import _timestamp


class SubtitleEvaluationTests(TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)

    def fixture(self, name, cues, txt=None):
        transcript, subtitles = self.directory / (name + ".txt"), self.directory / (name + ".srt")
        transcript.write_text(txt if txt is not None else "\n".join(cue[3] for cue in cues), "utf-8")
        subtitles.write_text("".join(
            f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}\n\n"
            for index, start, end, text in cues
        ), "utf-8")
        return transcript, subtitles

    def test_normalization_keeps_number_meaning_and_repetitions(self):
        self.assertEqual(normalize_text("師父，師父！ 3.5% / －0．２"), "师父师父3.5%-0.2")
        self.assertNotEqual(normalize_text("3.5"), normalize_text("35"))
        self.assertNotEqual(normalize_text("師父，師父。"), normalize_text("师父。"))

    def test_unique_text_matches_different_cue_ids(self):
        reference = self.fixture("gold", [(11, 200, 1600, "禮敬師父。"), (12, 1700, 3600, "三皈依。")])
        candidate = self.fixture("actual", [(1, 300, 1500, "礼敬师父！"), (2, 1850, 3650, "三皈依")])
        report = evaluate(*reference, *candidate)
        self.assertEqual(report["character_error_rate"], 0)
        self.assertEqual(report["timing"]["matched_cues"], 2)
        self.assertEqual(report["timing"]["start_error_p95_ms"], 150)
        self.assertEqual(report["timing"]["end_error_p95_ms"], 100)
        self.assertTrue(report["passes_checks"])

    def test_reads_bom_crlf_and_multiline_cues(self):
        reference = self.fixture("gold", [(11, 3_600_200, 3_601_600, "禮敬師父。")])
        candidate = self.fixture("actual", [(1, 3_600_200, 3_601_600, "礼敬师父。")])
        candidate[1].write_bytes("\ufeff0001\r\n01:00:00,200  -->  01:00:01,600\r\n礼敬\r\n师父。\r\n".encode("utf-8"))
        report = evaluate(*reference, *candidate)
        self.assertTrue(report["passes_checks"])
        self.assertEqual(report["timing"]["start_error_p95_ms"], 0)

    def test_repeated_cues_are_not_timing_matches(self):
        cues = [(1, 0, 900, "阿弥陀佛。"), (2, 1000, 1900, "阿弥陀佛。"), (3, 2000, 3000, "下面开始。")]
        report = evaluate(*self.fixture("gold", cues), *self.fixture("actual", cues))
        timing = report["timing"]
        self.assertEqual(timing["matched_cues"], 1)
        self.assertEqual([cue["id"] for cue in timing["unmatched_reference_cues"]], [1, 2])
        self.assertEqual(timing["start_error_p95_ms"], 0)
        self.assertFalse(report["checks"]["cue_coverage"])
        self.assertFalse(report["passes_checks"])

    def test_reports_character_edits_and_term_locations(self):
        reference = self.fixture("gold", [(1, 200, 1600, "師父講三皈依。")])
        candidate = self.fixture("actual", [(1, 200, 1600, "师父讲三皈。")])
        report = evaluate(*reference, *candidate, terms=["三皈依"])
        self.assertEqual(report["character_errors"]["deletions"], 1)
        self.assertEqual(report["character_error_rate"], 1 / 6)
        self.assertEqual(report["term_count_errors"], [{
            "term": "三皈依", "reference_count": 1, "candidate_count": 0,
            "reference_cues": [{"id": 1, "start_ms": 200, "end_ms": 1600}],
            "candidate_cues": [],
        }])
        self.assertIsNone(report["timing"]["end_error_p95_ms"])

    def test_check_exit_for_mismatched_transcript_export(self):
        reference = self.fixture("gold", [(1, 200, 1600, "师父开示。")])
        candidate = self.fixture("actual", [(1, 200, 1600, "师父开示。")], txt="师父开示。漏了后面。")
        args = []
        for name, path in zip(("reference-txt", "reference-srt", "candidate-txt", "candidate-srt"),
                              (*reference, *candidate), strict=True):
            args.extend([f"--{name}", str(path)])
        with redirect_stdout(io.StringIO()) as result:
            self.assertEqual(main([*args, "--check"]), 1)
        report = json.loads(result.getvalue())
        self.assertFalse(report["checks"]["candidate_txt_matches_srt"])
