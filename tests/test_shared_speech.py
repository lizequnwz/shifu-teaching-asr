from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import subtitle_export
import workflow_gemini
import workflow_openai


class RecordingLedger:
    def __init__(self, response):
        self.response = response
        self.send = None
        self.details = None

    def request(self, provider, model, purpose, details, reservation, prepare, send, parse, cost):
        self.details = details
        self.send = send
        return parse(self.response, self.response)


def _clip(path):
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 32000)
    return path


def _interaction(text, *annotations):
    return {
        "status": "completed",
        "steps": [{
            "type": "model_output",
            "content": [{"type": "text", "text": text, "annotations": list(annotations)}],
        }],
    }


class ProviderTests(TestCase):
    def test_gemini_request_and_word_annotations_use_recording_offsets(self):
        response = _interaction(
            "好，师父你好。",
            {"type": "word_info", "text": "好", "start_index": 0, "end_index": 3,
             "start_offset": "0s", "end_offset": "0.190s"},
            {"type": "word_info", "text": "师父", "start_index": 6, "end_index": 12,
             "start_offset": "0.240s", "end_offset": "0.650s"},
            {"type": "word_info", "text": "你", "start_offset": "0.650s", "end_offset": "0.850s"},
            {"type": "word_info", "text": "好", "start_offset": "0.850s", "end_offset": "1.150s"},
        )
        ledger = RecordingLedger(response)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            draft = workflow_gemini.GeminiClient(root, 1.0, ledger=ledger).transcribe_timed(_clip(root / "clip.wav"))

        self.assertEqual(draft["text"], "好，师父你好。")
        self.assertEqual([(word["text_start"], word["text_end"]) for word in draft["words"]],
                         [(0, 1), (2, 4), (4, 5), (5, 6)])
        self.assertEqual(draft["words"][-1]["end_seconds"], 1.15)
        send = Mock()
        ledger.send(SimpleNamespace(interactions=SimpleNamespace(create=send)))
        request = send.call_args.kwargs
        self.assertEqual(request["model"], "gemini-3.5-transcribe")
        self.assertEqual(request["generation_config"]["transcription_config"], {
            "mode": {"type": "verbatim", "timestamp_granularities": ["word"]},
            "language_codes": ["cmn-Hans-CN"],
        })

    def test_missing_or_out_of_range_word_timing_remains_unanchored(self):
        response = _interaction(
            "师父好",
            {"type": "word_info", "text": "师父", "start_offset": "0s", "end_offset": "1s"},
            {"type": "word_info", "text": "好", "start_offset": "1s", "end_offset": "10s"},
        )
        result = workflow_gemini._timed_transcript(response, 2.0)
        self.assertEqual(result["text"], "师父好")
        self.assertEqual([word["text"] for word in result["words"]], ["师父"])
        response["status"] = "in_progress"
        with self.assertRaisesRegex(ValueError, "did not complete"):
            workflow_gemini._timed_transcript(response, 2.0)

    def test_openai_uses_documented_language_and_confirmed_terms(self):
        ledger = RecordingLedger({"text": "  师父好。 "})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client = workflow_openai.OpenAIClient(root, 1.0, vocabulary=["师父"], ledger=ledger)
            self.assertEqual(client.transcribe(_clip(root / "clip.wav")), "师父好。")
        send = Mock()
        ledger.send(SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=send))))
        self.assertEqual(send.call_args.kwargs["languages"], ["zh-cn"])
        self.assertEqual(send.call_args.kwargs["keywords"], ["师父"])
        self.assertNotIn("timestamp_granularities", send.call_args.kwargs)


class SubtitleExportTests(TestCase):
    def test_groups_real_repetitions_and_uses_recording_time(self):
        text = "南无阿弥陀佛。南无阿弥陀佛。"
        words = [
            {"text": text[:6], "text_start": 0, "text_end": 6, "start_ms": 1000, "end_ms": 2100},
            {"text": text[7:13], "text_start": 7, "text_end": 13, "start_ms": 2400, "end_ms": 3500},
        ]
        cues = subtitle_export.group_timed_words(text, words, start_ms=500, end_ms=4000)
        self.assertEqual([cue["text"] for cue in cues], ["南无阿弥陀佛。"] * 2)
        srt = subtitle_export.render_srt(cues, 4000, text)
        self.assertIn("00:00:01,000 --> 00:00:02,100", srt)
        self.assertIn("00:00:02,400 --> 00:00:03,500", srt)

    def test_does_not_break_a_number_between_annotated_tokens(self):
        text = "2026年"
        words = [
            {"text": "20", "text_start": 0, "text_end": 2, "start_ms": 0, "end_ms": 800},
            {"text": "26", "text_start": 2, "text_end": 4, "start_ms": 1200, "end_ms": 1500},
            {"text": "年", "text_start": 4, "text_end": 5, "start_ms": 1500, "end_ms": 1900},
        ]
        cues = subtitle_export.group_timed_words(text, words, start_ms=0, end_ms=2000)
        self.assertEqual([cue["text"] for cue in cues], ["2026年"])

    def test_missing_anchor_needs_review_instead_of_a_guessed_time(self):
        words = [{"text": "师父", "text_start": 0, "text_end": 2, "start_ms": 0, "end_ms": 500}]
        with self.assertRaisesRegex(ValueError, "no word timing"):
            subtitle_export.group_timed_words("师父好。", words, start_ms=0, end_ms=1000)

    def test_validation_preserves_spaces_punctuation_and_cue_order(self):
        first = {"text": "Thank you。", "start_ms": 0, "end_ms": 1000}
        second = {"text": "师父。", "start_ms": 1200, "end_ms": 2000}
        subtitle_export.validate_manual_cues([first, second], "Thank you。师父。", 0, 2000)
        with self.assertRaisesRegex(ValueError, "change the selected wording"):
            subtitle_export.render_srt([{**first, "text": "Thankyou。"}, second], 2000, "Thank you。师父。")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            subtitle_export.render_srt([first, {**second, "start_ms": 800}], 2000, "Thank you。师父。")
