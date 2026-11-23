"""Gemini verbatim transcription and word-level timing."""
from __future__ import annotations

import base64
import math
from pathlib import Path
import re

from workflow_cloud import CloudError, CloudLedger, api_key, audio_clip, digest, encode_json, token_cost


TRANSCRIBE_MODEL = "gemini-3.5-transcribe"
_RATES = (2.0, 12.0)
_DURATION = re.compile(r"[0-9]+(?:\.[0-9]{1,9})?s")


def _api_key() -> str:
    return api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _offset(value):
    if not isinstance(value, str) or not _DURATION.fullmatch(value):
        return None
    seconds = float(value[:-1])
    return seconds if math.isfinite(seconds) else None


def _timed_transcript(raw: dict, duration: float) -> dict:
    if raw.get("status") != "completed":
        raise ValueError("Gemini did not complete the interaction")

    parts, words = [], []
    text_offset = 0
    for step in raw.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for content in step.get("content", []):
            if content.get("type") != "text":
                continue
            text = content["text"]
            if not isinstance(text, str):
                raise ValueError("Gemini did not return transcription text")
            parts.append(text)
            cursor = 0
            byte_positions = {0: 0}
            byte_index = 0
            for position, char in enumerate(text, 1):
                byte_index += len(char.encode("utf-8"))
                byte_positions[byte_index] = position

            for annotation in content.get("annotations", []):
                if annotation.get("type") != "word_info":
                    continue
                word = annotation.get("text") or ""
                if not isinstance(word, str):
                    continue
                start_index = annotation.get("start_index")
                end_index = annotation.get("end_index")
                if start_index is None and end_index is None:
                    # The indices are optional. Search in order, leaving any unanchored
                    # lexical text visible to the review pipeline.
                    if not word.strip():
                        continue
                    start = text.find(word, cursor)
                    end = start + len(word)
                elif type(start_index) is int and type(end_index) is int:
                    start = byte_positions.get(start_index, -1)
                    end = byte_positions.get(end_index, -1)
                else:
                    continue
                if not cursor <= start < end <= len(text):
                    continue
                span = text[start:end]
                if not span.strip() or (word and span.strip() != word.strip()):
                    continue
                cursor = end
                begin = _offset(annotation.get("start_offset"))
                finish = _offset(annotation.get("end_offset"))
                if begin is None or finish is None or not 0 <= begin <= finish <= duration:
                    continue
                words.append({
                    "text": span,
                    "text_start": text_offset + start,
                    "text_end": text_offset + end,
                    "start_seconds": begin,
                    "end_seconds": finish,
                })
            text_offset += len(text)

    if not parts:
        raise ValueError("Gemini did not return transcription text")
    return {"text": "".join(parts), "words": words}


class GeminiClient:
    def __init__(self, job: Path, budget_usd: float, *, ledger: CloudLedger | None = None):
        self.ledger = ledger if ledger is not None else CloudLedger(job, budget_usd)
        self._client = None

    def _sdk(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError:
                raise CloudError("Install the pinned google-genai SDK for Gemini requests") from None
            key = _api_key()
            try:
                self._client = genai.Client(
                    api_key=key, enterprise=False,
                    http_options=types.HttpOptions(base_url="https://generativelanguage.googleapis.com",
                                                  api_version="v1beta", timeout=600000,
                                                  retry_options=types.HttpRetryOptions(attempts=1)),
                )
            except Exception:
                raise CloudError("Could not initialize the Gemini SDK") from None
        return self._client

    def transcribe_timed(self, clip: Path) -> dict:
        data, duration = audio_clip(clip, 14_000_000)
        if duration > 1800:
            raise ValueError("Gemini word timestamps require audio clips of at most 30 minutes")
        options = {
            "transcription_config": {
                "mode": {"type": "verbatim", "timestamp_granularities": ["word"]},
                "language_codes": ["cmn-Hans-CN"],
            },
            "max_output_tokens": 16384,
        }
        payload = {
            "model": TRANSCRIBE_MODEL, "generation_config": options, "store": False,
            "automatic_continuation": False,
            "input": [{"type": "audio", "data": base64.b64encode(data).decode("ascii"), "mime_type": "audio/wav"}],
        }
        # These fields are part of the recorded request identity used to resume and retry jobs.
        audit = {"api_version": "v1beta", "generation_config": options,
                 "prompt_sha256": digest(b""), "schema": None,
                 "audio_sha256": digest(data), "duration_seconds": duration, "audio_bytes": len(data)}
        rate_in, rate_out = _RATES
        input_tokens = math.ceil(duration * 100) + len(encode_json(options)) + len(encode_json(audit["schema"])) + 1024
        reservation = (input_tokens * rate_in + options["max_output_tokens"] * rate_out) / 1_000_000

        def read(_response, raw):
            return _timed_transcript(raw, duration)

        def cost(usage):
            return token_cost(usage, rate_in, rate_out, input_key="total_input_tokens",
                              output_key="total_output_tokens", thought_key="total_thought_tokens")

        return self.ledger.request("Gemini", TRANSCRIBE_MODEL, "timed_draft", audit, reservation, self._sdk,
                                   lambda sdk: sdk.interactions.create(**payload), read, cost)
