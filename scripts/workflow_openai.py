"""OpenAI recorded-audio transcription."""
from __future__ import annotations

from pathlib import Path

from workflow_cloud import CloudError, CloudLedger, api_key, audio_clip, digest


TRANSCRIBE_MODEL = "gpt-transcribe"
_TRANSCRIBE_RATE = 0.0045  # USD / audio minute
_VERBATIM_CHINESE = "逐字记录这段录音，使用简体中文。保留原话、停顿词、重复、数字和专有名词，不要改写、翻译或总结。"


def _api_key() -> str:
    return api_key("OPENAI_API_KEY")


class OpenAIClient:
    def __init__(self, job: Path, budget_usd: float, *, vocabulary: list[str] | None = None,
                 ledger: CloudLedger | None = None, context: str | None = None):
        self.ledger = ledger if ledger is not None else CloudLedger(job, budget_usd)
        self.vocabulary = list(vocabulary or [])
        self.prompt = context or _VERBATIM_CHINESE
        if any(not isinstance(word, str) or not word.strip() or any(char in word for char in "<>\r\n")
               for word in self.vocabulary):
            raise ValueError("OpenAI vocabulary requires nonempty lines without angle brackets")
        self._client = None

    def _sdk(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise CloudError("Install the pinned OpenAI SDK for OpenAI requests") from None
            key = _api_key()
            try:
                self._client = OpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=600.0)
            except Exception:
                raise CloudError("Could not initialize the OpenAI SDK") from None
        return self._client

    def transcribe(self, clip: Path, *, hints: bool = True, purpose: str = "draft") -> str:
        if purpose not in {"draft", "blind"}:
            raise ValueError("Transcription purpose must be draft or blind")
        data, duration = audio_clip(clip, 25_000_000)
        context = {"languages": ["zh-cn"]}
        if hints and self.vocabulary:
            context["keywords"] = self.vocabulary
        audit = {"endpoint": "/v1/audio/transcriptions", "audio_sha256": digest(data),
                 "audio_bytes": len(data), "duration_seconds": duration, "context": context,
                 "prompt_sha256": digest(self.prompt.encode("utf-8"))}
        estimated = duration * _TRANSCRIBE_RATE / 60

        def send(sdk):
            return sdk.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=("clip.wav", data, "audio/wav"),
                                                   prompt=self.prompt, response_format="json", **context)

        def read(response, raw):
            text = raw.get("text")
            if not isinstance(text, str):
                raise ValueError("OpenAI did not return transcription text")
            return text.strip()

        return self.ledger.request("OpenAI", TRANSCRIBE_MODEL, purpose, audit, estimated * 1.25,
                                   self._sdk, send, read, lambda usage: estimated)
