"""Bounded CPU audio decoding and windows on the original recording clock."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import tempfile
import wave


SAMPLE_RATE = 16_000
MAX_SECONDS = 120 * 60
MAX_BYTES = 512 * 1024 * 1024
AUDIO_SUFFIXES = frozenset({
    ".m4a", ".mp3", ".wav", ".flac", ".aac", ".mp4", ".mov", ".webm", ".ogg", ".aiff", ".aif", ".opus",
})
_FORMAT_OPTIONS = [
    "-protocol_whitelist", "file,pipe",
    "-format_whitelist", "mov,mp4,m4a,3gp,3g2,mj2,mp3,wav,flac,aac,matroska,webm,ogg,aiff",
]
_FRAME_SAMPLES = 480  # WebRTC VAD uses fixed 30 ms frames at 16 kHz.
_CORE_SAMPLES = 180 * SAMPLE_RATE
_CONTEXT_SAMPLES = 5 * SAMPLE_RATE


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    request_start: int
    request_end: int


@dataclass(frozen=True)
class Recording:
    pcm: Path
    samples: int

    def clip(self, start: int, end: int, output: Path):
        if not 0 <= start < end <= self.samples:
            raise ValueError("Audio clip is outside the recording")
        with self.pcm.open("rb") as source:
            source.seek(2 * start)
            data = source.read(2 * (end - start))
        if len(data) != 2 * (end - start):
            raise ValueError("Decoded audio ended before the clip")
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with wave.open(temporary, "wb") as out:
                    out.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                    out.writeframes(data)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            temporary_path.replace(output)
        finally:
            temporary_path.unlink(missing_ok=True)

    def speech(self) -> list[tuple[int, int]]:
        import webrtcvad

        detector = webrtcvad.Vad(2)
        detected = []
        with self.pcm.open("rb") as source:
            for start in range(0, self.samples, _FRAME_SAMPLES):
                frame = source.read(_FRAME_SAMPLES * 2).ljust(_FRAME_SAMPLES * 2, b"\x00")
                if not detector.is_speech(frame, SAMPLE_RATE):
                    continue
                end = min(self.samples, start + _FRAME_SAMPLES)
                if detected and start - detected[-1][1] <= SAMPLE_RATE // 4:
                    detected[-1] = (detected[-1][0], end)
                else:
                    detected.append((start, end))
        return detected


def probe(source: Path) -> float | None:
    if source.suffix.lower() not in AUDIO_SUFFIXES or not 0 < source.stat().st_size <= MAX_BYTES:
        raise ValueError("Choose a supported, nonempty recording up to 512 MiB")
    try:
        output = subprocess.run(
            ["ffprobe", "-v", "error", *_FORMAT_OPTIONS, "-select_streams", "a:0",
             "-show_entries", "stream=index:format=duration", "-of", "json", str(source.resolve())],
            check=True, capture_output=True, timeout=30,
        ).stdout
        info = json.loads(output)
    except (subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("The recording could not be read") from exc
    if not info.get("streams"):
        raise ValueError("The recording does not contain an audio stream")
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        return None  # e.g. a WebM file with an unknown container duration.
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("The recording duration is invalid")
    if duration > MAX_SECONDS:
        raise ValueError("The recording exceeds the 120-minute limit")
    return duration


@contextmanager
def decode(source: Path, scratch: Path, *, limit_seconds: float | None = None):
    probe(source)
    if limit_seconds is not None and not 0 < limit_seconds <= MAX_SECONDS:
        raise ValueError("The pilot length must be between 0 and 120 minutes")
    limit = MAX_SECONDS + 1 if limit_seconds is None else limit_seconds
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="audio-", dir=scratch) as directory:
        pcm = Path(directory) / "recording.pcm"
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", *_FORMAT_OPTIONS, "-i", str(source.resolve()),
                 "-map", "0:a:0", "-threads", "1", "-t", str(limit), "-ar", str(SAMPLE_RATE),
                 "-ac", "1", "-c:a", "pcm_s16le", "-f", "s16le", str(pcm)],
                check=True, capture_output=True, timeout=180,
            )
        except subprocess.SubprocessError as exc:
            raise ValueError("FFmpeg could not decode the selected audio") from exc
        samples = pcm.stat().st_size // 2
        if not 0 < samples <= MAX_SECONDS * SAMPLE_RATE:
            raise ValueError("The decoded duration must be between 0 and 120 minutes")
        yield Recording(pcm, samples)


def windows(samples: int, speech: list[tuple[int, int]]) -> list[Window]:
    """Prefer a verified pause near each nominal cut, keeping unique core ownership."""
    pauses = [(0, speech[0][0])] if speech else []
    pauses.extend((end, next_start) for (_, end), (next_start, _) in zip(speech, speech[1:]))
    if speech:
        pauses.append((speech[-1][1], samples))
    boundaries = [0]
    while boundaries[-1] + _CORE_SAMPLES + _CONTEXT_SAMPLES < samples:
        desired = boundaries[-1] + _CORE_SAMPLES
        candidates = [
            min(max(desired, start + _FRAME_SAMPLES), end - _FRAME_SAMPLES)
            for start, end in pauses
            if end - start >= 2 * _FRAME_SAMPLES
            and start <= desired + _CONTEXT_SAMPLES
            and end >= desired - _CONTEXT_SAMPLES
        ]
        cut = min(candidates, key=lambda position: (abs(desired - position), position), default=desired)
        boundaries.append(cut if abs(desired - cut) <= _CONTEXT_SAMPLES else desired)
    boundaries.append(samples)
    return [
        Window(start, end, max(0, start - _CONTEXT_SAMPLES), min(samples, end + _CONTEXT_SAMPLES))
        for start, end in zip(boundaries, boundaries[1:])
    ]
