"""Shared Chinese transcription, evidence review, and revision rendering."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import tempfile

import opencc

from subtitle_export import group_timed_words, render_srt, validate_manual_cues
from workflow_audio import SAMPLE_RATE, Recording, Window, decode, windows
from workflow_review import compare


CHOICES = ("gemini", "openai", "suggestion", "edited", "inaudible", "no_speech")
PIPELINE_VERSION = "2026-11-cpu-1"
SIMPLIFIED = opencc.OpenCC("t2s.json")
_REVIEW_SAMPLES = 30 * SAMPLE_RATE
_MIN_COVERAGE_GAP = SAMPLE_RATE
_MS = SAMPLE_RATE // 1000


def indexed_text(text):
    """Keep a mapping from simplified lexical characters to the original wording."""
    chars, offsets = [], []
    for index, char in enumerate(text):
        for part in SIMPLIFIED.convert(char).casefold():
            if part.isalnum():
                chars.append(part)
                offsets.append(index)
    return "".join(chars), offsets


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _issue(category, reason, detail, *, start=None, end=None):
    issue = {"category": category, "reason": reason, "detail": detail}
    if start is not None:
        issue.update(start_sample=start, end_sample=end)
    return issue


def _disagreement_bounds(text, span, words, passage_start, passage_end):
    left, right = span
    words = [word for word in words if any(char.isalnum() for char in word["text"])]
    if left == right:
        # An insertion has no Gemini words of its own. Replay both neighbors
        # only when they tightly bracket the corresponding lexical boundary.
        before = next((word for word in reversed(words) if word["text_end"] <= left), None)
        after = next((word for word in words if word["text_start"] >= right), None)
        if before is None or after is None:
            return None, None
        if (any(char.isalnum() for char in text[before["text_end"]:after["text_start"]])
                or not 0 <= after["start_sample"] - before["end_sample"] <= SAMPLE_RATE):
            return None, None
        anchors = [before, after]
    else:
        anchors = [word for word in words if word["text_start"] < right and word["text_end"] > left]
        if not anchors:
            return None, None
        cursor = left
        for word in anchors:
            if any(char.isalnum() for char in text[cursor:word["text_start"]]):
                return None, None
            cursor = word["text_end"]
        if any(char.isalnum() for char in text[cursor:right]):
            return None, None

    start, end = anchors[0]["start_sample"], anchors[-1]["end_sample"]
    if not passage_start <= start < end <= passage_end or any(
        not start <= word["start_sample"] < word["end_sample"] <= end for word in anchors
    ):
        return None, None
    return start, end


def _compare(gemini, openai, vocabulary, words, passage_start, passage_end):
    spans = {"gemini": [], "openai": []}
    issues = {}
    if openai is None:
        return spans, []
    for difference in compare(gemini, openai)["differences"]:
        for source, key in (("gemini", "left_span"), ("openai", "right_span")):
            start, end = difference[key]
            spans[source].append({"start": start, "end": end})
        if difference["risk"] == "high":
            reason = difference["reason"]
            additional = reason.startswith("additional_")
            category = "coverage" if reason == "missing_speech" or additional else "wording"
            reason = "candidate_addition" if additional else reason
            if "start_sample" in issues.get(reason, {}):
                continue
            start, end = _disagreement_bounds(gemini, difference["left_span"], words, passage_start, passage_end)
            issues[reason] = _issue(
                category, reason,
                "The two transcriptions disagree. Listen to the original audio before choosing wording.",
                start=start, end=end,
            )
    if any(SIMPLIFIED.convert(gemini).count(term) != SIMPLIFIED.convert(openai).count(term)
           for term in map(SIMPLIFIED.convert, vocabulary)):
        issues["glossary_difference"] = _issue("wording", "glossary_difference", "Check the confirmed terms against the recording.")
    return spans, list(issues.values())


def _timed_words(draft, window: Window):
    words = []
    for source in draft["words"]:
        start = window.request_start + round(source["start_seconds"] * SAMPLE_RATE)
        end = window.request_start + round(source["end_seconds"] * SAMPLE_RATE)
        if not window.request_start <= start <= end <= window.request_end:
            raise ValueError("A Gemini word falls outside the requested audio interval")
        words.append({
            "text": source["text"], "text_start": source["text_start"], "text_end": source["text_end"],
            "start_sample": start, "end_sample": end,
        })
    return words


def _core_selection(words, window: Window, previous_words):
    """Keep one source-clock owner and remove only verified duplicate seam words."""
    first = next((i for i, word in enumerate(words)
                  if word["start_sample"] + word["end_sample"] >= 2 * window.start), len(words))
    end = len(words)
    if window.end < window.request_end:
        end = next((i for i, word in enumerate(words[first:], first)
                    if word["start_sample"] + word["end_sample"] >= 2 * window.end), len(words))
    for word in words[first:end]:
        lexical, _ = indexed_text(word["text"])
        if not lexical:
            break
        matches = []
        for previous in previous_words:
            overlap = min(word["end_sample"], previous["end_sample"]) - max(word["start_sample"], previous["start_sample"])
            if (overlap > 0 and 4 * overlap >= 3 * min(
                word["end_sample"] - word["start_sample"], previous["end_sample"] - previous["start_sample"]
            ) and lexical == indexed_text(previous["text"])[0]):
                matches.append(previous)
        if len(matches) != 1:
            break
        first += 1
    return first, end


def _text_cuts(reference, candidate):
    """Map interior boundaries only through unique matching lexical context."""
    left, left_offsets = indexed_text(reference)
    right, right_offsets = indexed_text(candidate)
    cuts = {0: 0, len(reference): len(candidate)}
    for a, b, length in SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks():
        if length < 4:
            continue
        for step in range(1, length):
            begin, end = max(a, a + step - 3), min(a + length, a + step + 3)
            anchor = left[begin:end]
            if len(anchor) < 4 or left.find(anchor) != left.rfind(anchor) or right.find(anchor) != right.rfind(anchor):
                continue
            cuts[left_offsets[a + step]] = right_offsets[b + step]
    return cuts


def _punctuation_before(text, words, index):
    if index == 0:
        return 0
    if index == len(words):
        return len(text)
    offset = words[index]["text_start"]
    return offset


def _coverage_gaps(speech, words, start, end):
    # Allow modest provider boundary drift before asking the reviewer to listen.
    pad = SAMPLE_RATE // 4
    coverage = [(max(start, word["start_sample"] - pad), min(end, word["end_sample"] + pad)) for word in words]
    gaps = []
    for begin, finish in speech:
        left, right = max(start, begin), min(end, finish)
        if right - left < _MIN_COVERAGE_GAP:
            continue
        cursor = left
        for a, b in coverage:
            if b <= cursor:
                continue
            if a >= right:
                break
            if a - cursor >= _MIN_COVERAGE_GAP:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        if right - cursor >= _MIN_COVERAGE_GAP:
            gaps.append((cursor, right))
    return gaps


def _windows_passages(draft, other, window, speech, vocabulary, previous_words):
    source = draft["text"]
    words = _timed_words(draft, window)
    first, last = _core_selection(words, window, previous_words)
    cuts = _text_cuts(source, other)
    core_left = _punctuation_before(source, words, first)
    core_right = _punctuation_before(source, words, last)
    candidate_available = core_left in cuts and core_right in cuts
    boundaries = [(first, window.start)]
    for index in range(first + 1, last):
        before, after = words[index - 1], words[index]
        if before["end_sample"] > after["start_sample"]:
            continue
        split = (before["end_sample"] + after["start_sample"]) // 2
        if (split - boundaries[-1][1] >= _REVIEW_SAMPLES and split < window.end
                and (not candidate_available or after["text_start"] in cuts)):
            boundaries.append((index, split))
    boundaries.append((last, window.end))
    passages = []
    for (a, start), (b, end) in zip(boundaries, boundaries[1:]):
        left, right = _punctuation_before(source, words, a), _punctuation_before(source, words, b)
        original = source[left:right] if a != b else source if not words else ""
        text = original.strip()
        shift = left + len(original) - len(original.lstrip())
        selected = [{**word, "text_start": word["text_start"] - shift, "text_end": word["text_end"] - shift}
                    for word in words[a:b]]
        candidate = other[cuts[left]:cuts[right]].strip() if candidate_available else None
        if not words:
            candidate = other.strip()
        elif a == b and indexed_text(source)[0] != indexed_text(other)[0]:
            candidate = None
        candidates = {"gemini": text}
        if candidate is not None:
            candidates["openai"] = candidate
        spans, issues = _compare(text, candidate, vocabulary, selected, start, end)
        if (not words and (text or candidate)
                and (window.start != window.request_start or window.end != window.request_end)):
            issues.append(_issue("seam", "unanchored_window_text", "Check whether unanchored words belong to this audio window."))
        if candidate is None and other.strip():
            issues.append(_issue("seam", "unlocated_candidate", "OpenAI wording could not be tied to this window boundary."))
        if any(word["start_sample"] < start or word["end_sample"] > end for word in selected):
            issues.append(_issue("seam", "word_crosses_boundary", "A word touches the next or previous audio passage."))
        if a == first and previous_words and selected and previous_words[-1]["end_sample"] > selected[0]["start_sample"]:
            issues.append(_issue("seam", "overlapping_window_words", "Two neighboring model windows place words at the same time."))
        for gap_start, gap_end in _coverage_gaps(speech, selected, start, end):
            issues.append(_issue("coverage", "speech_without_words", "Possible speech has no timed words.",
                                 start=gap_start, end=gap_end))
        block_id = f"{start}-{end}"
        passages.append({
            "id": block_id, "start_sample": start, "end_sample": end,
            "candidates": candidates, "candidate_spans": spans, "issues": issues,
            "words": selected, "text": text, "unplaced_openai": other if candidate is None else None,
        })
    return passages, [word for word in words[first:last] if word["end_sample"] > window.end - 2 * SAMPLE_RATE]


def _clips(recording: Recording, job: Path, passage: dict):
    start, end = passage["start_sample"], passage["end_sample"]
    path = Path("clips") / f"audio-{start}-{end}.wav"
    recording.clip(start, end, job / path)
    return [{"path": str(path), "start_seconds": start / SAMPLE_RATE, "end_seconds": end / SAMPLE_RATE}]


def transcribe(source, job, manifest, gemini, openai, operation, *, authorize_duration=None, checkpoint=None):
    """Decode once and durably retain both independent candidates for each audio window."""
    source, job = Path(source), Path(job)
    operation.update("audio_preflight")
    with decode(source, job, limit_seconds=manifest.get("limit_seconds")) as recording:
        if authorize_duration is not None:
            authorize_duration(recording.samples / SAMPLE_RATE)
        speech = recording.speech()
        spans = windows(recording.samples, speech)
        passages = []
        previous_words = []
        for index, window in enumerate(spans, 1):
            operation.update("cloud_draft", completed=index - 1, total=len(spans))
            cache = job / "windows" / f"{index:04d}.json"
            identity = {"window": asdict(window), "source_sha256": manifest["source_sha256"],
                        "vocabulary": manifest["vocabulary"], "pipeline": PIPELINE_VERSION,
                        "models": manifest.get("models")}
            saved = json.loads(cache.read_text("utf-8")) if cache.is_file() else {}
            if saved.get("identity") != identity:
                clip = job / ".working" / f"{index:04d}.wav"
                recording.clip(window.request_start, window.request_end, clip)
                try:
                    saved = {"identity": identity, "gemini": gemini.transcribe_timed(clip),
                             "openai": openai.transcribe(clip, purpose="draft")}
                    _write_json(cache, saved)
                finally:
                    clip.unlink(missing_ok=True)
            if checkpoint is not None:
                checkpoint(cache)
            blocks, previous_words = _windows_passages(
                saved["gemini"], saved["openai"], window, speech, manifest["vocabulary"], previous_words)
            for block in blocks:
                block["clips"] = _clips(recording, job, block)
            passages.extend(blocks)
        operation.update("review_draft", completed=len(spans), total=len(spans))
        return {"samples": recording.samples, "passages": passages}


def _anchors(text):
    # Preserve punctuation inside numbers and Latin words when deciding if an edit is lexical.
    positions = []
    for index, char in enumerate(text):
        previous = text[index - 1] if index else ""
        after = text[index + 1:index + 2]
        internal = char in ".'’-–/," and previous and after and all(
            item.isascii() and item.isalnum() for item in (previous, after))
        numeric = (char in "-+−" and after.isdecimal()) or (char in "%‰％" and previous.isdecimal())
        if char.isalnum() or internal or numeric:
            positions.append(index)
    return "".join(text[index] for index in positions), positions


def _project_words(original, selected, words):
    original_lexical, source = _anchors(original)
    chosen_lexical, target = _anchors(selected)
    if SIMPLIFIED.convert(original_lexical) != chosen_lexical or len(source) != len(target) or not target:
        raise ValueError("Edited words need manual time boundaries")
    projected = []
    for word in words:
        a = bisect_left(source, word["text_start"])
        b = bisect_left(source, word["text_end"])
        if b <= a:
            continue
        start, end = target[a], target[b - 1] + 1
        projected.append({"text": selected[start:end], "text_start": start, "text_end": end,
                          "start_ms": math.ceil(word["start_sample"] / _MS),
                          "end_ms": math.floor(word["end_sample"] / _MS)})
    return projected


def make_decision(choice, text, timings):
    if not isinstance(choice, str) or choice not in CHOICES:
        raise ValueError("Select a candidate or enter your review decision")
    if choice == "edited":
        if not isinstance(text, str) or not text.strip() or len(text) > 24_000 or "\x00" in text:
            raise ValueError("Enter the complete corrected wording")
    elif text is not None:
        raise ValueError("Provide wording only for an edited decision")
    if timings is not None and choice in {"inaudible", "no_speech"}:
        raise ValueError("Inaudible and no-speech decisions have fixed timing")
    result = {"choice": choice}
    if text is not None:
        result["text"] = text
    if timings is not None:
        result["timings"] = timings
    return result


def _resolve(block, decision):
    issues = list(block["issues"]) if decision is None else []
    choice = decision["choice"] if decision else "suggestion"
    start = math.ceil(block["start_sample"] / _MS)
    end = math.floor(block["end_sample"] / _MS)
    original = block["text"]
    if choice == "no_speech":
        return "", [], issues
    if choice == "inaudible":
        label = "［听不清］"
        if end - start < 250:
            raise ValueError("The passage is too short for an inaudible caption. Mark it as no speech or edit it.")
        return label, [{"text": label, "start_ms": start, "end_ms": min(end, start + 2000)}], issues
    if choice == "edited":
        text = decision["text"].strip()
    elif choice == "suggestion":
        text = original
        if decision and not text.strip():
            raise ValueError("Select a nonempty reading or mark the passage as no speech")
    else:
        text = block["candidates"].get(choice, "")
        if not text.strip():
            raise ValueError("That candidate contains no usable text for this passage")
    text = SIMPLIFIED.convert(text.strip())
    if not text:
        return "", [], issues
    if decision and "timings" in decision:
        timings = decision["timings"]
        if (not isinstance(timings, list) or not 1 <= len(timings) <= 100
                or any(not isinstance(cue, dict) or not isinstance(cue.get("text"), str) for cue in timings)):
            raise ValueError("Provide between 1 and 100 timed subtitles")
        timings = [{**cue, "text": SIMPLIFIED.convert(cue["text"])} for cue in timings]
        validate_manual_cues(timings, text, start, end)
        return text, timings, issues
    try:
        words = _project_words(original, text, block["words"])
        cues = group_timed_words(text, words, start_ms=start, end_ms=end)
    except ValueError:
        issues.append(_issue("timing", "manual_timing_required", "Set cue boundaries against the recording."))
        cues = []
    return text, cues, issues


def validate_decision(data, passage_id, choice, text, timings=None):
    block = next((row for row in data["passages"] if row["id"] == passage_id), None)
    if block is None:
        raise ValueError("This passage is not in the current job")
    decision = make_decision(choice, text, timings)
    _resolve(block, decision)
    return decision


def build_review(data, decisions, directory):
    if not isinstance(decisions, dict):
        raise ValueError("Expected decisions keyed by passage ID")
    blocks = data["passages"]
    if decisions.keys() - {block["id"] for block in blocks}:
        raise ValueError("A review decision refers to an unknown passage")
    summaries, pending, cues, lines = [], [], [], []
    previous_sample = 0
    for block in blocks:
        if block["start_sample"] != previous_sample or block["end_sample"] <= previous_sample:
            raise ValueError("Passages do not cover the recording exactly once")
        previous_sample = block["end_sample"]
        decision = decisions.get(block["id"])
        text, timings, issues = _resolve(block, decision)
        entry = {
            "id": block["id"], "start_seconds": block["start_sample"] / SAMPLE_RATE,
            "end_seconds": block["end_sample"] / SAMPLE_RATE, "selected_text": text,
            "suggested_text": SIMPLIFIED.convert(block["text"]), "candidates": block["candidates"],
            "candidate_spans": block["candidate_spans"], "human_reviewed": decision is not None,
            "timings": timings, "clips": block["clips"], "issues": issues,
            "unplaced_openai": block.get("unplaced_openai"),
        }
        summaries.append(entry)
        if issues:
            pending.append(entry)
        if text:
            lines.append(text)
            cues.extend(timings)
    if previous_sample != data["samples"]:
        raise ValueError("The review does not cover the full recording")
    expected = "\n".join(lines)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    srt = destination / "zh.srt"
    if expected and all(not row["selected_text"] or row["timings"] for row in summaries):
        srt.write_text(render_srt(cues, data["samples"] // _MS, expected), "utf-8")
    else:
        srt.unlink(missing_ok=True)
    (destination / "zh.txt").write_text(expected + "\n" if expected else "", "utf-8")
    summary = {"duration_seconds": data["samples"] / SAMPLE_RATE, "audio_blocks": len(blocks),
               "human_decisions": len(decisions), "needs_listening": len(pending),
               "listening_seconds": round(sum(entry["end_seconds"] - entry["start_seconds"] for entry in pending), 1),
               "status": "needs_review" if pending else "ready_to_publish" if expected else "no_speech"}
    for name, value in (("passages", summaries), ("review", pending), ("summary", summary), ("decisions", decisions)):
        _write_json(destination / f"{name}.json", value)
    return summary
