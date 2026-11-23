"""Shared Chinese transcription, evidence review, and revision rendering."""

from __future__ import annotations

from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from difflib import SequenceMatcher
from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile

import opencc

from subtitle_export import _wrapped, group_timed_words, render_srt, validate_manual_cues
from workflow_audio import DEFAULT_CORE_SECONDS, SAMPLE_RATE, Recording, Window, decode, windows
from workflow_review import compare


CHOICES = ("gemini", "openai", "suggestion", "edited", "inaudible", "no_speech")
LEGACY_PIPELINE_VERSION = "2026-11-cpu-1"
PIPELINE_VERSION = "2026-11-cpu-2"
SIMPLIFIED = opencc.OpenCC("t2s.json")
_REVIEW_SAMPLES = 30 * SAMPLE_RATE
_MIN_COVERAGE_GAP = SAMPLE_RATE
_SEAM_SAMPLES = SAMPLE_RATE
_SEAM_SILENCE_PAD = SAMPLE_RATE // 4
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


def _term_spans(text, terms):
    lexical, offsets = indexed_text(text)
    for term in terms:
        index = lexical.find(term)
        while index >= 0:
            yield offsets[index], offsets[index + len(term) - 1] + 1
            index = lexical.find(term, index + 1)


def _compare(gemini, openai, vocabulary, words, passage_start, passage_end):
    spans = {"gemini": [], "openai": []}
    issues = []
    if openai is None:
        return spans, []
    terms = {lexical for term in vocabulary if (lexical := indexed_text(term)[0])}
    glossary = {
        "gemini": list(_term_spans(gemini, terms)),
        "openai": list(_term_spans(openai, terms)),
    }
    for difference in compare(gemini, openai)["differences"]:
        term_difference = False
        for source, key in (("gemini", "left_span"), ("openai", "right_span")):
            start, end = difference[key]
            spans[source].append({"start": start, "end": end})
            term_difference |= any(a < end and start < b for a, b in glossary[source])
        if difference["risk"] == "high" or term_difference:
            reason = difference["reason"]
            additional = reason.startswith("additional_")
            category = "coverage" if reason == "missing_speech" or additional else "wording"
            if additional:
                reason = "candidate_addition"
            elif term_difference and difference["risk"] == "low":
                reason = "glossary_difference"
            start, end = _disagreement_bounds(gemini, difference["left_span"], words, passage_start, passage_end)
            issues.append(_issue(
                category, reason,
                "The two transcriptions disagree. Listen to the original audio before choosing wording.",
                start=start, end=end,
            ))
    if not issues and any(SIMPLIFIED.convert(gemini).count(term) != SIMPLIFIED.convert(openai).count(term)
                          for term in map(SIMPLIFIED.convert, vocabulary)):
        issues.append(_issue("wording", "glossary_difference", "Check the confirmed terms against the recording."))
    return spans, issues


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


def _core_selection(words, window: Window):
    """Find the nominal word range before reconciling neighboring requests."""
    first = next((i for i, word in enumerate(words)
                  if word["start_sample"] + word["end_sample"] >= 2 * window.start), len(words))
    end = len(words)
    if window.end < window.request_end:
        end = next((i for i, word in enumerate(words[first:], first)
                    if word["start_sample"] + word["end_sample"] >= 2 * window.end), len(words))
    return first, end


def _near_seam(words, cut):
    return [(index, key, word["start_sample"], word["end_sample"])
            for index, word in enumerate(words)
            if word["start_sample"] < cut + _SEAM_SAMPLES
            and word["end_sample"] > cut - _SEAM_SAMPLES
            and (key := _anchors(SIMPLIFIED.convert(word["text"]).casefold())[0])]


def _seam_matches(left, right):
    """Return word pairs with one plausible occurrence on each request."""
    from_left, from_right = {}, {}
    for i, text, start, end in left:
        for j, other, begin, finish in right:
            overlap = min(end, finish) - max(start, begin)
            if (text == other and abs(start + end - begin - finish) <= SAMPLE_RATE
                    and overlap >= -_SEAM_SILENCE_PAD):
                strong = overlap > 0 and 4 * overlap >= 3 * min(end - start, finish - begin)
                from_left.setdefault(i, []).append((j, strong))
                from_right.setdefault(j, []).append((i, strong))

    def unambiguous(possible):
        overlapping = [index for index, strong in possible if strong]
        if len(overlapping) == 1:
            return overlapping[0]
        return possible[0][0] if len(possible) == 1 else None

    return [(i, j) for i, possible in from_left.items()
            if (j := unambiguous(possible)) is not None and unambiguous(from_right[j]) == i]


def _reconcile_seam(left, right, cut, left_end, right_start, speech):
    left_near, right_near = _near_seam(left, cut), _near_seam(right, cut)
    matches = _seam_matches(left_near, right_near)
    matched_left, matched_right = {i for i, _ in matches}, {j for _, j in matches}

    def conflicts(end, start):
        # Each shared word should survive in exactly one window. A discarded
        # word with no match deserves a replay even when the other words agree.
        return (sum((i < end) == (j >= start) for i, j in matches)
                + sum(i >= end and i not in matched_left for i, *_ in left_near)
                + sum(j < start and j not in matched_right for j, *_ in right_near))

    candidates = []
    for i, j in matches:
        for end, start in ((i, j), (i + 1, j + 1)):
            last_end = left[end - 1]["end_sample"] if end else cut - _SEAM_SAMPLES
            next_start = right[start]["start_sample"] if start < len(right) else cut + _SEAM_SAMPLES
            if last_end > next_start:
                continue
            boundary = min(max(cut, last_end), next_start)
            if abs(boundary - cut) < _SEAM_SAMPLES:
                candidates.append((conflicts(end, start), abs(boundary - cut), boundary, end, start))
    if candidates:
        count, _, boundary, end, start = min(candidates)
        if count <= conflicts(left_end, right_start):
            return boundary, end, start, count > 0
    at_cut = any(start < cut + _SEAM_SILENCE_PAD and end > cut - _SEAM_SILENCE_PAD for start, end in speech)
    return cut, left_end, right_start, at_cut or conflicts(left_end, right_start) > 0


def _reconciled_windows(spans, saved, speech):
    timed = [_timed_words(result["gemini"], window) for window, result in zip(spans, saved, strict=True)]
    ranges = [_core_selection(words, window) for words, window in zip(timed, spans, strict=True)]
    starts = [window.start for window in spans] + [spans[-1].end]
    seam_issues = {}
    for index, (left, right) in enumerate(zip(timed, timed[1:]), 1):
        cut = starts[index]
        boundary, end, start, uncertain = _reconcile_seam(
            left, right, cut, ranges[index - 1][1], ranges[index][0], speech,
        )
        ranges[index - 1] = ranges[index - 1][0], end
        ranges[index] = start, ranges[index][1]
        starts[index] = boundary
        if uncertain:
            seam_issues[index] = _issue(
                "seam", "ambiguous_window_overlap", "Listen across this audio cut to check for missing or repeated words.",
                start=max(0, min(cut, boundary) - _SEAM_SAMPLES),
                end=min(starts[-1], max(cut, boundary) + _SEAM_SAMPLES),
            )
    for index, (window, words, selection) in enumerate(zip(spans, timed, ranges, strict=True)):
        adjusted = Window(starts[index], starts[index + 1], window.request_start, window.request_end)
        yield adjusted, words, selection, seam_issues.get(index)


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


def _windows_passages(draft, other, window, speech, vocabulary, previous_words, *,
                      word_selection=None, seam_issue=None):
    source = draft["text"]
    if word_selection is None:
        words = _timed_words(draft, window)
        first, last = _core_selection(words, window)
    else:
        words, (first, last) = word_selection
    cuts = _text_cuts(source, other) if other is not None else {}
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
        if other is not None and (not words or not other.strip()):
            candidate = other.strip()
        elif a == b and other is not None and indexed_text(source)[0] != indexed_text(other)[0]:
            candidate = None
        candidates = {"gemini": text}
        if candidate is not None:
            candidates["openai"] = candidate
        spans, issues = _compare(text, candidate, vocabulary, selected, start, end)
        if (not words and (text or candidate)
                and (window.start != window.request_start or window.end != window.request_end)):
            issues.append(_issue("seam", "unanchored_window_text", "Check whether unanchored words belong to this audio window."))
        if candidate is None and other and other.strip():
            issues.append(_issue("seam", "unlocated_candidate", "OpenAI wording could not be tied to this window boundary."))
        if not passages and seam_issue is not None:
            issues.append(seam_issue)
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
            "comparison_skipped": other is None,
        })
    return passages, [word for word in words[first:last] if word["end_sample"] > window.end - 2 * SAMPLE_RATE]


def _clips(recording: Recording, job: Path, passage: dict):
    start, end = passage["start_sample"], passage["end_sample"]
    for issue in passage["issues"]:
        if issue["reason"] == "ambiguous_window_overlap":
            start, end = min(start, issue["start_sample"]), max(end, issue["end_sample"])
    path = Path("clips") / f"audio-{start}-{end}.wav"
    recording.clip(start, end, job / path)
    return [{"path": str(path), "start_seconds": start / SAMPLE_RATE, "end_seconds": end / SAMPLE_RATE}]


def _comparison_windows(aligned, checkpoints, speech, vocabulary, source_hash):
    flagged = set()
    terms = {key for term in vocabulary if (key := indexed_text(term)[0])}
    for index, (window, words, selection, seam_issue) in enumerate(aligned):
        if seam_issue:
            flagged.update((index - 1, index))
            continue
        blocks, _ = _windows_passages(checkpoints[index]["gemini"], None, window, speech, vocabulary, [],
                                      word_selection=(words, selection))
        for block in blocks:
            lexical, _ = indexed_text(block["text"])
            if any(term in lexical for term in terms) or _resolve(block, None)[2]:
                flagged.add(index)
                break
    routine = sorted(set(range(len(aligned))) - flagged, key=lambda index: sha256(
        f"{source_hash}:{index}".encode("ascii"),
    ).digest())
    # Always audit at least one routine window when any are available.
    flagged.update(routine[:math.ceil(len(routine) / 10)])
    return flagged


def transcribe(source, job, manifest, gemini, openai, operation, *, authorize_duration=None, checkpoint=None):
    """Decode once and durably retain both independent candidates for each audio window."""
    source, job = Path(source), Path(job)
    version = manifest.get("pipeline", PIPELINE_VERSION)
    operation.update("audio_preflight")
    with (decode(source, job, limit_seconds=manifest.get("limit_seconds"),
                 media_clock=version != LEGACY_PIPELINE_VERSION) as recording,
          ThreadPoolExecutor(max_workers=2, thread_name_prefix="studio-asr") as executor):
        if authorize_duration is not None:
            authorize_duration(recording.samples / SAMPLE_RATE)
        speech = recording.speech()
        spans = windows(recording.samples, speech, core_seconds=manifest.get("core_seconds", DEFAULT_CORE_SECONDS))
        second_pass = partial(openai.transcribe, purpose="draft")

        def window_result(index, providers):
            window = spans[index]
            cache = job / "windows" / f"{index + 1:04d}.json"
            identity = {"window": asdict(window), "source_sha256": manifest["source_sha256"],
                        "vocabulary": manifest["vocabulary"], "pipeline": version,
                        "models": manifest.get("models")}
            if "context" in manifest:
                identity["context"] = manifest["context"]
            saved = json.loads(cache.read_text("utf-8")) if cache.is_file() else {}
            if saved.get("identity") != identity:
                saved = {"identity": identity}
            missing = [(provider, request) for provider, request in providers.items() if provider not in saved]
            if missing:
                clip = job / ".working" / f"{index + 1:04d}.wav"
                recording.clip(window.request_start, window.request_end, clip)
                pending = [(provider, executor.submit(request, clip)) for provider, request in missing]
                error = None
                try:
                    for provider, future in pending:
                        try:
                            saved[provider] = future.result()
                        except Exception as exc:
                            if error is None:
                                error = exc
                    _write_json(cache, saved)
                finally:
                    clip.unlink(missing_ok=True)
            else:
                error = None
            if checkpoint is not None:
                checkpoint(cache)
            if error is not None:
                raise error
            return saved

        targeted = manifest.get("comparison") == "targeted"
        providers = {"gemini": gemini.transcribe_timed}
        if not targeted:
            providers["openai"] = second_pass
        checkpoints = []
        for index in range(len(spans)):
            operation.update("cloud_draft", completed=index, total=len(spans))
            checkpoints.append(window_result(index, providers))

        aligned = list(_reconciled_windows(spans, checkpoints, speech))
        if targeted:
            targets = _comparison_windows(aligned, checkpoints, speech, manifest["vocabulary"], manifest["source_sha256"])
            for completed, index in enumerate(sorted(targets)):
                operation.update("cloud_comparison", completed=completed, total=len(targets))
                checkpoints[index] = window_result(index, {"openai": second_pass})

        passages = []
        previous_words = []
        for saved, (window, words, selection, issue) in zip(checkpoints, aligned, strict=True):
            blocks, previous_words = _windows_passages(
                saved["gemini"], saved.get("openai"), window, speech, manifest["vocabulary"], previous_words,
                word_selection=(words, selection), seam_issue=issue,
            )
            for block in blocks:
                block["clips"] = _clips(recording, job, block)
            passages.extend(blocks)
        operation.update("review_draft", completed=len(spans), total=len(spans))
        compared = [window for window, result in zip(spans, checkpoints, strict=True) if "openai" in result]
        return {
            "samples": recording.samples, "passages": passages,
            "comparison": {
                "mode": "targeted" if targeted else "full",
                "windows": len(spans), "compared_windows": len(compared),
                "audio_seconds": sum(window.request_end - window.request_start for window in compared) / SAMPLE_RATE,
            },
        }


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


def _suggest_timings(text, reference):
    """Preserve complete unchanged cues and suggest the time range of each edit."""
    source = ""
    offsets = [0]
    for cue in reference:
        source += _anchors(cue["text"])[0]
        offsets.append(len(source))
    selected, positions = _anchors(text)
    matches = SequenceMatcher(None, source, selected, autojunk=False).get_matching_blocks()
    anchors = []
    index = 0
    for original, target, size in matches:
        while index < len(reference) and offsets[index] < original + size:
            start, end = offsets[index:index + 2]
            if original <= start < end <= original + size:
                anchors.append((index, target + start - original, target + end - original))
            index += 1

    def text_cut(offset):
        if offset == 0:
            return 0
        if offset == len(positions):
            return len(text)
        return positions[offset]

    template = []
    previous_cue, previous_end = 0, 0
    for index, start, end in [*anchors, (len(reference), len(selected), len(selected))]:
        changed = text[text_cut(previous_end):text_cut(start)].strip()
        if changed:
            affected = reference[previous_cue:index]
            template.append({
                "text": changed, "needs_confirmation": True,
                "start_ms": affected[0]["start_ms"] if affected else None,
                "end_ms": affected[-1]["end_ms"] if affected else None,
            })
        if index < len(reference):
            cue_text = text[text_cut(start):text_cut(end)].strip()
            wrapped = _wrapped(cue_text)
            phrase = source[offsets[index]:offsets[index + 1]]
            template.append({
                **reference[index], "text": wrapped or cue_text,
                "needs_confirmation": wrapped is None or source.count(phrase) != selected.count(phrase),
            })
        previous_cue, previous_end = index + 1, end
    return template


def identify_issues(issues):
    return [{"id": f"issue-{index}", **issue} for index, issue in enumerate(issues, 1)]


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
    decision = decision or {}
    accepted = "choice" in decision
    choice = decision.get("choice", "suggestion")
    acknowledged = set(decision.get("resolved_issues", []))
    issues = [
        issue for issue in identify_issues(block["issues"])
        if issue["id"] not in acknowledged and (not accepted or issue["category"] != "wording")
    ]
    start = math.ceil(block["start_sample"] / _MS)
    end = math.floor(block["end_sample"] / _MS)
    original = block["text"]
    if choice == "no_speech":
        return "", [], [], []
    if choice == "inaudible":
        label = "［听不清］"
        if end - start < 250:
            raise ValueError("The passage is too short for an inaudible caption. Mark it as no speech or edit it.")
        return label, [{"text": label, "start_ms": start, "end_ms": min(end, start + 2000)}], [], []
    if choice == "edited":
        text = decision["text"].strip()
    elif choice == "suggestion":
        text = original
        if accepted and not text.strip():
            raise ValueError("Select a nonempty reading or mark the passage as no speech")
    else:
        text = block["candidates"].get(choice, "")
        if not text.strip():
            raise ValueError("That candidate contains no usable text for this passage")
    text = SIMPLIFIED.convert(text.strip())
    if not text:
        return "", [], issues, []
    if "timings" in decision:
        timings = decision["timings"]
        if (not isinstance(timings, list) or not 1 <= len(timings) <= 100
                or any(not isinstance(cue, dict) or not isinstance(cue.get("text"), str) for cue in timings)):
            raise ValueError("Provide between 1 and 100 timed subtitles")
        timings = [{**cue, "text": SIMPLIFIED.convert(cue["text"])} for cue in timings]
        validate_manual_cues(timings, text, start, end)
        return text, timings, issues, []
    basis = [{**cue, "text": SIMPLIFIED.convert(cue["text"])} for cue in decision.get("timing_basis", [])]
    if basis:
        carried = basis
        if _anchors(text)[0] == "".join(_anchors(cue["text"])[0] for cue in basis):
            carried = [{key: cue[key] for key in ("text", "start_ms", "end_ms")}
                       for cue in _suggest_timings(text, basis)]
        try:
            validate_manual_cues(carried, text, start, end)
            return text, carried, issues, []
        except (TypeError, ValueError):
            pass
    try:
        words = _project_words(original, text, block["words"])
        cues = group_timed_words(text, words, start_ms=start, end_ms=end)
        return text, cues, issues, []
    except ValueError:
        issues.append({"id": "manual-timing", **_issue(
            "timing", "manual_timing_required", "Check the affected cue boundaries against the recording.",
        )})
    if not basis:
        try:
            baseline = SIMPLIFIED.convert(original.strip())
            words = _project_words(original, baseline, block["words"])
            basis = group_timed_words(baseline, words, start_ms=start, end_ms=end)
        except ValueError:
            pass
    return text, [], issues, _suggest_timings(text, basis) if basis else []


def validate_decision(data, passage_id, choice, text, timings=None, *, previous=None, resolved_issues=None):
    block = next((row for row in data["passages"] if row["id"] == passage_id), None)
    if block is None:
        raise ValueError("This passage is not in the current job")
    previous = previous or {}
    if resolved_issues is not None:
        eligible = {issue["id"] for issue in identify_issues(block["issues"]) if issue["category"] in {"coverage", "seam"}}
        if (not isinstance(resolved_issues, list) or not resolved_issues or
                any(not isinstance(issue, str) or issue not in eligible for issue in resolved_issues)):
            raise ValueError("Select a coverage or boundary issue from this passage")
    if choice is None:
        if text is not None or timings is not None or resolved_issues is None:
            raise ValueError("Select a reading or confirm a review issue")
        decision = dict(previous)
    else:
        decision = make_decision(choice, text, timings)
        if "timings" not in decision and choice not in {"inaudible", "no_speech"}:
            if basis := previous.get("timings") or previous.get("timing_basis"):
                decision["timing_basis"] = basis
    acknowledged = set(previous.get("resolved_issues", [])) | set(resolved_issues or [])
    if acknowledged:
        decision["resolved_issues"] = sorted(acknowledged)
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
        text, timings, issues, template = _resolve(block, decision)
        entry = {
            "id": block["id"], "start_seconds": block["start_sample"] / SAMPLE_RATE,
            "end_seconds": block["end_sample"] / SAMPLE_RATE, "selected_text": text,
            "suggested_text": SIMPLIFIED.convert(block["text"]), "candidates": block["candidates"],
            "candidate_spans": block["candidate_spans"], "human_reviewed": decision is not None,
            "timings": timings, "timing_template": template, "clips": block["clips"], "issues": issues,
            "unplaced_openai": block.get("unplaced_openai"),
            "comparison_skipped": block.get("comparison_skipped", False),
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
    if "comparison" in data:
        summary["comparison"] = data["comparison"]
    for name, value in (("passages", summaries), ("review", pending), ("summary", summary), ("decisions", decisions)):
        _write_json(destination / f"{name}.json", value)
    return summary
