"""Compare published Chinese subtitles to a human transcript and timed cues offline."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
import json
import math
from pathlib import Path
import re
import unicodedata

import opencc

_SIMPLIFIED = opencc.OpenCC("t2s.json")
_MAX_CER = 0.02
_MAX_BOUNDARY_P95_MS = 750
_MIN_CUE_MATCH = 0.8
_MAX_GROUP_CUES = 3
_SRT_TIME = r"([0-9]{2,}):([0-5][0-9]):([0-5][0-9]),([0-9]{3})"
_SRT_TIMING = re.compile(rf"{_SRT_TIME}[ \t]+-->[ \t]+{_SRT_TIME}")


@dataclass(frozen=True)
class TimedText:
    id: int
    start_ms: int
    end_ms: int
    text: str

    def position(self):
        return {"id": self.id, "start_ms": self.start_ms, "end_ms": self.end_ms}


def normalize_text(text):
    """Ignore layout and sentence punctuation while retaining spoken-number marks."""
    text = _SIMPLIFIED.convert(unicodedata.normalize("NFKC", text)).casefold()
    output = []
    for index, char in enumerate(text):
        left, right = text[index - 1] if index else "", text[index + 1:index + 2]
        if unicodedata.category(char)[0] in {"L", "M", "N"}:
            output.append(char)
        elif char in "+-−" and right.isdecimal():
            output.append("-" if char == "−" else char)
        elif char == "." and right.isdecimal() and (
            left.isdecimal() or not left or left.isspace() or left in "+-−(（"
        ):
            output.append(char)
        elif char == "/" and left.isdecimal() and right.isdecimal():
            output.append(char)
        elif char in "%‰" and left.isdecimal():
            output.append(char)
    return "".join(output)


def _milliseconds(parts):
    hours, minutes, seconds, millis = map(int, parts)
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def _timed_text(path):
    cues = []
    previous_number, previous_start = 0, -1
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for nonblank, group in groupby(lines, key=lambda line: bool(line.strip())):
        if not nonblank:
            continue
        block = list(group)
        if len(block) < 3 or not re.fullmatch(r"[0-9]{1,15}", block[0]):
            raise ValueError(f"Invalid number or text in SRT block {len(cues) + 1}.")
        number = int(block[0])
        timing = _SRT_TIMING.fullmatch(block[1])
        if number <= previous_number or timing is None:
            raise ValueError(f"Invalid number or timestamp in SRT cue {number}.")
        start = _milliseconds(timing.groups()[:4])
        end = _milliseconds(timing.groups()[4:])
        if start < previous_start or end <= start:
            raise ValueError(f"Invalid time range in SRT cue {number}.")
        cues.append(TimedText(number, start, end, normalize_text("\n".join(block[2:]))))
        previous_number, previous_start = number, start
    if not cues:
        raise ValueError("The SRT contains no subtitle cues.")
    return cues


def _percentile(values, percentile):
    if not values:
        return None
    return sorted(values)[math.ceil(len(values) * percentile) - 1]


def _cue_offsets(cues):
    offsets = [0]
    for cue in cues:
        offsets.append(offsets[-1] + len(cue.text))
    return offsets


def _timing_groups(reference, candidate, equal):
    # Compare a boundary only when both sides place it at the same position
    # in the aligned transcript. Bound groups so a wholesale reflow stays visible.
    reference_offsets = _cue_offsets(reference)
    starts = {offset: index for index, offset in enumerate(reference_offsets[:-1]) if reference[index].text}
    ends = {offset: index for index, offset in enumerate(reference_offsets[1:], 1) if reference[index - 1].text}
    candidate_offsets = _cue_offsets(candidate)
    other_starts = {offset: index for index, offset in enumerate(candidate_offsets[:-1]) if candidate[index].text}
    other_ends = {offset: index for index, offset in enumerate(candidate_offsets[1:], 1) if candidate[index - 1].text}
    for block in equal:
        shift = block.dest_start - block.src_start
        low = bisect_left(reference_offsets, block.src_start)
        high = bisect_right(reference_offsets, block.src_end)
        shared = sorted({offset for offset in reference_offsets[low:high] if offset + shift in other_starts
                         or offset + shift in other_ends})
        for start, end in zip(shared, shared[1:]):
            first, last = starts.get(start), ends.get(end)
            begin, finish = other_starts.get(start + shift), other_ends.get(end + shift)
            if (None not in (first, last, begin, finish)
                    and 0 < last - first <= _MAX_GROUP_CUES and 0 < finish - begin <= _MAX_GROUP_CUES):
                yield reference[first:last], candidate[begin:finish]


def _duration(cues):
    total = end = 0
    for cue in sorted(cues, key=lambda item: item.start_ms):
        if cue.end_ms > end:
            total += cue.end_ms - max(cue.start_ms, end)
            end = cue.end_ms
    return total


def _timing_metrics(reference, candidate, equal):
    matches = list(_timing_groups(reference, candidate, equal))
    matched_reference = {cue.id for gold, _ in matches for cue in gold}
    matched_candidate = {cue.id for _, actual in matches for cue in actual}
    starts = [abs(gold[0].start_ms - actual[0].start_ms) for gold, actual in matches]
    ends = [abs(gold[-1].end_ms - actual[-1].end_ms) for gold, actual in matches]
    reference_duration, candidate_duration = _duration(reference), _duration(candidate)
    union_duration = _duration([*reference, *candidate])
    return {
        "method": "Monotonic normalized text alignment with shared cue boundaries",
        "matched_cues": len(matched_reference),
        "matched_groups": len(matches),
        "matched_candidate_cues": len(matched_candidate),
        "reference_cues": len(reference),
        "candidate_cues": len(candidate),
        "reference_match_fraction": len(matched_reference) / len(reference),
        "candidate_match_fraction": len(matched_candidate) / len(candidate),
        "reference_duration_match_fraction": _duration([cue for gold, _ in matches for cue in gold]) / reference_duration,
        "candidate_duration_match_fraction": _duration([cue for _, actual in matches for cue in actual]) / candidate_duration,
        "reference_cue_ms_without_candidate": union_duration - candidate_duration,
        "candidate_cue_ms_without_reference": union_duration - reference_duration,
        "start_error_p50_ms": _percentile(starts, 0.50),
        "end_error_p50_ms": _percentile(ends, 0.50),
        "start_error_p95_ms": _percentile(starts, 0.95),
        "end_error_p95_ms": _percentile(ends, 0.95),
        "unmatched_reference_cues": [cue.position() for cue in reference if cue.id not in matched_reference],
        "unmatched_candidate_cues": [cue.position() for cue in candidate if cue.id not in matched_candidate],
    }


def _term_errors(terms, reference_text, candidate_text, reference_cues, candidate_cues):
    normalized = {normalize_text(term): term for term in terms}
    errors = []
    for key, term in normalized.items():
        if not key:
            continue
        expected, actual = reference_text.count(key), candidate_text.count(key)
        if expected != actual:
            errors.append({
                "term": term, "reference_count": expected, "candidate_count": actual,
                "reference_cues": [cue.position() for cue in reference_cues if key in cue.text],
                "candidate_cues": [cue.position() for cue in candidate_cues if key in cue.text],
            })
    return errors


def _occurrences(term, text, cues, offsets):
    index = text.find(term)
    while index >= 0:
        end = index + len(term)
        first = cues[bisect_right(offsets, index) - 1]
        last = cues[bisect_left(offsets, end) - 1]
        yield index, {
            "first_cue": first.id, "last_cue": last.id,
            "start_ms": first.start_ms, "end_ms": last.end_ms,
        }
        index = text.find(term, end)


def _term_occurrences(terms, reference, candidate, ref_text, cand_text, equal):
    reference_offsets, candidate_offsets = map(_cue_offsets, (reference, candidate))
    block_starts = [block.src_start for block in equal]
    results = []
    for key, term in {normalize_text(term): term for term in terms}.items():
        if not key:
            continue
        expected = list(_occurrences(key, ref_text, reference, reference_offsets))
        actual = dict(_occurrences(key, cand_text, candidate, candidate_offsets))
        missed = []
        for start, location in expected:
            position = bisect_right(block_starts, start) - 1
            block = equal[position] if position >= 0 else None
            if block is not None and start + len(key) <= block.src_end:
                candidate_start = block.dest_start + start - block.src_start
                match = actual.get(candidate_start)
                if (match and match["start_ms"] <= location["end_ms"] + _MAX_BOUNDARY_P95_MS
                        and location["start_ms"] <= match["end_ms"] + _MAX_BOUNDARY_P95_MS):
                    actual.pop(candidate_start)
                    continue
            missed.append(location)
        results.append({
            "term": term, "matched": len(expected) - len(missed),
            "missed": missed, "extra": list(actual.values()),
        })
    matched = sum(row["matched"] for row in results)
    missed = sum(len(row["missed"]) for row in results)
    extra = sum(len(row["extra"]) for row in results)
    return {"matched": matched, "missed": missed, "extra": extra,
            "precision": matched / (matched + extra) if matched + extra else None,
            "recall": matched / (matched + missed) if matched + missed else None, "by_term": results}


def evaluate(reference_txt, reference_srt, candidate_txt, candidate_srt, terms=(), *,
             audio_minutes=None, review_minutes=None):
    from rapidfuzz.distance import Levenshtein

    reference_text = normalize_text(reference_txt.read_text("utf-8-sig"))
    candidate_text = normalize_text(candidate_txt.read_text("utf-8-sig"))
    if not reference_text:
        raise ValueError("The reference transcript contains no scorable characters.")
    changes = Counter(edit.tag for edit in Levenshtein.editops(reference_text, candidate_text))
    cer = changes.total() / len(reference_text)
    if (reference_srt is None) != (candidate_srt is None):
        raise ValueError("Provide both sets of subtitles for a timing comparison")
    reference_cues = _timed_text(reference_srt) if reference_srt is not None else []
    candidate_cues = _timed_text(candidate_srt) if candidate_srt is not None else []
    term_errors = _term_errors(terms, reference_text, candidate_text, reference_cues, candidate_cues)
    checks = {"cer": cer <= _MAX_CER, "term_counts": not term_errors}
    timing = term_locations = None
    if reference_cues:
        reference_caption_text = "".join(cue.text for cue in reference_cues)
        candidate_caption_text = "".join(cue.text for cue in candidate_cues)
        equal = [block for block in Levenshtein.opcodes(reference_caption_text, candidate_caption_text)
                 if block.tag == "equal"]
        timing = _timing_metrics(reference_cues, candidate_cues, equal)
        term_locations = _term_occurrences(
            terms, reference_cues, candidate_cues, reference_caption_text, candidate_caption_text, equal,
        )
        checks.update(
            cue_coverage=min(timing["reference_match_fraction"], timing["candidate_match_fraction"]) >= _MIN_CUE_MATCH,
            timing_p95=all(value is not None and value <= _MAX_BOUNDARY_P95_MS
                           for value in (timing["start_error_p95_ms"], timing["end_error_p95_ms"])),
            reference_txt_matches_srt=reference_text == reference_caption_text,
            candidate_txt_matches_srt=candidate_text == candidate_caption_text,
            term_occurrences=term_locations["missed"] == term_locations["extra"] == 0,
        )
    effort = None
    if audio_minutes is not None or review_minutes is not None:
        if (audio_minutes is None or review_minutes is None or not math.isfinite(audio_minutes)
                or not math.isfinite(review_minutes) or audio_minutes <= 0 or review_minutes < 0):
            raise ValueError("Provide a positive audio length and a nonnegative review time in minutes")
        effort = {"audio_minutes": audio_minutes, "review_minutes": review_minutes,
                  "minutes_per_audio_hour": 60 * review_minutes / audio_minutes}
    return {
        "character_error_rate": cer,
        "reference_characters": len(reference_text),
        "candidate_characters": len(candidate_text),
        "character_errors": {
            "insertions": changes["insert"], "deletions": changes["delete"], "substitutions": changes["replace"],
        },
        "timing": timing,
        "term_count_errors": term_errors,
        "term_occurrences": term_locations,
        "review_effort": effort,
        "limits": {"cer": _MAX_CER, "p95_ms": _MAX_BOUNDARY_P95_MS, "minimum_cue_match_fraction": _MIN_CUE_MATCH},
        "checks": checks,
        "passes_checks": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Use TXT and SRT from the same interval of the same recording. "
        "Timing scores cover shared boundaries in aligned text, including small groups of split or merged cues.",
    )
    for name in ("reference-txt", "candidate-txt"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--reference-srt", type=Path)
    parser.add_argument("--candidate-srt", type=Path)
    parser.add_argument("--text-only", action="store_true", help="score a raw transcript before cue timing")
    parser.add_argument("--terms", type=Path, help="UTF-8 glossary, one expected term per line")
    parser.add_argument("--audio-minutes", type=float, help="duration of the scored audio")
    parser.add_argument("--review-minutes", type=float, help="active time spent reviewing the candidate")
    parser.add_argument("--check", action="store_true", help="Exit 1 when any acceptance check fails")
    options = parser.parse_args(argv)
    if not options.text_only and (options.reference_srt is None or options.candidate_srt is None):
        parser.error("Provide both SRT files or use --text-only for a raw transcript.")
    try:
        terms = options.terms.read_text("utf-8-sig").splitlines() if options.terms else ()
        report = evaluate(
            options.reference_txt, None if options.text_only else options.reference_srt,
            options.candidate_txt, None if options.text_only else options.candidate_srt, terms,
            audio_minutes=options.audio_minutes, review_minutes=options.review_minutes,
        )
    except ModuleNotFoundError as error:
        if error.name != "rapidfuzz":
            raise
        parser.error("Install requirements-evaluation.txt to run the evaluator.")
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(options.check and not report["passes_checks"])


if __name__ == "__main__":
    raise SystemExit(main())
