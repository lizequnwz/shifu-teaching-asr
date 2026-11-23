"""Compare published Chinese subtitles to a human transcript and timed cues offline."""
from __future__ import annotations

import argparse
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


def _timing_metrics(reference, candidate):
    reference_counts = Counter(cue.text for cue in reference)
    candidate_counts = Counter(cue.text for cue in candidate)
    unique_candidates = {cue.text: cue for cue in candidate if cue.text and candidate_counts[cue.text] == 1}
    matches = [
        (cue, unique_candidates[cue.text])
        for cue in reference if reference_counts[cue.text] == 1 and cue.text in unique_candidates
    ]
    matched_reference = {gold.id for gold, _ in matches}
    matched_candidate = {actual.id for _, actual in matches}
    starts = [abs(gold.start_ms - actual.start_ms) for gold, actual in matches]
    ends = [abs(gold.end_ms - actual.end_ms) for gold, actual in matches]
    return {
        "method": "Exact normalized cue text occurring once in each SRT",
        "matched_cues": len(matches),
        "reference_cues": len(reference),
        "candidate_cues": len(candidate),
        "reference_match_fraction": len(matches) / len(reference),
        "candidate_match_fraction": len(matches) / len(candidate),
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


def evaluate(reference_txt, reference_srt, candidate_txt, candidate_srt, terms=()):
    from rapidfuzz.distance import Levenshtein

    reference_text = normalize_text(reference_txt.read_text("utf-8-sig"))
    candidate_text = normalize_text(candidate_txt.read_text("utf-8-sig"))
    if not reference_text:
        raise ValueError("The reference transcript contains no scorable characters.")
    reference_cues = _timed_text(reference_srt)
    candidate_cues = _timed_text(candidate_srt)
    changes = Counter(edit.tag for edit in Levenshtein.editops(reference_text, candidate_text))
    cer = changes.total() / len(reference_text)
    timing = _timing_metrics(reference_cues, candidate_cues)
    term_errors = _term_errors(terms, reference_text, candidate_text, reference_cues, candidate_cues)
    checks = {
        "cer": cer <= _MAX_CER,
        "cue_coverage": min(timing["reference_match_fraction"], timing["candidate_match_fraction"]) >= _MIN_CUE_MATCH,
        "timing_p95": all(
            value is not None and value <= _MAX_BOUNDARY_P95_MS
            for value in (timing["start_error_p95_ms"], timing["end_error_p95_ms"])
        ),
        "reference_txt_matches_srt": reference_text == "".join(cue.text for cue in reference_cues),
        "candidate_txt_matches_srt": candidate_text == "".join(cue.text for cue in candidate_cues),
        "term_counts": not term_errors,
    }
    return {
        "character_error_rate": cer,
        "reference_characters": len(reference_text),
        "candidate_characters": len(candidate_text),
        "character_errors": {
            "insertions": changes["insert"], "deletions": changes["delete"], "substitutions": changes["replace"],
        },
        "timing": timing,
        "term_count_errors": term_errors,
        "limits": {"cer": _MAX_CER, "p95_ms": _MAX_BOUNDARY_P95_MS, "minimum_cue_match_fraction": _MIN_CUE_MATCH},
        "checks": checks,
        "passes_checks": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Use TXT and SRT from the same interval of the same recording. "
        "Repeated and split or merged cues are reported as unmatched. The timing score covers only unique cue matches.",
    )
    for name in ("reference-txt", "reference-srt", "candidate-txt", "candidate-srt"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--terms", type=Path, help="UTF-8 glossary, one expected term per line")
    parser.add_argument("--check", action="store_true", help="Exit 1 when any acceptance check fails")
    options = parser.parse_args(argv)
    try:
        terms = options.terms.read_text("utf-8-sig").splitlines() if options.terms else ()
        report = evaluate(options.reference_txt, options.reference_srt, options.candidate_txt, options.candidate_srt, terms)
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
