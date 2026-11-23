"""Build and validate Chinese subtitle cues without a local model runtime."""
from __future__ import annotations

import re
import unicodedata


MAX_LINE_WIDTH = 42
MAX_CUE_MS = 8_000
_SENTENCE_END = "。！？!?；;"
_OPENING_MARKS = "（〔【《〈「『“‘("
_CLOSING_MARKS = "）〕】》〉」』”’)"


def _timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def _narrow_letter_or_number(char: str) -> bool:
    return char.isalnum() and unicodedata.east_asian_width(char) not in {"W", "F"}


def _inside_word(text: str, index: int) -> bool:
    left, right = text[index - 1], text[index]
    return (
        (_narrow_letter_or_number(left) and _narrow_letter_or_number(right))
        or unicodedata.category(right).startswith("M")
        or "\u200d" in (left, right)
    )


def _text_signature(text: str) -> str:
    # Chinese line breaking can change spacing. A space between Latin words is
    # meaningful and must survive both wrapping and cue boundaries.
    parts = re.findall(r"\S+", text)
    if not parts:
        return ""
    signature = [parts[0]]
    for left, right in zip(parts, parts[1:]):
        if _narrow_letter_or_number(left[-1]) and _narrow_letter_or_number(right[0]):
            signature.append(" ")
        signature.append(right)
    return "".join(signature)


def _readable_length(text: str, duration: int) -> bool:
    return 0 < duration <= MAX_CUE_MS and (duration >= 200 or sum(char.isalnum() for char in text) < 3)


def validate_manual_cues(cues: list[dict], expected_text: str, start_ms: int, end_ms: int) -> None:
    """Validate cue bounds, readability and the complete selected wording."""
    if type(start_ms) is not int or type(end_ms) is not int or not 0 <= start_ms <= end_ms:
        raise ValueError("Invalid subtitle time range")
    if not isinstance(cues, list) or not isinstance(expected_text, str):
        raise ValueError("Expected a cue list and transcript text")

    previous_end = start_ms
    for number, cue in enumerate(cues, 1):
        if not isinstance(cue, dict) or set(cue) != {"text", "start_ms", "end_ms"}:
            raise ValueError(f"Invalid subtitle cue {number}")
        start, end, text = cue["start_ms"], cue["end_ms"], cue["text"]
        if type(start) is not int or type(end) is not int or not previous_end <= start < end <= end_ms:
            raise ValueError(f"Cue {number} overlaps another cue or lies outside its time range")
        if not isinstance(text, str) or any(unicodedata.category(char) == "Cc" and char != "\n" for char in text):
            raise ValueError(f"Cue {number} contains invalid text")
        lines = text.split("\n")
        if not 1 <= len(lines) <= 2 or any(
            not line.strip() or _display_width(line) > MAX_LINE_WIDTH for line in lines
        ):
            raise ValueError(f"Cue {number} must use one or two lines of at most {MAX_LINE_WIDTH} columns")
        if not _readable_length(text, end - start):
            raise ValueError(f"Cue {number} has an unreadable duration")
        previous_end = end

    actual = "\n".join(cue["text"] for cue in cues)
    if _text_signature(actual) != _text_signature(expected_text):
        raise ValueError("Subtitle cues change the selected wording or punctuation")


def render_srt(cues: list[dict], duration_ms: int, expected_text: str) -> str:
    """Return validated SRT text so the caller can publish it atomically."""
    validate_manual_cues(cues, expected_text, 0, duration_ms)
    return "".join(
        f"{number}\n{_timestamp(cue['start_ms'])} --> {_timestamp(cue['end_ms'])}\n{cue['text']}\n\n"
        for number, cue in enumerate(cues, 1)
    )


def _wrapped(text: str) -> str | None:
    text = " ".join(text.split())
    width = _display_width(text)
    if not width or width > 2 * MAX_LINE_WIDTH + 1:
        return None
    if width <= MAX_LINE_WIDTH:
        return text

    cuts = []
    for index in range(1, len(text)):
        if _inside_word(text, index):
            continue
        left, right = text[:index].rstrip(), text[index:].lstrip()
        left_width, right_width = _display_width(left), _display_width(right)
        if max(left_width, right_width) <= MAX_LINE_WIDTH:
            if _text_signature(left + "\n" + right) != _text_signature(text):
                continue
            bad_edge = left[-1] in _OPENING_MARKS or right[0] in "，。；、：？！,.!?;" + _CLOSING_MARKS
            cuts.append((bad_edge, abs(left_width - right_width), index))
    if not cuts:
        return None
    index = min(cuts)[2]
    return text[:index].rstrip() + "\n" + text[index:].lstrip()


def _word_boundaries(text: str, words: list[dict], start_ms: int, end_ms: int) -> list[int]:
    boundaries = [0]
    previous_end = 0
    previous_start_ms = previous_end_ms = start_ms
    for word in words:
        left, right = word["text_start"], word["text_end"]
        begin, finish = word["start_ms"], word["end_ms"]
        if (
            type(left) is not int or type(right) is not int
            or not previous_end <= left < right <= len(text)
            or word["text"] != text[left:right]
            or type(begin) is not int or type(finish) is not int
            or not previous_start_ms <= begin <= finish <= end_ms
            or finish < previous_end_ms
        ):
            raise ValueError("The recording and text ranges of a timed word do not align")
        gap = text[previous_end:left]
        if any(unicodedata.category(char)[0] in {"L", "M", "N"} for char in gap):
            raise ValueError("Some of the selected text has no word timing")
        if previous_end:
            # Keep an opening mark with the following word, and punctuation
            # that closes the preceding phrase with that phrase.
            first_opening = next((index for index, char in enumerate(gap) if char in _OPENING_MARKS), len(gap))
            boundaries.append(previous_end + first_opening)
        previous_end = right
        previous_start_ms, previous_end_ms = begin, finish

    if any(unicodedata.category(char)[0] in {"L", "M", "N"} for char in text[previous_end:]):
        raise ValueError("Some of the selected text has no word timing")
    boundaries.append(len(text))
    return boundaries


def group_timed_words(text: str, words: list[dict], *, start_ms: int, end_ms: int) -> list[dict]:
    """Group already anchored words into cue text and recording-relative milliseconds."""
    if not text.strip():
        return []
    if not words:
        raise ValueError("Selected text has no word timing")
    boundaries = _word_boundaries(text, words, start_ms, end_ms)
    cues = []
    start = 0
    while start < len(words):
        begin = words[start]["start_ms"]
        best = None
        for last in range(start, len(words)):
            finish = words[last]["end_ms"]
            if finish - begin > MAX_CUE_MS:
                break
            boundary = boundaries[last + 1]
            raw = text[boundaries[start]:boundary].strip()
            wrapped = _wrapped(raw)
            if wrapped is None:
                if _display_width(" ".join(raw.split())) > 2 * MAX_LINE_WIDTH + 1:
                    break
                continue
            next_start = words[last + 1]["start_ms"] if last + 1 < len(words) else end_ms
            if finish > next_start or not _readable_length(raw, finish - begin):
                continue
            if boundary < len(text) and _inside_word(text, boundary):
                continue
            best = (last, {"start_ms": begin, "end_ms": finish, "text": wrapped})
            phrase_end = raw.rstrip(_CLOSING_MARKS).endswith(tuple(_SENTENCE_END))
            if finish - begin >= 750 and (phrase_end or next_start - finish >= 350):
                break
        if best is None:
            raise ValueError("Word timing could not be grouped into readable nonoverlapping cues")
        last, cue = best
        cues.append(cue)
        start = last + 1
    validate_manual_cues(cues, text, start_ms, end_ms)
    return cues
