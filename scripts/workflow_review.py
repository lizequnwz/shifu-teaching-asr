"""Find material disagreements between two ASR readings."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import re
import unicodedata

import opencc


_SIMPLIFIED = opencc.OpenCC("t2s.json")
_FILLERS = frozenset("啊嗯呃哦唔哎呀嘛欸")
_PHRASE_PADDING = "啊嗯呃哦唔哎呀嘛欸哈哼 \t\r\n"
_SEPARATORS = frozenset("，。！？；：、,.!?;:")
_NEGATIVES = frozenset({"不", "没", "未", "无", "非", "勿", "莫", "别", "no", "not", "never"})
_OPPOSITES = {
    frozenset(pair) for pair in (
        "上下", "对错", "真假", "好坏", "多少", "生死", "开关", "前后", "左右",
        "杀救", "攻守", "抓放", "增减", "升降", "进退", "赢输", "买卖",
    )
}
_BODY_PARTS = frozenset("肺肝肾心胃胰脾胆肠脑口乳骨甲咽喉")
_DIAGNOSIS_SUFFIXES = frozenset("癌瘤炎病")
_DIGITS = "〇零一二两三四五六七八九"
_TO_DIGITS = str.maketrans(_DIGITS, "001223456789")
_SMALL_UNITS = {"千": 1000, "百": 100, "十": 10}
_LARGE_UNITS = {"兆": 10**12, "亿": 10**8, "万": 10**4}
_MEASURES = frozenset("岁年月日时分秒周天里米克斤吨元度倍号次页届期章节品位人本层组")
_NUMBER = re.compile(
    rf"(?:百分之)?(?:负|零下|[-−+])?(?:[0-9]{{1,3}}(?:,[0-9]{{3}})+|[0-9{_DIGITS}十百千万亿兆]+)"
    rf"(?:[.点][0-9{_DIGITS}]+[十百千万亿兆]*)?%?"
)
_LATIN = re.compile(r"[a-z]+")
_REPEATED_NO = re.compile(r"(?:no){2,}")


@dataclass(frozen=True)
class _Token:
    key: str
    start: int
    end: int
    weight: int
    number: bool = False
    material_number: bool = False
    negative: bool = False


def _simplify_with_offsets(text: str) -> tuple[str, list[tuple[int, int]]]:
    parts = [unicodedata.normalize("NFKC", char).casefold() for char in text]
    offsets = [(index, index + 1) for index, part in enumerate(parts) for _ in part]
    wide_normalized = "".join(parts)
    simplified = _SIMPLIFIED.convert(wide_normalized)
    if len(simplified) == len(offsets):
        return simplified, offsets

    # A phrase conversion can change length. Its converted characters all point
    # to the original phrase, keeping clip selection conservative.
    converted_offsets = []
    matcher = SequenceMatcher(None, wide_normalized, simplified, autojunk=False)
    for _, start, end, new_start, new_end in matcher.get_opcodes():
        if end - start == new_end - new_start:
            converted_offsets.extend(offsets[start:end])
        elif new_end > new_start:
            left = offsets[start][0] if start < len(offsets) else len(text)
            right = offsets[end - 1][1] if end > start else left
            converted_offsets.extend([(left, right)] * (new_end - new_start))
    return simplified, converted_offsets


def _number_value(number: str) -> Decimal:
    for unit, multiplier in _LARGE_UNITS.items():
        if unit in number:
            if number.count(unit) > 1:
                raise InvalidOperation
            head, tail = number.split(unit, 1)
            return _number_value(head or "1") * multiplier + _number_value(tail or "0")
    total = Decimal(0)
    digits = ""
    previous_unit = 10_000
    for char in number:
        if char in _SMALL_UNITS:
            unit = _SMALL_UNITS[char]
            if unit >= previous_unit:
                raise InvalidOperation
            total += Decimal(digits or "1") * unit
            digits = ""
            previous_unit = unit
        else:
            digits += char
    return total + Decimal(digits or "0")


def _number_key(text: str) -> str:
    number = text.removeprefix("百分之").removesuffix("%")
    sign = ""
    for prefix in ("零下", "负", "−", "-", "+"):
        if number.startswith(prefix):
            sign = "-" if prefix != "+" else ""
            number = number.removeprefix(prefix)
            break
    # 三五千 and 两三百 usually describe ranges. Avoid turning them into 35,000 and 2,300.
    possible_range = bool(re.search(r"[一二两三四五六七八九]{2,}[十百千万亿兆]", number))
    number = number.translate(_TO_DIGITS).replace(",", "").replace("点", ".")
    has_unit = any(char in _SMALL_UNITS or char in _LARGE_UNITS for char in number)
    # Keep ambiguous spoken shorthand such as 一万六 distinct from an exact value.
    if has_unit and not possible_range and not re.search(r"[百千万亿兆][1-9]$", number):
        try:
            number = format(_number_value(number), "f").removesuffix(".0")
        except InvalidOperation:
            pass
    suffix = "%" if text.startswith("百分之") or text.endswith("%") else ""
    return "#" + sign + number + suffix


def _tokens(text: str) -> list[_Token]:
    simplified, offsets = _simplify_with_offsets(text)
    tokens = []
    index = 0
    while index < len(simplified):
        char = simplified[index]
        match = _NUMBER.match(simplified, index) or _LATIN.match(simplified, index)
        if char in "-−+" and index and simplified[index - 1] in "0123456789" + _DIGITS:
            # A dash inside a numeric range is a separator, not a negative sign.
            match = None
        end = match.end() if match else index + 1
        word = simplified[index:end]
        next_char = simplified[end:end + 1]
        if _REPEATED_NO.fullmatch(word):
            for start in range(index, end, 2):
                tokens.append(_Token("no", offsets[start][0], offsets[start + 1][1], 2, negative=True))
        elif (char.isalnum() or match) and char not in _FILLERS and not (
            char in "哈哼" and (not next_char or not next_char.isalnum())
        ):
            is_number = bool(_NUMBER.fullmatch(word))
            if word == "千万" and next_char in {"不", "别", "要"}:
                is_number = False
            material_number = is_number and (
                len(word) > 1 or word.isascii() or next_char in _MEASURES
            )
            material_number |= word == "上" and next_char in _SMALL_UNITS.keys() | _LARGE_UNITS.keys()
            negative = word in _NEGATIVES and not (word == "非" and next_char in {"常", "洲"})
            tokens.append(_Token(
                _number_key(word) if is_number else word,
                offsets[index][0], offsets[end - 1][1], len(word),
                is_number, material_number, negative,
            ))
        index = end
    return tokens


def _matcher(left: list[_Token], right: list[_Token]) -> SequenceMatcher:
    return SequenceMatcher(None, [t.key for t in left], [t.key for t in right], autojunk=False)


def _source_span(tokens: list[_Token], start: int, end: int) -> list[int]:
    if end > start:
        return [tokens[start].start, tokens[end - 1].end]
    boundary = tokens[start].start if start < len(tokens) else tokens[-1].end if tokens else 0
    return [boundary, boundary]


def _reason(left: list[_Token], right: list[_Token], missing: bool) -> str | None:
    if missing:
        return "missing_speech"
    if [t.key for t in left if t.negative] != [t.key for t in right if t.negative]:
        return "negation"
    if any(t.material_number for t in (*left, *right)) or (
        any(t.number for t in left) and any(t.number for t in right)
    ):
        return "numeric"
    if len(left) == len(right) == 1 and frozenset((left[0].key, right[0].key)) in _OPPOSITES:
        return "opposite_wording"
    left_width, right_width = (sum(t.weight for t in side) for side in (left, right))
    if max(left_width, right_width) >= 3 or min(left_width, right_width) >= 2:
        if not left:
            return "additional_right_text"
        if not right:
            return "additional_left_text"
        return "substantial_replacement"
    return None


def _standalone_phrase(text: str, span: list[int]) -> bool:
    start, end = span
    before = text[:start].rstrip(_PHRASE_PADDING)
    after = text[end:].lstrip(_PHRASE_PADDING)
    return (not before or before[-1] in _SEPARATORS) and (not after or after[0] in _SEPARATORS)


def _disagreements(
    left: list[_Token], right: list[_Token], matcher: SequenceMatcher, left_text: str, right_text: str,
) -> list[dict]:
    differences = []
    for operation, i, j, k, l in matcher.get_opcodes():
        if operation == "equal":
            continue
        reason = _reason(left[i:j], right[k:l], not left or not right)
        left_span = _source_span(left, i, j)
        right_span = _source_span(right, k, l)
        if (
            reason is None and j - i == l - k == 1 and j < len(left) and l < len(right)
            and left[i].key in _BODY_PARTS and right[k].key in _BODY_PARTS
            and left[j].key == right[l].key and left[j].key in _DIAGNOSIS_SUFFIXES
        ):
            reason = "different_diagnosis_term"
        # Even a two-character utterance can add meaning when it is set off
        # as its own phrase. Keep small in-sentence particles low-risk.
        if reason is None and i == j and l - k >= 2 and _standalone_phrase(right_text, right_span):
            reason = "additional_right_text"
        if reason is None and k == l and j - i >= 2 and _standalone_phrase(left_text, left_span):
            reason = "additional_left_text"
        differences.append({
            "reason": reason,
            "left_span": left_span,
            "right_span": right_span,
        })
    return differences


def _describe(item: dict, left: str, right: str) -> dict:
    return {
        "risk": "high" if item["reason"] else "low",
        "reason": item["reason"] or "small_wording_difference",
        "left_span": item["left_span"],
        "right_span": item["right_span"],
        "left_text": left[slice(*item["left_span"])],
        "right_text": right[slice(*item["right_span"])],
    }


def compare(left: str, right: str) -> dict:
    """Classify differences and report half-open offsets into the original texts.

    Similarity is diagnostic only. One changed number or negative can be material
    even when almost the entire clip agrees.
    """
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    matcher = _matcher(left_tokens, right_tokens)
    differences = [
        _describe(item, left, right) for item in _disagreements(left_tokens, right_tokens, matcher, left, right)
    ]
    reasons = list(dict.fromkeys(item["reason"] for item in differences if item["risk"] == "high"))
    return {
        "risk": "high" if reasons else "low",
        "reason": ", ".join(reasons) if reasons else "small_wording_difference" if differences else "normalized_match",
        "similarity": matcher.ratio(),
        "differences": differences,
    }
