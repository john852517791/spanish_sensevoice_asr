#!/usr/bin/env python3
"""Spanish transcript normalization for ASR tokenizer training."""

from __future__ import annotations

import re
import unicodedata

_ALLOWED_LETTERS = "a-záéíóúüñ"
_PUNCT_TO_SPACE = re.compile(r"[\t\r\n\"“”‘’`´.,;:!?¿¡()\[\]{}<>/\\|@#$%^&*_+=~·…•、。！，；：（）《》「」『』]")
_DASHES = re.compile(r"[-‐‑‒–—―]")
_SPACES = re.compile(r"\s+")
_DISALLOWED = re.compile(fr"[^ {_ALLOWED_LETTERS}]")
_NUMERIC_DATE = re.compile(r"\b(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})\b")
_NUMERIC_VALUE = r"\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:[,.]\d+)?"
_PERCENT = re.compile(fr"\b({_NUMERIC_VALUE})\s*%")
_EURO = re.compile(fr"\b({_NUMERIC_VALUE})\s*€")
_NUMBER = re.compile(fr"\b{_NUMERIC_VALUE}\b")

_MONTHS = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}

_UNITS = {
    0: "cero",
    1: "uno",
    2: "dos",
    3: "tres",
    4: "cuatro",
    5: "cinco",
    6: "seis",
    7: "siete",
    8: "ocho",
    9: "nueve",
    10: "diez",
    11: "once",
    12: "doce",
    13: "trece",
    14: "catorce",
    15: "quince",
    16: "dieciséis",
    17: "diecisiete",
    18: "dieciocho",
    19: "diecinueve",
    20: "veinte",
    21: "veintiuno",
    22: "veintidós",
    23: "veintitrés",
    24: "veinticuatro",
    25: "veinticinco",
    26: "veintiséis",
    27: "veintisiete",
    28: "veintiocho",
    29: "veintinueve",
}
_TENS = {
    30: "treinta",
    40: "cuarenta",
    50: "cincuenta",
    60: "sesenta",
    70: "setenta",
    80: "ochenta",
    90: "noventa",
}
_HUNDREDS = {
    100: "cien",
    200: "doscientos",
    300: "trescientos",
    400: "cuatrocientos",
    500: "quinientos",
    600: "seiscientos",
    700: "setecientos",
    800: "ochocientos",
    900: "novecientos",
}


def _apocopate_number_words(words: str) -> str:
    words = re.sub(r"\bveintiuno\b", "veintiún", words)
    words = re.sub(r"\buno\b", "un", words)
    return words


def _int_to_spanish(number: int, *, apocopate: bool = False) -> str:
    if number < 0:
        return "menos " + _int_to_spanish(abs(number), apocopate=apocopate)
    if number < 30:
        words = _UNITS[number]
    elif number < 100:
        ten = number // 10 * 10
        unit = number % 10
        words = _TENS[ten] if unit == 0 else f"{_TENS[ten]} y {_UNITS[unit]}"
    elif number < 1000:
        if number in _HUNDREDS:
            words = _HUNDREDS[number]
        else:
            hundred = number // 100 * 100
            rest = number % 100
            prefix = "ciento" if hundred == 100 else _HUNDREDS[hundred]
            words = f"{prefix} {_int_to_spanish(rest)}"
    elif number < 1_000_000:
        thousands = number // 1000
        rest = number % 1000
        prefix = "mil" if thousands == 1 else f"{_int_to_spanish(thousands, apocopate=True)} mil"
        words = prefix if rest == 0 else f"{prefix} {_int_to_spanish(rest)}"
    elif number < 1_000_000_000:
        millions = number // 1_000_000
        rest = number % 1_000_000
        prefix = "un millón" if millions == 1 else f"{_int_to_spanish(millions)} millones"
        words = prefix if rest == 0 else f"{prefix} {_int_to_spanish(rest)}"
    else:
        words = str(number)

    if apocopate:
        words = _apocopate_number_words(words)
    return words


def _number_to_spanish(value: str, *, apocopate: bool = False) -> str:
    value = value.replace(".", "") if "," in value else value
    if "," in value or "." in value:
        left, right = re.split(r"[,.]", value, maxsplit=1)
        decimal_digits = " ".join(_UNITS[int(digit)] for digit in right if digit.isdigit())
        return f"{_int_to_spanish(int(left), apocopate=apocopate)} coma {decimal_digits}".strip()
    return _int_to_spanish(int(value), apocopate=apocopate)


def _expand_numeric_date(match: re.Match[str]) -> str:
    first = int(match.group(1))
    second = int(match.group(2))
    third = int(match.group(3))

    if first > 31:
        year, month, day = first, second, third
    else:
        day, month, year = first, second, third

    if not (1 <= day <= 31 and month in _MONTHS):
        return " ".join(match.groups())

    day_words = _int_to_spanish(day)
    month_words = _MONTHS[month]
    if year < 100:
        return f"{day_words} de {month_words} del {_int_to_spanish(year)}"
    return f"{day_words} de {month_words} de {_int_to_spanish(year)}"


def expand_spanish_numbers(text: str) -> str:
    """Convert common numeric transcript forms to Spanish words."""

    text = _NUMERIC_DATE.sub(_expand_numeric_date, text)
    text = _PERCENT.sub(lambda match: f"{_number_to_spanish(match.group(1))} por ciento", text)
    text = _EURO.sub(lambda match: f"{_number_to_spanish(match.group(1), apocopate=True)} euros", text)
    text = _NUMBER.sub(lambda match: _number_to_spanish(match.group(0)), text)
    return text


def normalize_spanish_text(text: str) -> str:
    """Normalize Spanish speech transcripts for tokenizer training.

    The normalization is intentionally conservative for spoken ASR: lowercase,
    keep Spanish diacritics, remove punctuation, and preserve words such as
    "eh", "mmm", "pues", "o sea", etc. as ordinary text.
    """

    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = expand_spanish_numbers(text)
    text = _DASHES.sub(" ", text)
    text = _PUNCT_TO_SPACE.sub(" ", text)
    text = _DISALLOWED.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return text


def is_probably_spanish_text(text: str) -> bool:
    """Return True when normalized text still has useful ASR content."""

    if not text:
        return False
    if not re.search(fr"[{_ALLOWED_LETTERS}]", text):
        return False
    return True
