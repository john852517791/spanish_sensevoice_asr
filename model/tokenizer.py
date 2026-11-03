#!/usr/bin/env python3
"""Tokenizer wrappers for Spanish CTC ASR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - optional until subword tokenizers are used.
    spm = None

from spanish_sensevoice_asr.scripts.spanish_text_norm import normalize_spanish_text


class SpanishAsrTokenizer:
    """Encode normalized Spanish text and decode CTC token IDs."""

    def __init__(self, tokens_path: str | Path, sp_model_path: str | Path | None = None) -> None:
        self.tokens_path = Path(tokens_path)
        self.tokens: list[str] = json.loads(self.tokens_path.read_text(encoding="utf-8"))
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.blank_id = self.token_to_id["<blank>"]
        self.unk_id = self.token_to_id.get("<unk>", self.blank_id)
        self.ignore_id = -1
        self.special_ids = {
            idx for token, idx in self.token_to_id.items() if token.startswith("<") and token.endswith(">")
        }
        self.is_sentencepiece = sp_model_path is not None
        self.sp = None
        if sp_model_path is not None:
            if spm is None:
                raise RuntimeError("sentencepiece is required for subword tokenizers")
            self.sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def normalize(self, text: str) -> str:
        return normalize_spanish_text(text)

    def encode(self, text: str) -> list[int]:
        text = self.normalize(text)
        if not text:
            return []
        if self.sp is not None:
            pieces = self.sp.encode(text, out_type=str)
            return [self.token_to_id.get(piece, self.unk_id) for piece in pieces]
        return [self.token_to_id.get(char, self.unk_id) for char in text.replace(" ", "▁")]

    def decode_ctc(self, token_ids: Iterable[int]) -> str:
        words: list[str] = []
        last_id: int | None = None
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.blank_id:
                last_id = token_id
                continue
            if token_id == last_id:
                continue
            last_id = token_id
            if token_id in self.special_ids:
                continue
            if 0 <= token_id < len(self.tokens):
                words.append(self.tokens[token_id])
        text = "".join(words).replace("▁", " ")
        return " ".join(text.split())
