#!/usr/bin/env python3
"""Build three Spanish ASR tokenizer variants.

Outputs:
- es_char/tokens.json
- es_unigram_1000/tokens.json and sentencepiece model files
- es_unigram_2000/tokens.json and sentencepiece model files
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spanish_text_norm import is_probably_spanish_text, normalize_spanish_text

BASE_SPECIAL_TOKENS = ["<blank>", "<unk>", "<s>", "</s>"]
CONTROL_TOKENS = ["<|es|>", "<|ASR|>", "<|nospeech|>"]
TEXT_FIELDS = ("target", "text", "transcript", "sentence", "normalized_text")
CHAR_TOKENS = list("abcdefghijklmnopqrstuvwxyzáéíóúüñ") + ["▁"]


def iter_paths(inputs: Iterable[str]) -> Iterator[Path]:
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            for pattern in ("*.jsonl", "*.txt", "*.tsv", "*.csv"):
                yield from sorted(path.rglob(pattern))
        else:
            yield path


def extract_text_from_record(record: dict[str, object]) -> str | None:
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def iter_raw_text(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                text = extract_text_from_record(record)
                if text:
                    yield text
    elif suffix in {".tsv", ".csv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as file:
            sample = file.read(4096)
            file.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample else False
            if has_header:
                reader = csv.DictReader(file, delimiter=delimiter)
                for record in reader:
                    text = extract_text_from_record(record)
                    if text:
                        yield text
            else:
                reader = csv.reader(file, delimiter=delimiter)
                for row in reader:
                    if row:
                        yield row[-1]
    else:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    yield line


def collect_normalized_text(inputs: Iterable[str], output_text: Path) -> int:
    seen: set[str] = set()
    count = 0
    output_text.parent.mkdir(parents=True, exist_ok=True)
    with output_text.open("w", encoding="utf-8") as out:
        for path in iter_paths(inputs):
            for raw_text in iter_raw_text(path):
                text = normalize_spanish_text(raw_text)
                if not is_probably_spanish_text(text):
                    continue
                if text in seen:
                    continue
                seen.add(text)
                out.write(text + "\n")
                count += 1
    return count


def write_tokens_json(tokens: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(tokens, file, ensure_ascii=False, indent=2)
        file.write("\n")


def build_char_vocab(output_dir: Path) -> None:
    tokens = BASE_SPECIAL_TOKENS + CHAR_TOKENS + CONTROL_TOKENS
    write_tokens_json(tokens, output_dir / "es_char" / "tokens.json")


def train_sentencepiece(text_path: Path, output_dir: Path, vocab_size: int) -> None:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            "sentencepiece is required for unigram tokenizers. Install with: "
            "pip install sentencepiece"
        ) from exc

    variant_dir = output_dir / f"es_unigram_{vocab_size}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = variant_dir / f"es_unigram_{vocab_size}"
    user_defined_symbols = ",".join(CONTROL_TOKENS)

    spm.SentencePieceTrainer.Train(
        input=str(text_path),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=1.0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_id=-1,
        user_defined_symbols=user_defined_symbols,
        hard_vocab_limit=False,
        split_digits=True,
        byte_fallback=False,
        normalization_rule_name="identity",
    )

    vocab_path = model_prefix.with_suffix(".vocab")
    pieces: list[str] = []
    with vocab_path.open("r", encoding="utf-8") as file:
        for line in file:
            piece = line.split("\t", 1)[0]
            if piece in CONTROL_TOKENS:
                continue
            pieces.append(piece)

    tokens = ["<blank>"] + pieces + CONTROL_TOKENS
    if tokens[1:4] != ["<unk>", "<s>", "</s>"]:
        raise RuntimeError(f"Unexpected SentencePiece special token order: {tokens[:6]}")
    write_tokens_json(tokens, variant_dir / "tokens.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input transcript files or directories: jsonl, txt, csv, tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default="spanish_sensevoice_asr/data/vocab",
        help="Directory for tokenizer artifacts.",
    )
    parser.add_argument(
        "--normalized-text",
        default="spanish_sensevoice_asr/data/text/tokenizer_train.es.txt",
        help="Path for deduplicated normalized tokenizer text.",
    )
    parser.add_argument(
        "--skip-sentencepiece",
        action="store_true",
        help="Only build char vocab and normalized text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    normalized_text = Path(args.normalized_text)

    count = collect_normalized_text(args.input, normalized_text)
    if count == 0:
        print("No usable normalized Spanish text was found.", file=sys.stderr)
        return 1

    build_char_vocab(output_dir)

    if not args.skip_sentencepiece:
        train_sentencepiece(normalized_text, output_dir, 1000)
        train_sentencepiece(normalized_text, output_dir, 2000)

    print(f"normalized_lines={count}")
    print(f"normalized_text={normalized_text}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
