#!/usr/bin/env python3
"""Convert Common Voice TSV manifests to JSONL ASR manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Common Voice TSV file with a header row")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path")
    parser.add_argument("--audio-root", required=True, help="Root directory for Common Voice audio files")
    parser.add_argument(
        "--path-prefix",
        default="",
        help="Optional subdirectory between audio root and TSV path values, such as clips",
    )
    parser.add_argument("--audio-ext", default=".mp3", help="Audio extension to append when path has no suffix")
    parser.add_argument("--source", default="commonvoice_es", help="Source field written to every JSON object")
    parser.add_argument("--compute-duration", action="store_true", help="Read audio headers and fill real duration seconds")
    parser.add_argument("--skip-missing-audio", action="store_true", help="Drop rows whose audio file does not exist")
    return parser.parse_args()


def build_audio_path(audio_root: Path, tsv_path: str, path_prefix: str, audio_ext: str) -> Path:
    relative_path = Path(tsv_path)
    if not relative_path.suffix:
        relative_path = relative_path.with_suffix(audio_ext)
    if path_prefix:
        relative_path = Path(path_prefix) / relative_path
    return audio_root / relative_path


def audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return round(info.frames / float(info.samplerate), 3)
    except (ImportError, RuntimeError, OSError):
        pass

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("soundfile or librosa is required for --compute-duration with MP3 files") from exc

    return round(float(librosa.get_duration(path=str(path))), 3)


def convert_manifest(
    input_path: Path,
    output_path: Path,
    audio_root: Path,
    path_prefix: str,
    audio_ext: str,
    source: str,
    compute_duration: bool,
    skip_missing_audio: bool,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8", newline="") as src, output_path.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"input TSV has no header: {input_path}")

        required_fields = {"path", "sentence"}
        missing_fields = required_fields.difference(reader.fieldnames)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"input TSV missing required field(s): {missing}")

        for line_no, row in enumerate(reader, start=2):
            tsv_path = (row.get("path") or "").strip()
            text = (row.get("sentence") or "").strip()
            if not tsv_path or not text:
                skipped += 1
                print(f"skip line {line_no}: empty path or sentence")
                continue

            audio_path = build_audio_path(audio_root, tsv_path, path_prefix, audio_ext)
            if not audio_path.is_file():
                message = f"line {line_no}: audio not found: {audio_path}"
                if skip_missing_audio:
                    skipped += 1
                    print(f"skip {message}")
                    continue
                print(f"warn {message}")

            duration = 0.0
            if compute_duration and audio_path.is_file():
                try:
                    duration = audio_duration_seconds(audio_path)
                except (RuntimeError, OSError) as exc:
                    skipped += 1
                    print(f"skip line {line_no}: cannot read audio duration for {audio_path}: {exc}")
                    continue

            utt_id = Path(tsv_path).stem
            record = {
                "audio": str(audio_path),
                "text": text,
                "utt_id": utt_id,
                "duration": duration,
                "source": source,
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written, skipped


def main() -> None:
    args = parse_args()
    audio_ext = args.audio_ext if args.audio_ext.startswith(".") else f".{args.audio_ext}"
    written, skipped = convert_manifest(
        input_path=Path(args.input),
        output_path=Path(args.output),
        audio_root=Path(args.audio_root),
        path_prefix=args.path_prefix,
        audio_ext=audio_ext,
        source=args.source,
        compute_duration=args.compute_duration,
        skip_missing_audio=args.skip_missing_audio,
    )
    print(f"written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
