#!/usr/bin/env python3
"""Convert LibriSpeech-style Spanish text manifests to JSONL ASR manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input txt file: <utt_id><tab><transcription> per line")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path")
    parser.add_argument("--audio-root", required=True, help="Root directory containing two-level audio folders")
    parser.add_argument("--audio-ext", default=".flac", help="Audio extension to append when utt id has no suffix")
    parser.add_argument("--source", default="librispeech_es", help="Source field written to every JSON object")
    parser.add_argument("--compute-duration", action="store_true", help="Read audio headers and fill real duration seconds")
    parser.add_argument("--skip-missing-audio", action="store_true", help="Drop rows whose audio file does not exist")
    return parser.parse_args()


def build_audio_path(audio_root: Path, utt_id: str, audio_ext: str) -> Path:
    parts = utt_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"utt_id must contain at least 3 underscore-separated parts: {utt_id}")
    suffix = Path(utt_id).suffix
    file_name = utt_id if suffix else f"{utt_id}{audio_ext}"
    return audio_root / parts[0] / parts[1] / file_name


def audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for --compute-duration with FLAC/WAV files") from exc

    info = sf.info(str(path))
    return round(info.frames / float(info.samplerate), 3)


def convert_manifest(
    input_path: Path,
    output_path: Path,
    audio_root: Path,
    audio_ext: str,
    source: str,
    compute_duration: bool,
    skip_missing_audio: bool,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, raw_line in enumerate(src, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                utt_id, text = line.split("\t", 1)
            else:
                pieces = line.split(maxsplit=1)
                if len(pieces) != 2:
                    skipped += 1
                    print(f"skip line {line_no}: expected utt_id and text")
                    continue
                utt_id, text = pieces

            utt_id = utt_id.strip()
            text = text.strip()
            if not utt_id or not text:
                skipped += 1
                print(f"skip line {line_no}: empty utt_id or text")
                continue

            try:
                audio_path = build_audio_path(audio_root, utt_id, audio_ext)
            except ValueError as exc:
                skipped += 1
                print(f"skip line {line_no}: {exc}")
                continue

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
        audio_ext=audio_ext,
        source=args.source,
        compute_duration=args.compute_duration,
        skip_missing_audio=args.skip_missing_audio,
    )
    print(f"written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
