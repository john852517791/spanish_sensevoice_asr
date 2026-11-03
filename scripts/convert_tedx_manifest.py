#!/usr/bin/env python3
"""Convert TEDX Spanish text manifests to JSONL ASR manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Input txt file: <transcription><space><wav_id_or_name> per line",
    )
    parser.add_argument("--output", required=True, help="Output JSONL manifest path")
    parser.add_argument("--audio-root", required=True, help="Root directory containing TEDX wav files")
    parser.add_argument("--audio-ext", default=".wav", help="Audio extension to append when file id has no suffix")
    parser.add_argument("--source", default="tedx_es", help="Source field written to every JSON object")
    parser.add_argument("--compute-duration", action="store_true", help="Read audio headers and fill real duration seconds")
    parser.add_argument("--skip-missing-audio", action="store_true", help="Drop rows whose audio file does not exist")
    return parser.parse_args()


def build_audio_path(audio_root: Path, wav_id: str, audio_ext: str) -> Path:
    suffix = Path(wav_id).suffix
    file_name = wav_id if suffix else f"{wav_id}{audio_ext}"
    return audio_root / file_name


def audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for --compute-duration with WAV/FLAC files") from exc

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

            pieces = line.rsplit(maxsplit=1)
            if len(pieces) != 2:
                skipped += 1
                print(f"skip line {line_no}: expected text followed by wav id")
                continue

            text, wav_id = pieces[0].strip(), pieces[1].strip()
            if not text or not wav_id:
                skipped += 1
                print(f"skip line {line_no}: empty text or wav id")
                continue

            audio_path = build_audio_path(audio_root, wav_id, audio_ext)
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

            utt_id = Path(wav_id).stem
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
