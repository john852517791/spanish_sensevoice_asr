#!/usr/bin/env python3
"""Manifest dataset for Spanish CTC ASR training."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from .features import load_fbank
from .tokenizer import SpanishAsrTokenizer

AUDIO_FIELDS = ("audio", "audio_path", "audio_filepath", "wav", "path", "file")
TEXT_FIELDS = ("target", "text", "transcript", "sentence", "normalized_text")


@dataclass(frozen=True)
class AsrExample:
    audio_path: Path
    text: str
    duration: float | None = None
    utt_id: str | None = None
    source: str | None = None


def _read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix in {".tsv", ".csv"}:
        with path.open("r", encoding="utf-8", newline="") as stream:
            delimiter = "\t" if path.suffix == ".tsv" else ","
            return list(csv.DictReader(stream, delimiter=delimiter))
    if path.suffix == ".txt":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                rows.append({"path": parts[0], "text": parts[1]})
        return rows
    raise ValueError(f"Unsupported manifest format: {path}")


def read_manifest(manifest_path: str | Path, audio_root: str | Path | None = None) -> list[AsrExample]:
    manifest_path = Path(manifest_path)
    root = Path(audio_root) if audio_root is not None else manifest_path.parent
    examples: list[AsrExample] = []
    for row in _read_rows(manifest_path):
        audio_value = next((row[field] for field in AUDIO_FIELDS if row.get(field)), None)
        text = next((row[field] for field in TEXT_FIELDS if row.get(field)), None)
        if not audio_value or not text:
            continue
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = root / audio_path
        duration = float(row["duration"]) if row.get("duration") else None
        examples.append(
            AsrExample(
                audio_path=audio_path,
                text=text,
                duration=duration,
                utt_id=row.get("utt_id") or row.get("id"),
                source=row.get("source"),
            )
        )
    return examples


class SpanishAsrDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: SpanishAsrTokenizer,
        audio_root: str | Path | None = None,
        sample_rate: int = 8000,
        num_mel_bins: int = 80,
        max_duration_seconds: float | None = None,
        cache_resampled_wav: bool = True,
        lfr_m: int = 1,
        lfr_n: int = 1,
    ) -> None:
        self.examples = read_manifest(manifest_path, audio_root=audio_root)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.max_frames = None if max_duration_seconds is None else self.estimate_lfr_frames(max_duration_seconds)
        self.cache_resampled_wav = cache_resampled_wav
        self._token_lengths: list[int] | None = None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        features = load_fbank(
            example.audio_path,
            sample_rate=self.sample_rate,
            num_mel_bins=self.num_mel_bins,
            cache_resampled_wav=self.cache_resampled_wav,
            lfr_m=self.lfr_m,
            lfr_n=self.lfr_n,
        )
        if self.max_frames is not None:
            features = features[: self.max_frames]
        token_ids = torch.tensor(self.tokenizer.encode(example.text), dtype=torch.long)
        return {
            "features": features,
            "feature_length": torch.tensor(features.size(0), dtype=torch.long),
            "tokens": token_ids,
            "token_length": torch.tensor(token_ids.numel(), dtype=torch.long),
            "text": self.tokenizer.normalize(example.text),
            "audio_path": str(example.audio_path),
            "utt_id": example.utt_id or example.audio_path.stem,
            "source": example.source or "",
        }

    @property
    def feature_dim(self) -> int:
        return self.num_mel_bins * self.lfr_m

    def estimate_lfr_frames(self, duration_seconds: float | None) -> int:
        if duration_seconds is None or duration_seconds <= 0:
            return 100
        fbank_frames = max(1, int(math.ceil(duration_seconds * 100)))
        return max(1, int(math.ceil(fbank_frames / self.lfr_n)))

    def estimate_frames(self, index: int) -> int:
        return self.estimate_lfr_frames(self.examples[index].duration)

    def token_lengths(self) -> list[int]:
        if self._token_lengths is None:
            self._token_lengths = [len(self.tokenizer.encode(example.text)) for example in self.examples]
        return self._token_lengths


def collate_asr_batch(batch: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    usable = [item for item in batch if int(item["token_length"]) > 0 and int(item["feature_length"]) > 0]
    if not usable:
        raise ValueError("Batch contains no usable examples")
    features = pad_sequence([item["features"] for item in usable], batch_first=True)
    feature_lengths = torch.stack([item["feature_length"] for item in usable])
    tokens = pad_sequence([item["tokens"] for item in usable], batch_first=True, padding_value=-1)
    token_lengths = torch.stack([item["token_length"] for item in usable])
    return {
        "features": features,
        "feature_lengths": feature_lengths,
        "tokens": tokens,
        "token_lengths": token_lengths,
        "texts": [str(item["text"]) for item in usable],
        "audio_paths": [str(item["audio_path"]) for item in usable],
        "utt_ids": [str(item["utt_id"]) for item in usable],
        "sources": [str(item["source"]) for item in usable],
    }


class DynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: SpanishAsrDataset,
        max_examples: int,
        max_frames: int,
        max_tokens: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.max_examples = max(max_examples, 1)
        self.max_frames = max(max_frames, 1)
        self.max_tokens = max(max_tokens, 1)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices.sort(key=lambda i: self.dataset.estimate_frames(i) + rng.uniform(-5, 5))
        else:
            indices.sort(key=lambda i: self.dataset.estimate_frames(i))
        token_lengths = self.dataset.token_lengths()
        batches: list[list[int]] = []
        batch: list[int] = []
        batch_frames = 0
        batch_tokens = 0
        for index in indices:
            frames = self.dataset.estimate_frames(index)
            tokens = max(token_lengths[index], 1)
            would_exceed = (
                batch
                and (
                    len(batch) >= self.max_examples
                    or batch_frames + frames > self.max_frames
                    or batch_tokens + tokens > self.max_tokens
                )
            )
            if would_exceed:
                batches.append(batch)
                batch = []
                batch_frames = 0
                batch_tokens = 0
            batch.append(index)
            batch_frames += frames
            batch_tokens += tokens
        if batch:
            batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        batches = 0
        batch_size = 0
        batch_frames = 0
        batch_tokens = 0
        token_lengths = self.dataset.token_lengths()
        for index in range(len(self.dataset)):
            frames = self.dataset.estimate_frames(index)
            tokens = max(token_lengths[index], 1)
            would_exceed = (
                batch_size > 0
                and (
                    batch_size >= self.max_examples
                    or batch_frames + frames > self.max_frames
                    or batch_tokens + tokens > self.max_tokens
                )
            )
            if would_exceed:
                batches += 1
                batch_size = 0
                batch_frames = 0
                batch_tokens = 0
            batch_size += 1
            batch_frames += frames
            batch_tokens += tokens
        return batches + (1 if batch_size > 0 else 0)


class DistributedDynamicBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: SpanishAsrDataset,
        max_examples: int,
        max_frames: int,
        max_tokens: int,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.base_sampler = DynamicBatchSampler(
            dataset,
            max_examples=max_examples,
            max_frames=max_frames,
            max_tokens=max_tokens,
            shuffle=shuffle,
            seed=seed,
        )
        self.rank = rank
        self.world_size = max(world_size, 1)

    def set_epoch(self, epoch: int) -> None:
        self.base_sampler.set_epoch(epoch)

    def _rank_batches(self) -> list[list[int]]:
        batches = list(self.base_sampler)
        if not batches:
            return []
        remainder = len(batches) % self.world_size
        if remainder:
            pad_count = self.world_size - remainder
            batches.extend(batches[-1:] * pad_count)
        return batches[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._rank_batches()

    def __len__(self) -> int:
        total = len(self.base_sampler)
        if total == 0:
            return 0
        return int(math.ceil(total / self.world_size))
