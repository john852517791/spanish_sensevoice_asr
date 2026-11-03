#!/usr/bin/env python3
"""Audio loading and filterbank feature extraction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def cached_wav_path(audio_path: str | Path, sample_rate: int = 8000) -> Path:
    path = Path(audio_path)
    rate_label = f"{sample_rate // 1000}k" if sample_rate % 1000 == 0 else f"{sample_rate}hz"
    return path.with_name(f"{path.stem}_{rate_label}wav.wav")


def load_waveform(audio_path: str | Path, sample_rate: int) -> torch.Tensor:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required to load and resample training audio") from exc

    waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return torch.from_numpy(waveform).float()


def ensure_resampled_wav(audio_path: str | Path, sample_rate: int = 8000) -> Path:
    source_path = Path(audio_path)
    cache_path = cached_wav_path(source_path, sample_rate=sample_rate)
    if cache_path.is_file():
        return cache_path

    waveform = load_waveform(source_path, sample_rate=sample_rate).cpu().numpy()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{cache_path.name}.", suffix=".tmp", dir=str(cache_path.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        sf.write(str(temp_path), waveform, sample_rate, subtype="PCM_16", format="WAV")
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return cache_path


def compute_fbank(
    waveform: torch.Tensor,
    sample_rate: int = 8000,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> torch.Tensor:
    features = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        sample_frequency=sample_rate,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        dither=0.0,
        energy_floor=0.0,
    )
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1.0)
    return (features - mean) / std


def apply_lfr(features: torch.Tensor, lfr_m: int = 1, lfr_n: int = 1) -> torch.Tensor:
    if lfr_m <= 1 and lfr_n <= 1:
        return features
    if lfr_m <= 0 or lfr_n <= 0:
        raise ValueError(f"lfr_m and lfr_n must be positive, got lfr_m={lfr_m}, lfr_n={lfr_n}")
    num_frames, feature_dim = features.shape
    if num_frames == 0:
        return features.new_zeros((0, feature_dim * lfr_m))
    lfr_frames = (num_frames + lfr_n - 1) // lfr_n
    last_frame = features[-1:].expand(lfr_m, feature_dim)
    stacked: list[torch.Tensor] = []
    for frame_index in range(lfr_frames):
        start = frame_index * lfr_n
        stop = start + lfr_m
        window = features[start:stop]
        if window.size(0) < lfr_m:
            pad = last_frame[: lfr_m - window.size(0)]
            window = torch.cat([window, pad], dim=0)
        stacked.append(window.reshape(-1))
    return torch.stack(stacked, dim=0)


def load_fbank(
    audio_path: str | Path,
    sample_rate: int = 8000,
    num_mel_bins: int = 80,
    cache_resampled_wav: bool = False,
    lfr_m: int = 1,
    lfr_n: int = 1,
) -> torch.Tensor:
    input_path = ensure_resampled_wav(audio_path, sample_rate=sample_rate) if cache_resampled_wav else audio_path
    waveform = load_waveform(input_path, sample_rate=sample_rate)
    features = compute_fbank(waveform, sample_rate=sample_rate, num_mel_bins=num_mel_bins)
    return apply_lfr(features, lfr_m=lfr_m, lfr_n=lfr_n)
