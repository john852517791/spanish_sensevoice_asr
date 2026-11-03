#!/usr/bin/env python3
"""Train the Spanish SenseVoiceSmall SANM model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanish_sensevoice_asr.model.ctc_model import SenseVoiceSmallCTC
from spanish_sensevoice_asr.model.dataset import (
    DistributedDynamicBatchSampler,
    DynamicBatchSampler,
    SpanishAsrDataset,
    collate_asr_batch,
)
from spanish_sensevoice_asr.model.tokenizer import SpanishAsrTokenizer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config file with training arguments")
    parser.add_argument("--train-manifest")
    parser.add_argument("--dev-manifest")
    parser.add_argument("--audio-root")
    parser.add_argument("--tokens", default="spanish_sensevoice_asr/data/vocab/es_unigram_1000/tokens.json")
    parser.add_argument("--sp-model", default="spanish_sensevoice_asr/data/vocab/es_unigram_1000/es_unigram_1000.model")
    parser.add_argument("--exp-dir", default="spanish_sensevoice_asr/exp/es_unigram_1000_ctc_attention")
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--num-mel-bins", type=int, default=80)
    parser.add_argument("--lfr-m", type=int, default=7)
    parser.add_argument("--lfr-n", type=int, default=6)
    parser.add_argument("--encoder-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=50)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tp-blocks", type=int, default=20)
    parser.add_argument("--attention-dropout-rate", type=float, default=0.1)
    parser.add_argument("--kernel-size", type=int, default=11)
    parser.add_argument("--sanm-shfit", type=int, default=0)
    parser.add_argument("--sos", type=int, default=1)
    parser.add_argument("--eos", type=int, default=2)
    parser.add_argument("--length-normalized-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--decoder-loss-weight", type=float, default=1.0)
    parser.add_argument("--ctc-loss-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batching", choices=("fixed", "dynamic"), default="dynamic")
    parser.add_argument("--max-frames-per-batch", type=int, default=2400)
    parser.add_argument("--max-tokens-per-batch", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-duration-seconds", type=float, default=30.0)
    parser.add_argument("--cache-resampled-wav", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--specaug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freq-mask-width", type=int, default=30)
    parser.add_argument("--num-freq-mask", type=int, default=1)
    parser.add_argument("--time-mask-width", type=int, default=12)
    parser.add_argument("--num-time-mask", type=int, default=1)
    parser.add_argument("--ctc-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ctc-min-frames-per-token", type=float, default=1.2)
    parser.add_argument("--ctc-diagnostics-limit", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--scheduler", choices=("cosine", "linear", "warmuplr", "none"), default="warmuplr")
    parser.add_argument("--warmup-steps", type=int, default=25000)
    parser.add_argument("--max-steps", type=int, default=0, help="Override total optimizer steps for LR scheduling")
    parser.add_argument("--resume", help="Checkpoint path to resume from")
    parser.add_argument("--pretrained-model", help="Path to pretrained checkpoint (loads model weights only, resets optimizer/scheduler)")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from exp-dir/last.pt when it exists")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    config_args, _ = parser.parse_known_args()
    if config_args.config:
        config = json.loads(Path(config_args.config).read_text(encoding="utf-8"))
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            parser.error(f"Unknown config keys: {', '.join(unknown_keys)}")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.train_manifest:
        parser.error("--train-manifest is required unless provided by --config")
    return args


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return distributed, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_main(rank: int, message: str) -> None:
    if is_main_process(rank):
        print(message, flush=True)


def print_rank(rank: int, message: str) -> None:
    print(f"rank={rank} {message}", flush=True)


def resolve_device(args: argparse.Namespace, distributed: bool, local_rank: int) -> torch.device:
    if args.device.startswith("cuda") and torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device(args.device)
    return torch.device("cpu")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_average(value: float, count: int, device: torch.device, distributed: bool) -> float:
    totals = torch.tensor([value, float(count)], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item() / max(totals[1].item(), 1.0))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if scheduler_name == "none":
        return None
    total_steps = max(total_steps, 1)
    warmup_steps = max(warmup_steps, 0)

    def lr_lambda(step: int) -> float:
        if scheduler_name == "warmuplr":
            return warmup_steps ** 0.5 * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5) if warmup_steps > 0 else 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if scheduler_name == "linear":
            remaining = max(total_steps - step, 0)
            decay_steps = max(total_steps - warmup_steps, 1)
            return max(float(remaining) / float(decay_steps), 0.0)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_loader(
    dataset: SpanishAsrDataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
    batching: str = "fixed",
    max_frames_per_batch: int = 2400,
    max_tokens_per_batch: int = 2000,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, object | None]:
    if batching == "dynamic":
        if distributed:
            batch_sampler = DistributedDynamicBatchSampler(
                dataset,
                max_examples=batch_size,
                max_frames=max_frames_per_batch,
                max_tokens=max_tokens_per_batch,
                rank=rank,
                world_size=world_size,
                shuffle=shuffle,
            )
        else:
            batch_sampler = DynamicBatchSampler(
                dataset,
                max_examples=batch_size,
                max_frames=max_frames_per_batch,
                max_tokens=max_tokens_per_batch,
                shuffle=shuffle,
            )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_asr_batch,
            pin_memory=torch.cuda.is_available(),
        )
        return loader, batch_sampler

    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_asr_batch,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    distributed: bool,
    rank: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 5.0,
    log_interval: int = 20,
    global_step: int = 0,
    specaug: bool = False,
    freq_mask_width: int = 30,
    num_freq_mask: int = 1,
    time_mask_width: int = 12,
    num_time_mask: int = 1,
    ctc_diagnostics: bool = True,
    ctc_min_frames_per_token: float = 1.2,
    ctc_diagnostics_limit: int = 8,
) -> tuple[float, float, float, float, int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_dec_loss = 0.0
    total_dec_acc = 0.0
    total_batches = 0
    grad_accum_steps = max(grad_accum_steps, 1)

    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader, start=1):
        features = batch["features"].to(device, non_blocking=True)
        feature_lengths = batch["feature_lengths"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        token_lengths = batch["token_lengths"].to(device, non_blocking=True)
        if ctc_diagnostics:
            log_ctc_risk_examples(
                batch,
                feature_lengths,
                token_lengths,
                rank=rank,
                batch_idx=batch_idx,
                min_frames_per_token=ctc_min_frames_per_token,
                limit=ctc_diagnostics_limit,
            )
        if training and specaug:
            features = apply_specaug(features, feature_lengths, freq_mask_width, num_freq_mask, time_mask_width, num_time_mask)

        with torch.set_grad_enabled(training):
            loss, stats, _ = model(features, feature_lengths, tokens, token_lengths)
            if ctc_diagnostics and not torch.isfinite(loss.detach()):
                log_batch_examples(batch, feature_lengths, token_lengths, rank=rank, batch_idx=batch_idx, reason="nonfinite_loss")
            if training:
                (loss / grad_accum_steps).backward()
                should_step = batch_idx % grad_accum_steps == 0 or batch_idx == len(loader)
                if should_step:
                    torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

        total_loss += float(loss.detach().cpu())
        total_ctc_loss += float(stats["loss_ctc"].detach().cpu())
        total_dec_loss += float(stats["loss_decoder"].detach().cpu())
        total_dec_acc += float(stats["acc_decoder"].detach().cpu())
        total_batches += 1
        if training and batch_idx % log_interval == 0:
            loss_ctc = float(stats["loss_ctc"].detach().cpu())
            loss_decoder = float(stats["loss_decoder"].detach().cpu())
            lr = optimizer.param_groups[0]["lr"]
            print_main(
                rank,
                f"batch={batch_idx} step={global_step} lr={lr:.6g} "
                f"loss={total_loss / total_batches:.4f} loss_ctc={loss_ctc:.4f} loss_decoder={loss_decoder:.4f}",
            )

    return (
        reduce_average(total_loss, total_batches, device, distributed),
        reduce_average(total_ctc_loss, total_batches, device, distributed),
        reduce_average(total_dec_loss, total_batches, device, distributed),
        reduce_average(total_dec_acc, total_batches, device, distributed),
        global_step,
    )


def log_ctc_risk_examples(
    batch: dict[str, torch.Tensor | list[str]],
    feature_lengths: torch.Tensor,
    token_lengths: torch.Tensor,
    rank: int,
    batch_idx: int,
    min_frames_per_token: float,
    limit: int,
) -> None:
    if limit <= 0:
        return
    cpu_feature_lengths = feature_lengths.detach().cpu()
    cpu_token_lengths = token_lengths.detach().cpu()
    risky_indices = []
    for index, (frames, tokens) in enumerate(zip(cpu_feature_lengths.tolist(), cpu_token_lengths.tolist(), strict=False)):
        ratio = float(frames) / max(float(tokens), 1.0)
        if int(tokens) >= int(frames) or ratio < min_frames_per_token:
            risky_indices.append(index)
    if not risky_indices:
        return
    print_rank(
        rank,
        f"ctc_risk batch={batch_idx} count={len(risky_indices)} "
        f"min_frames_per_token={min_frames_per_token}",
    )
    for index in risky_indices[:limit]:
        print_rank(rank, format_batch_example(batch, cpu_feature_lengths, cpu_token_lengths, index, reason="ctc_risk"))


def log_batch_examples(
    batch: dict[str, torch.Tensor | list[str]],
    feature_lengths: torch.Tensor,
    token_lengths: torch.Tensor,
    rank: int,
    batch_idx: int,
    reason: str,
    limit: int = 8,
) -> None:
    cpu_feature_lengths = feature_lengths.detach().cpu()
    cpu_token_lengths = token_lengths.detach().cpu()
    print_rank(rank, f"{reason} batch={batch_idx} batch_size={len(cpu_feature_lengths)}")
    for index in range(min(len(cpu_feature_lengths), limit)):
        print_rank(rank, format_batch_example(batch, cpu_feature_lengths, cpu_token_lengths, index, reason=reason))


def format_batch_example(
    batch: dict[str, torch.Tensor | list[str]],
    feature_lengths: torch.Tensor,
    token_lengths: torch.Tensor,
    index: int,
    reason: str,
) -> str:
    frames = int(feature_lengths[index].item())
    tokens = int(token_lengths[index].item())
    frames_per_token = float(frames) / max(float(tokens), 1.0)
    utt_ids = batch.get("utt_ids", [])
    audio_paths = batch.get("audio_paths", [])
    sources = batch.get("sources", [])
    texts = batch.get("texts", [])
    utt_id = str(utt_ids[index]) if isinstance(utt_ids, list) and index < len(utt_ids) else ""
    audio_path = str(audio_paths[index]) if isinstance(audio_paths, list) and index < len(audio_paths) else ""
    source = str(sources[index]) if isinstance(sources, list) and index < len(sources) else ""
    text = str(texts[index]) if isinstance(texts, list) and index < len(texts) else ""
    return (
        f"{reason} index={index} utt_id={utt_id} source={source} "
        f"frames={frames} tokens={tokens} frames_per_token={frames_per_token:.3f} "
        f"audio={audio_path} text={text}"
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    epoch: int,
    global_step: int,
    train_loss: float,
    dev_loss: float | None,
    best_dev_loss: float,
    tokenizer: SpanishAsrTokenizer,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "best_dev_loss": best_dev_loss,
        "vocab_size": tokenizer.vocab_size,
        "args": vars(args),
        "model_conf": {
            "input_dim": args.num_mel_bins * args.lfr_m,
            "lfr_m": args.lfr_m,
            "lfr_n": args.lfr_n,
            "encoder_dim": args.encoder_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "feedforward_dim": args.feedforward_dim,
            "dropout": args.dropout,
            "tp_blocks": args.tp_blocks,
            "attention_dropout_rate": args.attention_dropout_rate,
            "kernel_size": args.kernel_size,
            "sanm_shfit": args.sanm_shfit,
            "blank_id": tokenizer.blank_id,
            "ignore_id": tokenizer.ignore_id,
            "sos": args.sos,
            "eos": args.eos,
            "length_normalized_loss": args.length_normalized_loss,
            "decoder_layers": args.decoder_layers,
            "decoder_loss_weight": args.decoder_loss_weight,
            "ctc_loss_weight": args.ctc_loss_weight,
        },
    }
    torch.save(checkpoint, path)


def apply_specaug(
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
    freq_mask_width: int,
    num_freq_mask: int,
    time_mask_width: int,
    num_time_mask: int,
) -> torch.Tensor:
    augmented = features.clone()
    batch_size, _, num_bins = augmented.shape
    for batch_index in range(batch_size):
        valid_frames = int(feature_lengths[batch_index].item())
        for _ in range(num_freq_mask):
            width = int(torch.randint(0, max(freq_mask_width, 0) + 1, ()).item())
            if width > 0 and width < num_bins:
                start = int(torch.randint(0, num_bins - width + 1, ()).item())
                augmented[batch_index, :valid_frames, start : start + width] = 0.0
        for _ in range(num_time_mask):
            width = int(torch.randint(0, max(time_mask_width, 0) + 1, ()).item())
            if width > 0 and width < valid_frames:
                start = int(torch.randint(0, valid_frames - width + 1, ()).item())
                augmented[batch_index, start : start + width, :] = 0.0
    return augmented


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_dev_loss = float(checkpoint.get("best_dev_loss", float("inf")))
    return start_epoch, global_step, best_dev_loss


def load_pretrained_weights(path: Path, model: torch.nn.Module, device: torch.device, rank: int) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    target_model = unwrap_model(model)
    target_state = target_model.state_dict()
    compatible_state = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        target_key = key[7:] if key.startswith("module.") else key
        if target_key in target_state and torch.is_tensor(value) and target_state[target_key].shape == value.shape:
            compatible_state[target_key] = value
        else:
            skipped.append(key)

    target_state.update(compatible_state)
    target_model.load_state_dict(target_state)
    print_main(
        rank,
        f"pretrained_model={path} loaded_tensors={len(compatible_state)} skipped_tensors={len(skipped)}",
    )
    if skipped:
        preview = ", ".join(skipped[:30])
        suffix = " ..." if len(skipped) > 30 else ""
        print_main(rank, f"pretrained_skipped={preview}{suffix}")


def main() -> None:
    args = get_args()
    distributed, rank, world_size, local_rank = setup_distributed()
    device = resolve_device(args, distributed, local_rank)
    exp_dir = Path(args.exp_dir)
    if is_main_process(rank):
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    if distributed:
        dist.barrier()

    tokenizer = SpanishAsrTokenizer(args.tokens, args.sp_model if args.sp_model else None)
    train_set = SpanishAsrDataset(
        args.train_manifest,
        tokenizer,
        audio_root=args.audio_root,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
        max_duration_seconds=args.max_duration_seconds,
        cache_resampled_wav=args.cache_resampled_wav,
        lfr_m=args.lfr_m,
        lfr_n=args.lfr_n,
    )
    train_loader, train_sampler = make_loader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=distributed,
        shuffle=True,
        batching=args.batching,
        max_frames_per_batch=args.max_frames_per_batch,
        max_tokens_per_batch=args.max_tokens_per_batch,
        rank=rank,
        world_size=world_size,
    )

    dev_loader = None
    dev_sampler = None
    if args.dev_manifest:
        dev_set = SpanishAsrDataset(
            args.dev_manifest,
            tokenizer,
            audio_root=args.audio_root,
            sample_rate=args.sample_rate,
            num_mel_bins=args.num_mel_bins,
            max_duration_seconds=args.max_duration_seconds,
            cache_resampled_wav=args.cache_resampled_wav,
            lfr_m=args.lfr_m,
            lfr_n=args.lfr_n,
        )
        dev_loader, dev_sampler = make_loader(
            dev_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            distributed=distributed,
            shuffle=False,
            batching=args.batching,
            max_frames_per_batch=args.max_frames_per_batch,
            max_tokens_per_batch=args.max_tokens_per_batch,
            rank=rank,
            world_size=world_size,
        )

    model = SenseVoiceSmallCTC(
        vocab_size=tokenizer.vocab_size,
        input_dim=train_set.feature_dim,
        encoder_dim=args.encoder_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        tp_blocks=args.tp_blocks,
        attention_dropout_rate=args.attention_dropout_rate,
        kernel_size=args.kernel_size,
        sanm_shfit=args.sanm_shfit,
        blank_id=tokenizer.blank_id,
        ignore_id=tokenizer.ignore_id,
        sos=args.sos,
        eos=args.eos,
        length_normalized_loss=args.length_normalized_loss,
        decoder_layers=args.decoder_layers,
        decoder_loss_weight=args.decoder_loss_weight,
        ctc_loss_weight=args.ctc_loss_weight,
    ).to(device)
    if distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model = DistributedDataParallel(model, **ddp_kwargs)

    resume_path = Path(args.resume) if args.resume else None
    if resume_path is None and args.auto_resume and (exp_dir / "last.pt").is_file():
        resume_path = exp_dir / "last.pt"

    if args.pretrained_model and resume_path is None:
        load_pretrained_weights(Path(args.pretrained_model), model, device, rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1))
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, args.scheduler, args.warmup_steps, total_steps)

    start_epoch = 1
    global_step = 0
    best_dev_loss = float("inf")
    if resume_path is not None:
        start_epoch, global_step, best_dev_loss = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        print_main(rank, f"resumed={resume_path} start_epoch={start_epoch} global_step={global_step}")

    print_main(
        rank,
        f"world_size={world_size} device={device} scheduler={args.scheduler} "
        f"warmup_steps={args.warmup_steps} total_steps={total_steps} grad_accum_steps={args.grad_accum_steps}",
    )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss, train_ctc_loss, train_dec_loss, train_dec_acc, global_step = run_epoch(
                model,
                train_loader,
                device,
                distributed=distributed,
                rank=rank,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_accum_steps=args.grad_accum_steps,
                max_grad_norm=args.max_grad_norm,
                log_interval=args.log_interval,
                global_step=global_step,
                specaug=args.specaug,
                freq_mask_width=args.freq_mask_width,
                num_freq_mask=args.num_freq_mask,
                time_mask_width=args.time_mask_width,
                num_time_mask=args.num_time_mask,
                ctc_diagnostics=args.ctc_diagnostics,
                ctc_min_frames_per_token=args.ctc_min_frames_per_token,
                ctc_diagnostics_limit=args.ctc_diagnostics_limit,
            )
            print_main(
                rank,
                f"epoch={epoch} train_loss={train_loss:.4f} train_ctc={train_ctc_loss:.4f} "
                f"train_dec={train_dec_loss:.4f} train_dec_acc={train_dec_acc:.4f}",
            )

            dev_loss = None
            dev_ctc_loss = None
            dev_dec_loss = None
            dev_dec_acc = None
            if dev_loader is not None:
                if dev_sampler is not None:
                    dev_sampler.set_epoch(epoch)
                with torch.inference_mode():
                    dev_loss, dev_ctc_loss, dev_dec_loss, dev_dec_acc, global_step = run_epoch(
                        model,
                        dev_loader,
                        device,
                        distributed=distributed,
                        rank=rank,
                        global_step=global_step,
                        ctc_diagnostics=args.ctc_diagnostics,
                        ctc_min_frames_per_token=args.ctc_min_frames_per_token,
                        ctc_diagnostics_limit=args.ctc_diagnostics_limit,
                    )
                print_main(
                    rank,
                    f"epoch={epoch} dev_loss={dev_loss:.4f} dev_ctc={dev_ctc_loss:.4f} "
                    f"dev_dec={dev_dec_loss:.4f} dev_dec_acc={dev_dec_acc:.4f}",
                )

            metric = train_ctc_loss if dev_ctc_loss is None else dev_ctc_loss
            if is_main_process(rank):
                improved = metric < best_dev_loss
                if improved:
                    best_dev_loss = metric
                for checkpoint_name in (f"epoch-{epoch}.pt", "last.pt"):
                    save_checkpoint(
                        exp_dir / checkpoint_name,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        train_loss,
                        dev_loss,
                        best_dev_loss,
                        tokenizer,
                        args,
                    )
                if improved:
                    save_checkpoint(
                        exp_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        train_loss,
                        dev_loss,
                        best_dev_loss,
                        tokenizer,
                        args,
                    )
            if distributed:
                dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
