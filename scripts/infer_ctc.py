#!/usr/bin/env python3
"""Run greedy SenseVoiceSmall CTC inference for one Spanish audio file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spanish_sensevoice_asr.model.ctc_model import SenseVoiceSmallCTC
from spanish_sensevoice_asr.model.features import load_fbank
from spanish_sensevoice_asr.model.tokenizer import SpanishAsrTokenizer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--tokens", default="spanish_sensevoice_asr/data/vocab/es_unigram_1000/tokens.json")
    parser.add_argument("--sp-model", default="spanish_sensevoice_asr/data/vocab/es_unigram_1000/es_unigram_1000.model")
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
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    tokenizer = SpanishAsrTokenizer(args.tokens, args.sp_model if args.sp_model else None)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_conf = checkpoint.get("model_conf", {})
    lfr_m = model_conf.get("lfr_m", args.lfr_m)
    lfr_n = model_conf.get("lfr_n", args.lfr_n)
    model = SenseVoiceSmallCTC(
        vocab_size=tokenizer.vocab_size,
        input_dim=model_conf.get("input_dim", args.num_mel_bins * lfr_m),
        encoder_dim=model_conf.get("encoder_dim", args.encoder_dim),
        num_layers=model_conf.get("num_layers", args.num_layers),
        num_heads=model_conf.get("num_heads", args.num_heads),
        feedforward_dim=model_conf.get("feedforward_dim", args.feedforward_dim),
        dropout=model_conf.get("dropout", args.dropout),
        tp_blocks=model_conf.get("tp_blocks", args.tp_blocks),
        attention_dropout_rate=model_conf.get("attention_dropout_rate", args.attention_dropout_rate),
        kernel_size=model_conf.get("kernel_size", args.kernel_size),
        sanm_shfit=model_conf.get("sanm_shfit", args.sanm_shfit),
        blank_id=model_conf.get("blank_id", tokenizer.blank_id),
        ignore_id=model_conf.get("ignore_id", tokenizer.ignore_id),
        sos=model_conf.get("sos", args.sos),
        eos=model_conf.get("eos", args.eos),
        length_normalized_loss=model_conf.get("length_normalized_loss", True),
        decoder_layers=model_conf.get("decoder_layers", args.decoder_layers),
        decoder_loss_weight=model_conf.get("decoder_loss_weight", 1.0),
        ctc_loss_weight=model_conf.get("ctc_loss_weight", 1.0),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    features = load_fbank(args.audio, sample_rate=args.sample_rate, num_mel_bins=args.num_mel_bins, lfr_m=lfr_m, lfr_n=lfr_n).unsqueeze(0).to(device)
    lengths = torch.tensor([features.size(1)], dtype=torch.long, device=device)
    token_ids, output_lengths = model.greedy_decode(features, lengths)
    text = tokenizer.decode_ctc(token_ids[0, : output_lengths[0]].detach().cpu().tolist())
    print(text)


if __name__ == "__main__":
    main()
