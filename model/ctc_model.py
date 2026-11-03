#!/usr/bin/env python3
"""Spanish CTC-attention adapter for the SenseVoiceSmall encoder."""

from __future__ import annotations

import torch
from funasr.models.ctc.ctc import CTC
from torch import nn

from spanish_sensevoice_asr.model.sensevoice_encoder import SenseVoiceEncoderSmall


class SenseVoiceSmallCTC(nn.Module):
    """SenseVoice SANM encoder with Spanish CTC and optional decoder CE losses.

    This keeps the core `SenseVoiceEncoderSmall` architecture locally while
    avoiding multilingual, ITN, emotion, and rich-query control tasks. The
    auxiliary decoder loss restores the `sos`, `eos`, and length-normalized
    training behavior for a Spanish-only ASR setting.
    """

    def __init__(
        self,
        vocab_size: int,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 50,
        num_heads: int = 4,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        tp_blocks: int = 20,
        attention_dropout_rate: float = 0.1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        blank_id: int = 0,
        ignore_id: int = -1,
        sos: int = 1,
        eos: int = 2,
        length_normalized_loss: bool = True,
        decoder_layers: int = 2,
        decoder_loss_weight: float = 1.0,
        ctc_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.blank_id = blank_id
        self.ignore_id = ignore_id
        self.sos = sos
        self.eos = eos
        self.vocab_size = vocab_size
        self.length_normalized_loss = length_normalized_loss
        self.decoder_loss_weight = decoder_loss_weight
        self.ctc_loss_weight = ctc_loss_weight
        self.encoder = SenseVoiceEncoderSmall(
            input_size=input_dim,
            output_size=encoder_dim,
            attention_heads=num_heads,
            linear_units=feedforward_dim,
            num_blocks=num_layers,
            tp_blocks=tp_blocks,
            dropout_rate=dropout,
            attention_dropout_rate=attention_dropout_rate,
            kernel_size=kernel_size,
            sanm_shfit=sanm_shfit,
        )
        self.ctc = CTC(odim=vocab_size, encoder_output_size=self.encoder.output_size())
        self.decoder_embed = nn.Embedding(vocab_size, encoder_dim, padding_idx=blank_id)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.decoder_out = nn.Linear(encoder_dim, vocab_size)

    def _make_decoder_io(self, text: torch.Tensor, text_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_text_len = text.size()
        decoder_in = text.new_full((batch_size, max_text_len + 1), self.ignore_id)
        decoder_target = text.new_full((batch_size, max_text_len + 1), self.ignore_id)
        decoder_in[:, 0] = self.sos
        for index in range(batch_size):
            length = int(text_lengths[index].item())
            if length > 0:
                decoder_in[index, 1 : length + 1] = text[index, :length]
                decoder_target[index, :length] = text[index, :length]
            decoder_target[index, length] = self.eos
        decoder_embed_in = decoder_in.masked_fill(decoder_in == self.ignore_id, self.blank_id)
        return decoder_embed_in, decoder_target

    def _calc_decoder_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_in, decoder_target = self._make_decoder_io(text, text_lengths)
        target_padding_mask = decoder_target.eq(self.ignore_id)
        memory_padding_mask = torch.arange(encoder_out.size(1), device=encoder_out.device).unsqueeze(0) >= encoder_out_lens.unsqueeze(1)
        causal_mask = torch.triu(
            torch.full((decoder_in.size(1), decoder_in.size(1)), float("-inf"), device=encoder_out.device),
            diagonal=1,
        )
        decoder_states = self.decoder(
            self.decoder_embed(decoder_in),
            encoder_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        logits = self.decoder_out(decoder_states)
        per_token_loss = nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            decoder_target.reshape(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        ).reshape_as(decoder_target)
        valid_tokens = decoder_target.ne(self.ignore_id)
        if self.length_normalized_loss:
            loss = per_token_loss.sum() / valid_tokens.sum().clamp_min(1)
        else:
            loss = per_token_loss.sum() / max(text.size(0), 1)
        predictions = logits.argmax(dim=-1)
        accuracy = ((predictions == decoder_target) & valid_tokens).sum().float() / valid_tokens.sum().clamp_min(1)
        return loss, accuracy.detach()

    def encode(self, speech: torch.Tensor, speech_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]
        return self.encoder(speech, speech_lengths)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | None], int]:
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
        if self.length_normalized_loss:
            avg_tokens = text_lengths.float().mean().clamp_min(1.0)
            loss_ctc = loss_ctc / avg_tokens
        loss_decoder, acc_decoder = self._calc_decoder_loss(encoder_out, encoder_out_lens, text, text_lengths)
        loss = self.ctc_loss_weight * loss_ctc + self.decoder_loss_weight * loss_decoder
        stats: dict[str, torch.Tensor | float | None] = {
            "loss_ctc": loss_ctc.detach(),
            "loss_decoder": loss_decoder.detach(),
            "acc_decoder": acc_decoder,
            "loss": loss.detach(),
        }
        weight = int((text_lengths + 1).sum().item()) if self.length_normalized_loss else speech.size(0)
        return loss, stats, weight

    @torch.inference_mode()
    def greedy_decode(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_out, encoder_out_lens = self.encode(features, feature_lengths)
        log_probs = self.ctc.log_softmax(encoder_out)
        token_ids = log_probs.argmax(dim=-1)
        return token_ids, encoder_out_lens

    @torch.inference_mode()
    def inference(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.greedy_decode(features, feature_lengths)


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    fake_audio_seconds = 1.0
    sample_rate = 8000
    frame_shift_ms = 10
    input_dim = 80
    feature_frames = int(fake_audio_seconds * 1000 / frame_shift_ms)
    vocab_size = 64
    blank_id = 0
    ignore_id = -1

    model = SenseVoiceSmallCTC(
        vocab_size=vocab_size,
        input_dim=input_dim,
        encoder_dim=64,
        num_layers=2,
        num_heads=4,
        feedforward_dim=128,
        dropout=0.1,
        blank_id=blank_id,
        ignore_id=ignore_id,
        sos=1,
        eos=2,
        decoder_layers=1,
    ).to(device)

    fake_empty_audio = torch.zeros(batch_size, int(fake_audio_seconds * sample_rate), device=device)
    fake_fbank_features = torch.zeros(batch_size, feature_frames, input_dim, device=device)
    feature_lengths = torch.full((batch_size,), feature_frames, dtype=torch.long, device=device)

    model.eval()
    decoded_ids, decoded_lengths = model.greedy_decode(fake_fbank_features, feature_lengths)
    print("fake_empty_audio_shape", tuple(fake_empty_audio.shape))
    print("decoded_ids_shape", tuple(decoded_ids.shape))
    print("decoded_lengths", decoded_lengths.detach().cpu().tolist())

    model.train()
    targets = torch.tensor(
        [
            [1, 2, 3, 4, ignore_id],
            [5, 6, 7, ignore_id, ignore_id],
        ],
        dtype=torch.long,
        device=device,
    )
    target_lengths = torch.tensor([4, 3], dtype=torch.long, device=device)
    loss, stats, weight = model(fake_fbank_features, feature_lengths, targets, target_lengths)
    print("loss", float(loss.detach().cpu()))
    print("loss_ctc", float(stats["loss_ctc"].detach().cpu()))
    print("weight", int(weight))

    assert decoded_ids.size(0) == batch_size
    assert decoded_lengths.size(0) == batch_size
    assert torch.isfinite(loss)
    assert int(weight) == int((target_lengths + 1).sum().item())
    print("smoke_test_passed")


if __name__ == "__main__":
    main()
