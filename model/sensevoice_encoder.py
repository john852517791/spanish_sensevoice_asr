#!/usr/bin/env python3
"""Self-contained SenseVoiceSmall SANM encoder for Spanish ASR."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPositionEncoder(nn.Module):
    def __init__(self, d_model: int = 80, dropout_rate: float = 0.1) -> None:
        super().__init__()

    def encode(
        self,
        positions: torch.Tensor,
        depth: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment)
        )
        inv_timescales = torch.reshape(inv_timescales, [batch_size, -1])
        scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales, [1, 1, -1])
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)
        return x + position_encoding


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int, dropout_rate: float, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MultiHeadedAttentionSANM(nn.Module):
    def __init__(
        self,
        n_head: int,
        in_feat: int,
        n_feat: int,
        dropout_rate: float,
        kernel_size: int,
        sanm_shfit: int = 0,
    ) -> None:
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fsmn_block = nn.Conv1d(n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False)
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding += sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None,
        mask_shfit_chunk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, _ = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (batch_size, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, timesteps, _ = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = torch.reshape(q, (batch_size, timesteps, self.h, self.d_k)).transpose(1, 2)
        k_h = torch.reshape(k, (batch_size, timesteps, self.h, self.d_k)).transpose(1, 2)
        v_h = torch.reshape(v, (batch_size, timesteps, self.h, self.d_k)).transpose(1, 2)
        return q_h, k_h, v_h, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor | None,
        mask_att_chunk_encoder: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = value.size(0)
        if mask is not None:
            if mask_att_chunk_encoder is not None:
                mask = mask * mask_att_chunk_encoder
            mask = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        mask_shfit_chunk: torch.Tensor | None = None,
        mask_att_chunk_encoder: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask, mask_shfit_chunk)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask, mask_att_chunk_encoder)
        return att_outs + fsmn_memory

    def forward_chunk(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        chunk_size: tuple[int, int, int] | None = None,
        look_back: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        q_h, k_h, v_h, v = self.forward_qkv(x)
        if chunk_size is not None and look_back > 0 or look_back == -1:
            if cache is not None:
                k_h_stride = k_h[:, :, : -(chunk_size[2]), :]
                v_h_stride = v_h[:, :, : -(chunk_size[2]), :]
                k_h = torch.cat((cache["k"], k_h), dim=2)
                v_h = torch.cat((cache["v"], v_h), dim=2)
                cache["k"] = torch.cat((cache["k"], k_h_stride), dim=2)
                cache["v"] = torch.cat((cache["v"], v_h_stride), dim=2)
                if look_back != -1:
                    cache["k"] = cache["k"][:, :, -(look_back * chunk_size[1]) :, :]
                    cache["v"] = cache["v"][:, :, -(look_back * chunk_size[1]) :, :]
            else:
                cache = {
                    "k": k_h[:, :, : -(chunk_size[2]), :],
                    "v": v_h[:, :, : -(chunk_size[2]), :],
                }
        fsmn_memory = self.forward_fsmn(v, None)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, None)
        return att_outs + fsmn_memory, cache


class LayerNorm(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def sequence_mask(
    lengths: torch.Tensor,
    maxlen: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()
    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size: int,
        size: int,
        self_attn: MultiHeadedAttentionSANM,
        feed_forward: PositionwiseFeedForward,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
        stochastic_depth_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cache: torch.Tensor | None = None,
        mask_shfit_chunk: torch.Tensor | None = None,
        mask_att_chunk_encoder: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        skip_layer = False
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            attn_out = self.self_attn(
                x,
                mask,
                mask_shfit_chunk=mask_shfit_chunk,
                mask_att_chunk_encoder=mask_att_chunk_encoder,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(attn_out)
            else:
                x = stoch_layer_coeff * self.dropout(attn_out)

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder

    def forward_chunk(
        self,
        x: torch.Tensor,
        cache: dict[str, torch.Tensor] | None = None,
        chunk_size: tuple[int, int, int] | None = None,
        look_back: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.in_size == self.size:
            attn, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
            x = residual + attn
        else:
            x, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x, cache


class SenseVoiceEncoderSmall(nn.Module):
    """SenseVoiceSmall SANM encoder used by the Spanish CTC model."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        tp_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        stochastic_depth_rate: float = 0.0,
        input_layer: Optional[str] = "conv2d",
        pos_enc_class: type[nn.Module] = SinusoidalPositionEncoder,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        kernel_size: int = 11,
        sanm_shfit: int = 0,
        selfattention_layer_type: str = "sanm",
        **kwargs: object,
    ) -> None:
        super().__init__()
        self._output_size = output_size
        self.embed = pos_enc_class()
        self.normalize_before = normalize_before

        positionwise_layer_args = (output_size, linear_units, dropout_rate)
        encoder_selfattn_layer_args0 = (
            attention_heads,
            input_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            output_size,
            attention_dropout_rate,
            kernel_size,
            sanm_shfit,
        )

        self.encoders0 = nn.ModuleList(
            [
                EncoderLayerSANM(
                    input_size,
                    output_size,
                    MultiHeadedAttentionSANM(*encoder_selfattn_layer_args0),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                )
                for _ in range(1)
            ]
        )
        self.encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    MultiHeadedAttentionSANM(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                )
                for _ in range(num_blocks - 1)
            ]
        )
        self.tp_encoders = nn.ModuleList(
            [
                EncoderLayerSANM(
                    output_size,
                    output_size,
                    MultiHeadedAttentionSANM(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                )
                for _ in range(tp_blocks)
            ]
        )
        self.after_norm = LayerNorm(output_size)
        self.tp_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs_pad: torch.Tensor, ilens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masks = sequence_mask(ilens, device=ilens.device)[:, None, :]
        xs_pad = xs_pad * self.output_size() ** 0.5
        xs_pad = self.embed(xs_pad)

        for encoder_layer in self.encoders0:
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        for encoder_layer in self.encoders:
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1).int()

        for encoder_layer in self.tp_encoders:
            encoder_outs = encoder_layer(xs_pad, masks)
            xs_pad, masks = encoder_outs[0], encoder_outs[1]

        xs_pad = self.tp_norm(xs_pad)
        return xs_pad, olens
