#!/usr/bin/env python3
"""
GluMind architecture module.

This file intentionally contains model-only code so checkpoints can be loaded
without pulling the full training pipeline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention: glucose (Q) attends to each auxiliary modality (K/V).
    Outputs are averaged, followed by FFN + residual + LayerNorm.
    """

    def __init__(self, d_model: int, n_heads: int, ff_units: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn_aux1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_aux2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )

    def forward(self, glucose: torch.Tensor, aux1: torch.Tensor,
                aux2: torch.Tensor) -> torch.Tensor:
        """All inputs: (seq_len, batch, d_model)."""
        # glucose queries aux1 (HR)
        out1, _ = self.attn_aux1(glucose, aux1, aux1)
        res1 = self.ln1(glucose + self.dropout(out1))
        # glucose queries aux2 (steps)
        out2, _ = self.attn_aux2(glucose, aux2, aux2)
        res2 = self.ln1(glucose + self.dropout(out2))
        # average merge
        merged = 0.5 * (res1 + res2)
        # FFN + residual
        ff = self.ffn(merged)
        return self.ln2(merged + self.dropout(ff))


class MultiScaleAttentionBlock(nn.Module):
    """
    Multi-scale self-attention at 3 resolutions: DS=1, DS=2, DS=4.
    Low-resolution outputs are upsampled back and summed with high-res.
    """

    def __init__(self, d_model: int, n_heads: int, ff_units: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attn_high = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_low2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.attn_low4 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool1d(kernel_size=4, stride=4)

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (seq_len, batch, d_model)."""
        seq_len = x.size(0)

        # DS=1: full resolution self-attention
        high_out, _ = self.attn_high(x, x, x)
        high = self.ln1(x + self.dropout(high_out))

        # Transpose for pooling: (batch, d_model, seq_len)
        xt = high.permute(1, 2, 0)

        # DS=2
        low2 = self.pool2(xt).permute(2, 0, 1)  # (seq/2, batch, d_model)
        low2_out, _ = self.attn_low2(low2, low2, low2)
        up2 = F.interpolate(
            low2_out.permute(1, 2, 0), size=seq_len, mode="nearest"
        ).permute(2, 0, 1)

        # DS=4
        low4 = self.pool4(xt).permute(2, 0, 1)  # (seq/4, batch, d_model)
        low4_out, _ = self.attn_low4(low4, low4, low4)
        up4 = F.interpolate(
            low4_out.permute(1, 2, 0), size=seq_len, mode="nearest"
        ).permute(2, 0, 1)

        # Fuse scales
        fused = high + self.dropout(up2) + self.dropout(up4)
        ff = self.ffn(fused)
        return self.ln2(fused + self.dropout(ff))


class GluMindParallelBlock(nn.Module):
    """
    One GluMind block: cross-attention and multi-scale run IN PARALLEL,
    then outputs are summed (the key architectural difference vs AttenGluco).
    """

    def __init__(self, d_model: int, n_heads: int, ff_units: int,
                 dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(d_model, n_heads, ff_units, dropout)
        self.multiscale = MultiScaleAttentionBlock(d_model, n_heads, ff_units, dropout)
        self.ln_fuse = nn.LayerNorm(d_model)

    def forward(self, glucose: torch.Tensor, aux1: torch.Tensor,
                aux2: torch.Tensor) -> torch.Tensor:
        """All inputs: (seq_len, batch, d_model). Returns same shape."""
        cross_out = self.cross_attn(glucose, aux1, aux2)
        ms_out = self.multiscale(glucose)
        fused = self.ln_fuse(cross_out + ms_out)
        return fused


class GluMindModel(nn.Module):
    """
    GluMind: Multimodal Parallel-Attention Transformer.

    Input:  (batch, seq_len, 3)  — [glucose, HR, steps]
    Output: (batch, horizon)     — predicted glucose values
    """

    def __init__(
        self,
        n_time_steps: int,
        n_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        ff_units: int = 128,
        n_blocks: int = 3,
        prediction_horizon: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_time_steps = n_time_steps
        self.d_model = d_model
        self.n_features = n_features

        # Per-channel linear embeddings
        self.embed_glucose = nn.Linear(1, d_model)
        self.embed_hr = nn.Linear(1, d_model)
        self.embed_steps = nn.Linear(1, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=n_time_steps)

        # Stacked parallel blocks
        self.blocks = nn.ModuleList(
            [
                GluMindParallelBlock(d_model, n_heads, ff_units, dropout)
                for _ in range(n_blocks)
            ]
        )

        # Output head
        self.flatten_fc = nn.Linear(d_model * n_time_steps, d_model)
        self.out_fc = nn.Linear(d_model, prediction_horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, 3)."""
        # 1. Split channels
        g = x[..., 0:1]  # (batch, seq, 1)
        h = x[..., 1:2]
        s = x[..., 2:3]

        # 2. Embed + positional encoding
        g_e = self.pos_enc(self.embed_glucose(g))  # (batch, seq, d_model)
        h_e = self.pos_enc(self.embed_hr(h))
        s_e = self.pos_enc(self.embed_steps(s))

        # 3. Transpose to (seq, batch, d_model) for attention
        g_e = g_e.permute(1, 0, 2)
        h_e = h_e.permute(1, 0, 2)
        s_e = s_e.permute(1, 0, 2)

        # 4. Stacked parallel blocks
        out = g_e
        for block in self.blocks:
            out = block(out, h_e, s_e)

        # 5. Output head: flatten → FC → prediction
        out = out.permute(1, 2, 0)  # (batch, d_model, seq)
        batch_size = out.size(0)
        out = out.reshape(batch_size, -1)  # (batch, d_model * seq)
        out = self.dropout(F.gelu(self.flatten_fc(out)))
        return self.out_fc(out)  # (batch, horizon)
