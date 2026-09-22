"""A small, interpretable causal Transformer.

Deliberately minimal: 2 attention-only layers are enough to grow an induction
head on this task, and the smaller the model the easier the circuit is to read.
forward(..., return_attn=True) exposes the attention pattern of every head in
every layer, which is what the probes in probes.py consume.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope_cache(T, d_head, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv_freq)                          # (T, d_head/2)
    return freqs.cos()[None, None], freqs.sin()[None, None]   # (1,1,T,d_head/2)


def _apply_rope(x, cos, sin):
    # x: (B, H, T, d_head). Rotary encoding makes q.k depend on RELATIVE position,
    # which is what lets a previous-token head ("attend one back") form easily.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack([xr1, xr2], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # each: (B, H, T, d_head)
        if self.use_rope:
            cos, sin = _rope_cache(T, self.d_head, x.device)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B, H, T, T)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf")).softmax(dim=-1)
        out = att @ v                                        # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out), att


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, use_rope=getattr(cfg, "use_rope", True))
        self.use_mlp = cfg.use_mlp
        if cfg.use_mlp:
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.mlp = MLP(cfg.d_model, cfg.d_mlp)

    def forward(self, x):
        a, att = self.attn(self.ln1(x))
        x = x + a
        if self.use_mlp:
            x = x + self.mlp(self.ln2(x))
        return x, att


class InductionTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.lnf = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        if getattr(cfg, "tie_embeddings", True):
            self.unembed.weight = self.tok.weight            # ties embed<->unembed: aids copying

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, return_attn: bool = False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        attns = []
        for blk in self.blocks:
            x, att = blk(x)
            attns.append(att)                                # (B, H, T, T) per layer
        logits = self.unembed(self.lnf(x))                   # (B, T, vocab)
        if return_attn:
            return logits, attns
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
