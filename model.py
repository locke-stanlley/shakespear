"""
model.py - Phase 4: Architectural Design
Decoder-only Transformer with:
  - RMSNorm (replaces LayerNorm)
  - Rotary Position Embeddings (RoPE)
  - Multi-Head Attention with causal mask
  - SwiGLU-style MLP (GELU approximation)
  - Proper weight initialization scaled by depth
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_LAYER, N_HEAD, N_EMBD, HEAD_DIM, MLP_HIDDEN,
    BLOCK_SIZE, DROPOUT, USE_ROPE, ROPE_THETA, VOCAB_SIZE
)


# -----------------------------
# RMSNorm
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# -----------------------------
# RoPE
# -----------------------------
def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute cos/sin tables for Rotary Position Embeddings."""
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)                 # (T, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    # Stack to (T, head_dim) by repeating
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the input tensor by the precomputed RoPE tables.
    x: (B, nh, T, hd)
    """
    # Split into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    # cos/sin broadcast over B, nh
    return x * cos + rotated * sin


# -----------------------------
# Attention
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, use_rope: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope

        # Combined QKV projection
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_cache) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        # Reshape for multi-head: (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = rope_cache
            cos, sin = cos[:T], sin[:T]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return y


# -----------------------------
# MLP (SwiGLU-like with GELU)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, n_embd: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, n_embd, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# -----------------------------
# Transformer Block
# -----------------------------
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, hidden: int, dropout: float, use_rope: bool):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, use_rope)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, hidden, dropout)

    def forward(self, x: torch.Tensor, rope_cache) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cache)
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# Full Model
# -----------------------------
@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = BLOCK_SIZE
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD
    mlp_hidden: int = MLP_HIDDEN
    dropout: float = DROPOUT
    use_rope: bool = USE_ROPE
    rope_theta: float = ROPE_THETA


class ShakespeareGPT(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.n_embd, cfg.n_head, cfg.mlp_hidden, cfg.dropout, cfg.use_rope)
            for _ in range(cfg.n_layer)
        ])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Tie weights between token embedding and lm_head
        self.lm_head.weight = self.token_emb.weight

        # Precompute RoPE cache (register as buffer so it moves with .to(device))
        if cfg.use_rope:
            cos, sin = precompute_rope_cache(
                cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(depth)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Sequence too long: {T} > {self.cfg.block_size}"

        x = self.drop(self.token_emb(idx))

        rope_cache = (self.rope_cos, self.rope_sin) if self.cfg.use_rope else None
        for block in self.blocks:
            x = block(x, rope_cache)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ShakespeareGPT()
    print(f"Parameters: {count_parameters(model):,}")
    x = torch.randint(0, VOCAB_SIZE, (2, 64))
    y = torch.randint(0, VOCAB_SIZE, (2, 64))
    logits, loss = model(x, y)
    print(f"logits: {tuple(logits.shape)}, loss: {loss.item():.4f}")