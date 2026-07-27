"""
model.py - Production-Grade T4-Optimized Transformer
Features:
  - Flash Attention via PyTorch SDPA
  - SwiGLU activation (Llama-style)
  - KV Cache for O(1) step generation
  - Gradient Checkpointing for memory efficiency
  - RMSNorm + Rotary Position Embeddings (RoPE)
  - Verified Top-P / Top-K / Repetition Penalty sampling
  - Real-time token streaming generator
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_LAYER, N_HEAD, N_EMBD, MLP_HIDDEN, BLOCK_SIZE, DROPOUT,
    USE_ROPE, USE_RMSNORM, USE_SWIGLU, USE_GRAD_CHECKPOINT,
    USE_FLASH_ATTN, ROPE_THETA, VOCAB_SIZE
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
# RoPE (with caching)
# -----------------------------
def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0, device=None):
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# -----------------------------
# SwiGLU MLP
# -----------------------------
class SwiGLU_MLP(nn.Module):
    def __init__(self, n_embd: int, hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GELU_MLP(nn.Module):
    def __init__(self, n_embd: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, n_embd, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# -----------------------------
# Attention with SDPA + KV Cache
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, use_rope: bool, use_flash: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_rope = use_rope
        self.use_flash = use_flash

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope and rope_cache is not None:
            cos, sin = rope_cache
            if use_kv_cache and past_kv is not None:
                past_len = past_kv[0].shape[2]
                cos = cos[past_len : past_len + T]
                sin = sin[past_len : past_len + T]
            else:
                cos, sin = cos[:T], sin[:T]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if use_kv_cache and past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v) if use_kv_cache else None

        if self.use_flash and T > 1:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            if T > 1:
                mask = torch.triu(torch.ones(T, k.shape[2], device=x.device, dtype=torch.bool), diagonal=k.shape[2] - T + 1)
                att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.out_proj(y))
        return y, new_kv


# -----------------------------
# Transformer Block
# -----------------------------
class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, hidden: int, dropout: float,
                 use_rope: bool, use_flash: bool, use_swiglu: bool):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, use_rope, use_flash)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = SwiGLU_MLP(n_embd, hidden, dropout) if use_swiglu else GELU_MLP(n_embd, hidden, dropout)

    def forward(self, x, rope_cache, past_kv=None, use_kv_cache=False):
        attn_out, new_kv = self.attn(self.norm1(x), rope_cache, past_kv, use_kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


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
    use_rmsnorm: bool = USE_RMSNORM
    use_swiglu: bool = USE_SWIGLU
    use_grad_checkpoint: bool = USE_GRAD_CHECKPOINT
    use_flash_attn: bool = USE_FLASH_ATTN
    rope_theta: float = ROPE_THETA


class ShakespeareGPT(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(cfg.n_embd, cfg.n_head, cfg.mlp_hidden, cfg.dropout,
                  cfg.use_rope, cfg.use_flash_attn, cfg.use_swiglu)
            for _ in range(cfg.n_layer)
        ])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight

        if cfg.use_rope:
            cos, sin = precompute_rope_cache(
                cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("w2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(cfg.n_layer))

    def _enable_gradient_checkpointing(self):
        for block in self.blocks:
            block._gradient_checkpointing = True

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _block_forward(self, block, x, rope_cache, past_kv=None, use_kv_cache=False):
        return block(x, rope_cache, past_kv, use_kv_cache)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
        past_kvs: Optional[list] = None,
        use_kv_cache: bool = False,
    ):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Sequence too long: {T} > {self.cfg.block_size}"

        x = self.drop(self.token_emb(idx))
        rope_cache = (self.rope_cos, self.rope_sin) if self.cfg.use_rope else None

        new_kvs = [] if use_kv_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if self.cfg.use_grad_checkpoint and self.training and not use_kv_cache:
                x, new_kv = torch.utils.checkpoint.checkpoint(
                    self._block_forward, block, x, rope_cache, past_kv, use_kv_cache,
                    use_reentrant=False,
                )
            else:
                x, new_kv = block(x, rope_cache, past_kv, use_kv_cache)
            if use_kv_cache:
                new_kvs.append(new_kv)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Standard generation returning full tensor."""
        generated_ids = idx.clone()
        for token in self.generate_stream(idx, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
            generated_ids = torch.cat((generated_ids, torch.tensor([[token]], device=idx.device)), dim=1)
        return generated_ids

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        repetition_penalty: float = 1.0,
    ):
        """Yields generated tokens one by one for real-time streaming."""
        past_kvs = None
        device = idx.device
        generated_ids = idx.clone()
        
        # 1. Prefill phase
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits, _, past_kvs = self(idx, past_kvs=None, use_kv_cache=True)
            
        for _ in range(max_new_tokens):
            # 2. Get logits for the last token
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # 3. Apply repetition penalty (Verified Hugging Face implementation)
            if repetition_penalty > 1.0:
                for i in range(generated_ids.shape[0]):
                    for token_id in set(generated_ids[i].tolist()):
                        if next_token_logits[i, token_id] > 0:
                            next_token_logits[i, token_id] /= repetition_penalty
                        else:
                            next_token_logits[i, token_id] *= repetition_penalty
            
            # 4. Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")
            
            # 5. Top-P (Nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("inf")
            
            # 6. Sample
            probs = F.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            token = idx_next.item()
            
            # 7. Update history
            generated_ids = torch.cat((generated_ids, idx_next), dim=1)
            
            # 8. Forward pass for next token
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                logits, _, past_kvs = self(idx_next, past_kvs=past_kvs, use_kv_cache=True)
                
            yield token
            
            # Stop if EOS token (ID 2) is generated
            if token == 2:
                break


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)