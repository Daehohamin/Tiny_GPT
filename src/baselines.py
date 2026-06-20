from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramNextChar(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.table(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def sample(self, start_id: int, steps: int) -> list[int]:
        ids = [start_id]
        cur = torch.tensor([start_id], dtype=torch.long)
        for _ in range(steps):
            logits, _ = self(cur)
            probs = F.softmax(logits[-1], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            ids.append(int(nxt.item()))
            cur = nxt
        return ids


class MLPNextChar(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 3, n_embd: int = 16, hidden: int = 64):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.net = nn.Sequential(
            nn.Linear(block_size * n_embd, hidden),
            nn.Tanh(),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        emb = self.embedding(x)
        flat = emb.view(emb.shape[0], -1)
        logits = self.net(flat)
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss, emb, flat


class SequenceNoAttention(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 8, n_embd: int = 24):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.ff = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU(), nn.Linear(n_embd, vocab_size))

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = x.shape
        tok = self.token_embedding(x)
        pos = self.position_embedding(torch.arange(t, device=x.device))
        logits = self.ff(tok + pos)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(b * t, -1), targets.view(b * t))
        return logits, loss


class SingleHeadMaskedSelfAttention(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.scale = head_size**-0.5

    def forward(self, x: torch.Tensor):
        _, t, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) * self.scale
        scores = scores.masked_fill(self.tril[:t, :t] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ v, weights
