"""
LSTM Language Model for scaling comparison with Transformers.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LSTMConfig:
    vocab_size: int = 128
    block_size: int = 512  # context length (for consistency with transformer)
    n_layer: int = 2
    n_embd: int = 256  # embedding dimension
    hidden_size: int = 512
    dropout: float = 0.1

    def num_params(self) -> int:
        """Estimate number of parameters."""
        # Embedding
        emb = self.vocab_size * self.n_embd
        # LSTM: 4 * ((input_size + hidden_size) * hidden_size + hidden_size) per layer
        # First layer: input is n_embd
        lstm_first = 4 * ((self.n_embd + self.hidden_size) * self.hidden_size + self.hidden_size)
        # Subsequent layers: input is hidden_size
        lstm_rest = (self.n_layer - 1) * 4 * ((self.hidden_size + self.hidden_size) * self.hidden_size + self.hidden_size)
        # Output projection
        proj = self.hidden_size * self.vocab_size
        return emb + lstm_first + lstm_rest + proj


class LSTMLM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.hidden_size,
            num_layers=config.n_layer,
            batch_first=True,
            dropout=config.dropout if config.n_layer > 1 else 0,
        )
        self.ln = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    torch.nn.init.orthogonal_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, hidden: tuple | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()

        x = self.embedding(idx)
        x = self.drop(x)
        x, hidden = self.lstm(x, hidden)
        x = self.ln(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        hidden = None
        # Process the prompt
        if idx.size(1) > 1:
            x = self.embedding(idx[:, :-1])
            x = self.drop(x)
            _, hidden = self.lstm(x, hidden)
            idx = idx[:, -1:]

        for _ in range(max_new_tokens):
            x = self.embedding(idx[:, -1:])
            x = self.drop(x)
            x, hidden = self.lstm(x, hidden)
            x = self.ln(x)
            logits = self.lm_head(x[:, -1, :]) / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Predefined configurations for scaling experiments (aligned ~2× transformer sizes)
LSTM_CONFIGS = {
    "tiny": LSTMConfig(n_layer=1, n_embd=192, hidden_size=384),        # ~1M
    "small": LSTMConfig(n_layer=2, n_embd=256, hidden_size=512),       # ~5M
    "medium": LSTMConfig(n_layer=3, n_embd=512, hidden_size=1024),     # ~20M
    "large": LSTMConfig(n_layer=4, n_embd=640, hidden_size=1280),      # ~50M
    "xl": LSTMConfig(n_layer=4, n_embd=896, hidden_size=1792),         # ~100M
}
