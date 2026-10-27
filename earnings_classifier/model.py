"""Model architecture.

  FinBERTEncoder : shared FinBERT, encodes a *bag* of sequences (document
                   chunks or headlines) into one pooled 768-d vector per event,
                   with mask-aware pooling over both tokens and bag elements.
  FusionHead     : fuses the document and news vectors and classifies. With the
                   default head_layers=0 it is dropout + ONE linear layer over
                   [t, n, |t-n|, t*n, presence flags]: a logistic-regression head
                   on fixed, expanded features (3,075 parameters for 768-d inputs).
  EndToEndModel  : encoder + head, for optional fine-tuning of FinBERT.

The frozen default trains FusionHead alone on precomputed embeddings.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .config import Config


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------
def masked_mean_tokens(last_hidden: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """[N, L, H], [N, L] -> [N, H] mean over non-pad tokens."""
    m = attn.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1.0)
    return summed / counts


def masked_mean_bag(emb: torch.Tensor, n_real: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """[B, K, H], [B] -> ([B, H] mean over real elements, [B] presence flag)."""
    B, K, _ = emb.shape
    idx = torch.arange(K, device=emb.device).unsqueeze(0)        # [1, K]
    row_mask = (idx < n_real.unsqueeze(1)).to(emb.dtype)         # [B, K]
    rm = row_mask.unsqueeze(-1)                                  # [B, K, 1]
    summed = (emb * rm).sum(dim=1)
    counts = rm.sum(dim=1).clamp(min=1.0)
    bag = summed / counts
    present = (n_real > 0).to(emb.dtype)                         # [B]
    return bag * present.unsqueeze(1), present


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class FinBERTEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import AutoModel

        from .utils import quiet_transformers

        quiet_transformers()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.model_name, revision=cfg.model_revision)
        loaded = getattr(self.backbone.config, "_commit_hash", None)
        if loaded and loaded != cfg.model_revision:
            raise RuntimeError(f"{cfg.model_name}: loaded revision {loaded} != pinned "
                               f"{cfg.model_revision}")
        self.hidden_size = self.backbone.config.hidden_size
        if cfg.freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def encode_bag(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, n_real: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, K, L] tokens -> ([B, H] pooled vector, [B] presence)."""
        B, K, L = input_ids.shape
        flat_ids = input_ids.reshape(B * K, L)
        flat_attn = attention_mask.reshape(B * K, L)
        out = self.backbone(input_ids=flat_ids, attention_mask=flat_attn)
        last_hidden = out.last_hidden_state                      # [B*K, L, H]
        if self.cfg.pooling == "cls":
            seq_vec = last_hidden[:, 0]
        else:
            seq_vec = masked_mean_tokens(last_hidden, flat_attn)
        seq_vec = seq_vec.reshape(B, K, self.hidden_size)        # [B, K, H]
        return masked_mean_bag(seq_vec, n_real)


# ---------------------------------------------------------------------------
# Fusion head
# ---------------------------------------------------------------------------
class FusionHead(nn.Module):
    """Fuse document + news embeddings and classify up/down."""

    def __init__(self, cfg: Config, embed_dim: int):
        super().__init__()
        self.cfg = cfg
        in_dim = embed_dim * cfg.fused_dim_multiplier
        if cfg.use_presence_flags:
            in_dim += 2

        layers = []
        d = in_dim
        for _ in range(cfg.head_layers):
            layers += [nn.Linear(d, cfg.head_hidden), nn.ReLU(), nn.Dropout(cfg.head_dropout)]
            d = cfg.head_hidden
        layers += [nn.Dropout(cfg.head_dropout), nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def fuse(
        self,
        t_emb: torch.Tensor,
        n_emb: torch.Tensor,
        t_present: torch.Tensor,
        n_present: torch.Tensor,
    ) -> torch.Tensor:
        # Zero-out absent modalities so they contribute nothing to concat/product.
        t = t_emb * t_present.unsqueeze(1)
        n = n_emb * n_present.unsqueeze(1)
        parts = [t, n]
        if self.cfg.use_interactions:
            parts += [(t - n).abs(), t * n]
        fused = torch.cat(parts, dim=1)
        if self.cfg.use_presence_flags:
            fused = torch.cat([fused, t_present.unsqueeze(1), n_present.unsqueeze(1)], dim=1)
        return fused

    def forward(
        self,
        t_emb: torch.Tensor,
        n_emb: torch.Tensor,
        t_present: torch.Tensor,
        n_present: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.fuse(t_emb, n_emb, t_present, n_present)
        return self.mlp(fused).squeeze(-1)   # [B] logits


# ---------------------------------------------------------------------------
# End-to-end model (encoder + head)
# ---------------------------------------------------------------------------
class EndToEndModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = FinBERTEncoder(cfg)
        self.head = FusionHead(cfg, embed_dim=self.encoder.hidden_size)

    def forward(self, batch: dict) -> torch.Tensor:
        t_emb, t_present = self.encoder.encode_bag(
            batch["t_input_ids"], batch["t_attention_mask"], batch["t_n_real"]
        )
        n_emb, n_present = self.encoder.encode_bag(
            batch["n_input_ids"], batch["n_attention_mask"], batch["n_n_real"]
        )
        return self.head(t_emb, n_emb, t_present, n_present)
