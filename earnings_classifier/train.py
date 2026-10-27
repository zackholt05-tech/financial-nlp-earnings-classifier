"""Training loop and model selection.

Two paths share one engine:
  * frozen (default) : train FusionHead on precomputed embeddings.
  * end-to-end       : fine-tune FinBERT + head on raw token bags.

Model selection is by validation AUC (primary metric); accuracy is too coarse
on small, class-imbalanced test sets.
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .evaluate import compute_metrics
from .utils import autocast_dtype, get_logger

log = get_logger()


# ---------------------------------------------------------------------------
# Optimizer / schedule
# ---------------------------------------------------------------------------
def build_param_groups(model: nn.Module, cfg: Config) -> List[Dict]:
    """Separate weight-decay vs. no-decay, and encoder-lr vs. head-lr."""
    groups: Dict[str, Dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_encoder = name.startswith("encoder.")
        no_decay = p.ndim == 1 or name.endswith(".bias")
        key = f"{'enc' if is_encoder else 'head'}_{'nd' if no_decay else 'wd'}"
        groups.setdefault(
            key,
            {
                "params": [],
                "lr": cfg.encoder_lr if is_encoder else cfg.head_lr,
                "weight_decay": 0.0 if no_decay else cfg.weight_decay,
            },
        )["params"].append(p)
    return list(groups.values())


def cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def pos_weight_from_labels(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    w = neg / pos if pos > 0 else 1.0
    return torch.tensor(w, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Batch movement / evaluation
# ---------------------------------------------------------------------------
def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            forward_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        logits = forward_fn(model, batch)
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(batch["label"].float().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: np.ndarray,
    cfg: Config,
    device: torch.device,
    forward_fn: Callable,
) -> Tuple[nn.Module, Dict]:
    y_val_all = np.concatenate([np.asarray(b["label"]) for b in val_loader]) \
        if len(val_loader) else np.array([])
    if len(np.unique(y_val_all)) < 2:
        raise ValueError(
            f"validation split has {len(y_val_all)} events and {len(np.unique(y_val_all))} "
            "class(es); AUC-based model selection is undefined. Fix the split instead of "
            "selecting on a degenerate validation set."
        )
    model.to(device)
    param_groups = build_param_groups(model, cfg)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    total_steps = max(1, len(train_loader) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_frac)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_warmup(s, warmup_steps, total_steps)
    )

    pw = pos_weight_from_labels(train_labels, device) if cfg.use_pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    amp_dtype = autocast_dtype(device)

    best_auc, best_state, best_epoch = -1.0, copy.deepcopy(model.state_dict()), 0
    patience = 0
    history: List[Dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = forward_fn(model, batch)
                    loss = criterion(logits, batch["label"].float())
            else:
                logits = forward_fn(model, batch)
                loss = criterion(logits, batch["label"].float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), cfg.grad_clip
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        y_val, p_val = predict(model, val_loader, device, forward_fn)
        m = compute_metrics(y_val, p_val)
        val_auc = m["auc"]  # defined: both classes are guaranteed present above
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(1, n_batches),
                        "val_auc": m["auc"], "val_log_loss": m["log_loss"],
                        "val_acc": m["accuracy"]})
        log.info(
            "epoch %2d | loss %.4f | val AUC %.3f | val acc %.3f",
            epoch, epoch_loss / max(1, n_batches), m["auc"], m["accuracy"],
        )

        if val_auc > best_auc + 1e-4:
            best_auc, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                log.info("Early stopping at epoch %d (best epoch %d, AUC %.3f).",
                         epoch, best_epoch, best_auc)
                break

    model.load_state_dict(best_state)
    return model, {"history": history, "selection_metric": "val_auc",
                   "best_val_auc": best_auc, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# Sanity: overfit a single batch (should drive train AUC -> ~1.0)
# ---------------------------------------------------------------------------
def overfit_one_batch(model: nn.Module, batch: Dict, cfg: Config,
                      device: torch.device, forward_fn: Callable, steps: int = 200) -> float:
    """Karpathy sanity check: if a model can't overfit one batch, it's buggy."""
    model.to(device).train()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.head_lr
    )
    criterion = nn.BCEWithLogitsLoss()
    batch = _to_device(batch, device)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = forward_fn(model, batch)
        loss = criterion(logits, batch["label"].float())
        loss.backward()
        opt.step()
    with torch.no_grad():
        prob = torch.sigmoid(forward_fn(model, batch)).cpu().numpy()
    return float(compute_metrics(batch["label"].cpu().numpy(), prob)["accuracy"])


# Forward-function adapters for the two paths.
def frozen_forward(model: nn.Module, batch: Dict) -> torch.Tensor:
    return model(batch["t_emb"], batch["n_emb"], batch["t_present"], batch["n_present"])


def e2e_forward(model: nn.Module, batch: Dict) -> torch.Tensor:
    return model(batch)
