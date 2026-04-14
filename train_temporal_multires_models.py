from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


RANDOM_STATE = 42

DEFAULT_BINARY_TARGETS = [
    "y_next_weight_gain_flag",
    "y_next_weight_loss_flag",
]
DEFAULT_REGRESSION_TARGETS = [
    "y_next_weight_delta_lb",
]


def log(msg: str) -> None:
    print(f"[temporal-multires] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_npz_bundle(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def align_bundle_to_anchors(
    anchors_df: pd.DataFrame,
    bundle: Dict[str, np.ndarray],
    name: str,
) -> Dict[str, np.ndarray]:
    anchor_ids = anchors_df["anchor_id"].astype(str).to_numpy()
    bundle_ids = np.array(bundle["anchor_ids"]).astype(str)
    pos = {aid: i for i, aid in enumerate(bundle_ids)}
    missing = [aid for aid in anchor_ids if aid not in pos]
    if missing:
        raise ValueError(f"{name}: missing {len(missing)} anchor_ids from bundle alignment. Example: {missing[:5]}")
    idx = np.array([pos[aid] for aid in anchor_ids], dtype=int)
    out = {}
    for k, v in bundle.items():
        if k == "feature_names":
            out[k] = v
        else:
            out[k] = v[idx]
    return out


def compute_masked_mean_std(X: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # X: [N, T, F], mask: [N, T]
    if X.size == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    m = mask[..., None].astype(np.float32)
    denom = np.maximum(m.sum(axis=(0, 1)), 1.0)
    mean = (X * m).sum(axis=(0, 1)) / denom
    var = (((X - mean[None, None, :]) * m) ** 2).sum(axis=(0, 1)) / denom
    std = np.sqrt(np.maximum(var, 1e-8))
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_modality(
    X: np.ndarray,
    mask: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if X.size == 0:
        return X.astype(np.float32), {"mean": np.zeros((0,), dtype=np.float32), "std": np.ones((0,), dtype=np.float32)}
    mean, std = compute_masked_mean_std(X[train_idx], mask[train_idx])
    Xn = (X - mean[None, None, :]) / std[None, None, :]
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Xn, {"mean": mean, "std": std}


def standardize_age(
    age: np.ndarray,
    mask: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if age.size == 0:
        return age.astype(np.float32), {"mean": 0.0, "std": 1.0}
    vals = age[train_idx][mask[train_idx] > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        mean, std = 0.0, 1.0
    else:
        mean = float(vals.mean())
        std = float(vals.std())
        if std < 1e-6:
            std = 1.0
    out = (age - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, {"mean": mean, "std": std}


def choose_threshold(y_true: np.ndarray, prob: np.ndarray, metric: str = "balanced_accuracy") -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.unique(np.round(np.concatenate([np.linspace(0.05, 0.95, 19), prob]), 4))
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (prob >= t).astype(int)
        if metric == "macro_f1":
            score = f1_score(y_true, pred, average="binary", zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = float(score)
            best_t = float(t)
    return best_t


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(pred)),
    }
    if len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, prob))
        except Exception:
            pass
    return out


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, pred)),
    }


class MultiResSequenceDataset(Dataset):
    def __init__(
        self,
        anchor_ids: np.ndarray,
        split: np.ndarray,
        modalities: Dict[str, Dict[str, np.ndarray]],
        static_feats: np.ndarray,
        binary_targets: Dict[str, np.ndarray],
        binary_masks: Dict[str, np.ndarray],
        regression_targets: Dict[str, np.ndarray],
        regression_masks: Dict[str, np.ndarray],
    ) -> None:
        self.anchor_ids = anchor_ids
        self.split = split
        self.modalities = modalities
        self.static_feats = static_feats.astype(np.float32)
        self.binary_targets = binary_targets
        self.binary_masks = binary_masks
        self.regression_targets = regression_targets
        self.regression_masks = regression_masks

    def __len__(self) -> int:
        return len(self.anchor_ids)

    def __getitem__(self, idx: int) -> Dict:
        item = {
            "anchor_id": self.anchor_ids[idx],
            "split": self.split[idx],
            "static_feats": torch.tensor(self.static_feats[idx], dtype=torch.float32),
            "binary_targets": {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.binary_targets.items()},
            "binary_masks": {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.binary_masks.items()},
            "regression_targets": {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.regression_targets.items()},
            "regression_masks": {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in self.regression_masks.items()},
        }
        for name, bundle in self.modalities.items():
            item[f"{name}_x"] = torch.tensor(bundle["X"][idx], dtype=torch.float32)
            item[f"{name}_mask"] = torch.tensor(bundle["mask"][idx], dtype=torch.float32)
            item[f"{name}_age"] = torch.tensor(bundle["age_days"][idx], dtype=torch.float32)
        return item


def collate_batch(batch: List[Dict]) -> Dict:
    out: Dict = {
        "anchor_id": [b["anchor_id"] for b in batch],
        "split": [b["split"] for b in batch],
        "static_feats": torch.stack([b["static_feats"] for b in batch], dim=0),
        "binary_targets": {},
        "binary_masks": {},
        "regression_targets": {},
        "regression_masks": {},
    }
    keys = [k for k in batch[0].keys() if k.endswith("_x") or k.endswith("_mask") or k.endswith("_age")]
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)

    for k in batch[0]["binary_targets"].keys():
        out["binary_targets"][k] = torch.stack([b["binary_targets"][k] for b in batch], dim=0)
        out["binary_masks"][k] = torch.stack([b["binary_masks"][k] for b in batch], dim=0)

    for k in batch[0]["regression_targets"].keys():
        out["regression_targets"][k] = torch.stack([b["regression_targets"][k] for b in batch], dim=0)
        out["regression_masks"][k] = torch.stack([b["regression_masks"][k] for b in batch], dim=0)
    return out


class ResidualTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ModalityEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        model_family: str,
        num_layers: int,
        dropout: float,
        seq_len_hint: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model_family = model_family
        self.seq_len_hint = seq_len_hint

        if input_dim <= 0:
            self.empty = True
            return
        self.empty = False

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        if model_family == "gru":
            self.encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif model_family == "tcn":
            self.encoder = nn.Sequential(*[
                ResidualTCNBlock(hidden_dim=hidden_dim, kernel_size=3, dropout=dropout)
                for _ in range(num_layers)
            ])
        elif model_family == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, min(8, hidden_dim // 32 if hidden_dim >= 32 else 1)),
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unsupported model_family: {model_family}")

        self.post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, age_days: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F], mask: [B, T], age_days: [B, T]
        B = x.size(0)
        device = x.device
        if self.empty or x.numel() == 0 or x.size(-1) == 0:
            return torch.zeros(B, self.hidden_dim, device=device)

        age_feat = age_days.unsqueeze(-1)
        h = self.input_proj(torch.cat([x, age_feat], dim=-1))

        if self.model_family == "gru":
            out, _ = self.encoder(h)
        elif self.model_family == "tcn":
            z = h.transpose(1, 2)
            out = self.encoder(z).transpose(1, 2)
        else:  # transformer
            src_key_padding_mask = mask <= 0
            out = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.post(pooled)


class TemporalMultiResModel(nn.Module):
    def __init__(
        self,
        day_input_dim: int,
        meal_input_dim: int,
        week_input_dim: int,
        static_input_dim: int,
        model_family: str,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        binary_heads: List[str],
        regression_heads: List[str],
        day_seq_len: int,
        meal_seq_len: int,
        week_seq_len: int,
    ) -> None:
        super().__init__()
        self.day_encoder = ModalityEncoder(day_input_dim, hidden_dim, model_family, num_layers, dropout, day_seq_len)
        self.meal_encoder = ModalityEncoder(meal_input_dim, hidden_dim, model_family, num_layers, dropout, meal_seq_len)
        self.week_encoder = ModalityEncoder(week_input_dim, hidden_dim, model_family, num_layers, dropout, week_seq_len)

        self.static_proj = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        fused_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.binary_heads = nn.ModuleDict({k: nn.Linear(hidden_dim, 1) for k in binary_heads})
        self.regression_heads = nn.ModuleDict({k: nn.Linear(hidden_dim, 1) for k in regression_heads})

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        day_h = self.day_encoder(batch["days_x"], batch["days_mask"], batch["days_age"])
        meal_h = self.meal_encoder(batch["meals_x"], batch["meals_mask"], batch["meals_age"])
        week_h = self.week_encoder(batch["weeks_x"], batch["weeks_mask"], batch["weeks_age"])
        static_h = self.static_proj(batch["static_feats"])

        fused = self.fusion(torch.cat([day_h, meal_h, week_h, static_h], dim=-1))

        out = {"binary": {}, "regression": {}}
        for k, head in self.binary_heads.items():
            out["binary"][k] = head(fused).squeeze(-1)
        for k, head in self.regression_heads.items():
            out["regression"][k] = head(fused).squeeze(-1)
        return out


@dataclass
class TargetStats:
    mean: float
    std: float


def compute_regression_target_stats(values: np.ndarray, mask: np.ndarray) -> TargetStats:
    vals = values[mask > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return TargetStats(0.0, 1.0)
    mean = float(vals.mean())
    std = float(vals.std())
    if std < 1e-6:
        std = 1.0
    return TargetStats(mean, std)


def build_datasets(
    dataset_dir: Path,
    binary_targets: List[str],
    regression_targets: List[str],
) -> Tuple[MultiResSequenceDataset, Dict]:
    anchors = pd.read_csv(dataset_dir / "anchors.csv", low_memory=False)
    masks_df = pd.read_csv(dataset_dir / "modality_masks.csv", low_memory=False)

    day_bundle = load_npz_bundle(dataset_dir / "days_numeric_sequences.npz")
    meal_bundle = load_npz_bundle(dataset_dir / "meals_numeric_sequences.npz")
    week_bundle = load_npz_bundle(dataset_dir / "weeks_numeric_sequences.npz")

    anchors["anchor_id"] = anchors["anchor_id"].astype(str)
    masks_df["anchor_id"] = masks_df["anchor_id"].astype(str)
    anchors = anchors.merge(masks_df, on="anchor_id", how="left")

    day_bundle = align_bundle_to_anchors(anchors, day_bundle, "days")
    meal_bundle = align_bundle_to_anchors(anchors, meal_bundle, "meals")
    week_bundle = align_bundle_to_anchors(anchors, week_bundle, "weeks")

    split = anchors["split_suggested"].astype(str).to_numpy()
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Train/val/test splits are missing from anchors.csv")

    # Standardize modalities using train split only.
    day_X, day_stats = standardize_modality(day_bundle["X"], day_bundle["mask"], train_idx)
    day_age, day_age_stats = standardize_age(day_bundle["age_days"], day_bundle["mask"], train_idx)

    meal_X, meal_stats = standardize_modality(meal_bundle["X"], meal_bundle["mask"], train_idx)
    meal_age, meal_age_stats = standardize_age(meal_bundle["age_days"], meal_bundle["mask"], train_idx)

    week_X, week_stats = standardize_modality(week_bundle["X"], week_bundle["mask"], train_idx)
    week_age, week_age_stats = standardize_age(week_bundle["age_days"], week_bundle["mask"], train_idx)

    modalities = {
        "days": {"X": day_X, "mask": day_bundle["mask"].astype(np.float32), "age_days": day_age},
        "meals": {"X": meal_X, "mask": meal_bundle["mask"].astype(np.float32), "age_days": meal_age},
        "weeks": {"X": week_X, "mask": week_bundle["mask"].astype(np.float32), "age_days": week_age},
    }

    static_cols = [
        "has_meals", "has_days", "has_weeks",
        "n_meals_steps_observed", "n_days_steps_observed", "n_weeks_steps_observed",
    ]
    for col in static_cols:
        if col not in anchors.columns:
            anchors[col] = 0.0
    static_feats = anchors[static_cols].copy()
    static_feats["n_meals_steps_observed"] = static_feats["n_meals_steps_observed"] / max(float(meal_bundle["mask"].shape[1]), 1.0)
    static_feats["n_days_steps_observed"] = static_feats["n_days_steps_observed"] / max(float(day_bundle["mask"].shape[1]), 1.0)
    static_feats["n_weeks_steps_observed"] = static_feats["n_weeks_steps_observed"] / max(float(week_bundle["mask"].shape[1]) if week_bundle["mask"].ndim == 2 else 1.0, 1.0)
    static_feats = static_feats.fillna(0.0).astype(np.float32).to_numpy()

    binary_target_dict: Dict[str, np.ndarray] = {}
    binary_mask_dict: Dict[str, np.ndarray] = {}
    pos_weight_map: Dict[str, float] = {}

    for t in binary_targets:
        if t not in anchors.columns:
            raise ValueError(f"Missing binary target in anchors.csv: {t}")
        series = anchors[t].astype("boolean")
        mask = series.notna().to_numpy().astype(np.float32)
        vals = series.astype("float").fillna(0.0).to_numpy(dtype=np.float32)
        binary_target_dict[t] = vals
        binary_mask_dict[t] = mask
        train_vals = vals[train_idx][mask[train_idx] > 0]
        if train_vals.size == 0:
            pos_weight_map[t] = 1.0
        else:
            pos = float(train_vals.sum())
            neg = float(train_vals.shape[0] - pos)
            pos_weight_map[t] = float(neg / max(pos, 1.0))

    regression_target_dict: Dict[str, np.ndarray] = {}
    regression_mask_dict: Dict[str, np.ndarray] = {}
    regression_stats: Dict[str, Dict[str, float]] = {}

    for t in regression_targets:
        if t not in anchors.columns:
            raise ValueError(f"Missing regression target in anchors.csv: {t}")
        series = pd.to_numeric(anchors[t], errors="coerce")
        mask = series.notna().to_numpy().astype(np.float32)
        vals = series.fillna(0.0).to_numpy(dtype=np.float32)
        stats = compute_regression_target_stats(vals[train_idx], mask[train_idx])
        vals_scaled = ((vals - stats.mean) / stats.std).astype(np.float32)
        regression_target_dict[t] = vals_scaled
        regression_mask_dict[t] = mask
        regression_stats[t] = {"mean": stats.mean, "std": stats.std}

    dataset = MultiResSequenceDataset(
        anchor_ids=anchors["anchor_id"].astype(str).to_numpy(),
        split=split,
        modalities=modalities,
        static_feats=static_feats,
        binary_targets=binary_target_dict,
        binary_masks=binary_mask_dict,
        regression_targets=regression_target_dict,
        regression_masks=regression_mask_dict,
    )

    meta = {
        "n_rows": int(len(anchors)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "day_feature_names": day_bundle["feature_names"].astype(str).tolist(),
        "meal_feature_names": meal_bundle["feature_names"].astype(str).tolist(),
        "week_feature_names": week_bundle["feature_names"].astype(str).tolist(),
        "pos_weight_map": pos_weight_map,
        "regression_stats": regression_stats,
        "standardization": {
            "days": {"feature_mean": day_stats["mean"].tolist(), "feature_std": day_stats["std"].tolist(), **day_age_stats},
            "meals": {"feature_mean": meal_stats["mean"].tolist(), "feature_std": meal_stats["std"].tolist(), **meal_age_stats},
            "weeks": {"feature_mean": week_stats["mean"].tolist(), "feature_std": week_stats["std"].tolist(), **week_age_stats},
        },
        "seq_shapes": {
            "days_X": list(day_bundle["X"].shape),
            "meals_X": list(meal_bundle["X"].shape),
            "weeks_X": list(week_bundle["X"].shape),
        },
    }
    return dataset, meta


def split_dataset_indices(dataset: MultiResSequenceDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = np.array(dataset.split)
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    return train_idx, val_idx, test_idx


def make_loaders(
    dataset: MultiResSequenceDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_idx, val_idx, test_idx = split_dataset_indices(dataset)
    train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_ds = torch.utils.data.Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_batch, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_batch, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_batch, drop_last=False,
    )
    return train_loader, val_loader, test_loader


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {
        "anchor_id": batch["anchor_id"],
        "split": batch["split"],
        "static_feats": batch["static_feats"].to(device, non_blocking=True),
        "binary_targets": {},
        "binary_masks": {},
        "regression_targets": {},
        "regression_masks": {},
        "days_x": batch["days_x"].to(device, non_blocking=True),
        "days_mask": batch["days_mask"].to(device, non_blocking=True),
        "days_age": batch["days_age"].to(device, non_blocking=True),
        "meals_x": batch["meals_x"].to(device, non_blocking=True),
        "meals_mask": batch["meals_mask"].to(device, non_blocking=True),
        "meals_age": batch["meals_age"].to(device, non_blocking=True),
        "weeks_x": batch["weeks_x"].to(device, non_blocking=True),
        "weeks_mask": batch["weeks_mask"].to(device, non_blocking=True),
        "weeks_age": batch["weeks_age"].to(device, non_blocking=True),
    }
    for k in batch["binary_targets"]:
        out["binary_targets"][k] = batch["binary_targets"][k].to(device, non_blocking=True)
        out["binary_masks"][k] = batch["binary_masks"][k].to(device, non_blocking=True)
    for k in batch["regression_targets"]:
        out["regression_targets"][k] = batch["regression_targets"][k].to(device, non_blocking=True)
        out["regression_masks"][k] = batch["regression_masks"][k].to(device, non_blocking=True)
    return out


def compute_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    batch: Dict,
    binary_targets: List[str],
    regression_targets: List[str],
    pos_weight_map: Dict[str, float],
    binary_loss_weight: float,
    regression_loss_weight: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total_loss = torch.tensor(0.0, device=device)
    parts: Dict[str, float] = {}
    n_binary = 0
    n_reg = 0

    for t in binary_targets:
        logits = outputs["binary"][t]
        target = batch["binary_targets"][t]
        mask = batch["binary_masks"][t]
        valid = mask > 0
        if valid.sum() == 0:
            continue
        pos_weight = torch.tensor([pos_weight_map.get(t, 1.0)], device=device, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        loss = loss_fn(logits[valid], target[valid]).mean()
        total_loss = total_loss + binary_loss_weight * loss
        parts[f"binary__{t}"] = float(loss.detach().cpu().item())
        n_binary += 1

    for t in regression_targets:
        pred = outputs["regression"][t]
        target = batch["regression_targets"][t]
        mask = batch["regression_masks"][t]
        valid = mask > 0
        if valid.sum() == 0:
            continue
        loss = nn.functional.smooth_l1_loss(pred[valid], target[valid])
        total_loss = total_loss + regression_loss_weight * loss
        parts[f"regression__{t}"] = float(loss.detach().cpu().item())
        n_reg += 1

    denom = max(n_binary + n_reg, 1)
    total_loss = total_loss / denom
    parts["total"] = float(total_loss.detach().cpu().item())
    return total_loss, parts


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    binary_targets: List[str],
    regression_targets: List[str],
    regression_stats: Dict[str, Dict[str, float]],
    use_amp: bool,
) -> Dict:
    model.eval()
    out = {
        "anchor_id": [],
        "binary": {t: {"prob": [], "true": [], "mask": []} for t in binary_targets},
        "regression": {t: {"pred": [], "true": [], "mask": []} for t in regression_targets},
    }
    with torch.no_grad():
        for batch_cpu in loader:
            batch = move_batch_to_device(batch_cpu, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                outputs = model(batch)

            out["anchor_id"].extend(batch["anchor_id"])
            for t in binary_targets:
                prob = torch.sigmoid(outputs["binary"][t]).detach().cpu().numpy()
                true = batch["binary_targets"][t].detach().cpu().numpy()
                mask = batch["binary_masks"][t].detach().cpu().numpy()
                out["binary"][t]["prob"].append(prob)
                out["binary"][t]["true"].append(true)
                out["binary"][t]["mask"].append(mask)

            for t in regression_targets:
                pred = outputs["regression"][t].detach().cpu().numpy()
                true = batch["regression_targets"][t].detach().cpu().numpy()
                mask = batch["regression_masks"][t].detach().cpu().numpy()

                mean = regression_stats[t]["mean"]
                std = regression_stats[t]["std"]
                pred = pred * std + mean
                true = true * std + mean

                out["regression"][t]["pred"].append(pred)
                out["regression"][t]["true"].append(true)
                out["regression"][t]["mask"].append(mask)

    # concatenate
    for t in binary_targets:
        for k in ("prob", "true", "mask"):
            out["binary"][t][k] = np.concatenate(out["binary"][t][k], axis=0) if out["binary"][t][k] else np.array([])
    for t in regression_targets:
        for k in ("pred", "true", "mask"):
            out["regression"][t][k] = np.concatenate(out["regression"][t][k], axis=0) if out["regression"][t][k] else np.array([])
    return out


def evaluate_predictions(
    preds: Dict,
    binary_targets: List[str],
    regression_targets: List[str],
    tuned_thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    metrics: Dict[str, Dict] = {"binary": {}, "regression": {}}
    threshold_map = tuned_thresholds or {}
    bin_bal = []
    bin_auc = []
    reg_mae_norm = []

    for t in binary_targets:
        mask = preds["binary"][t]["mask"] > 0
        if mask.sum() == 0:
            continue
        y_true = preds["binary"][t]["true"][mask].astype(int)
        prob = preds["binary"][t]["prob"][mask].astype(float)
        thr = float(threshold_map.get(t, 0.5))
        m = binary_metrics(y_true, prob, threshold=thr)
        m["threshold"] = thr
        metrics["binary"][t] = m
        if "balanced_accuracy" in m:
            bin_bal.append(m["balanced_accuracy"])
        if "roc_auc" in m:
            bin_auc.append(m["roc_auc"])

    for t in regression_targets:
        mask = preds["regression"][t]["mask"] > 0
        if mask.sum() == 0:
            continue
        y_true = preds["regression"][t]["true"][mask].astype(float)
        pred = preds["regression"][t]["pred"][mask].astype(float)
        m = regression_metrics(y_true, pred)
        metrics["regression"][t] = m

    metrics["summary"] = {
        "mean_binary_balanced_accuracy": float(np.mean(bin_bal)) if bin_bal else None,
        "mean_binary_roc_auc": float(np.mean(bin_auc)) if bin_auc else None,
        "n_binary_targets": len(metrics["binary"]),
        "n_regression_targets": len(metrics["regression"]),
    }
    return metrics


def composite_score(
    val_metrics: Dict,
    regression_stats: Dict[str, Dict[str, float]],
) -> float:
    score = 0.0
    n = 0
    for t, m in val_metrics["binary"].items():
        score += m.get("balanced_accuracy", 0.0)
        score += 0.5 * m.get("roc_auc", 0.5)
        n += 1
    for t, m in val_metrics["regression"].items():
        scale = max(float(regression_stats[t]["std"]), 1e-6)
        score += -0.25 * (m["mae"] / scale)
        n += 1
    if n == 0:
        return -1e9
    return float(score / n)


def build_prediction_frame(
    preds: Dict,
    split_name: str,
    binary_targets: List[str],
    regression_targets: List[str],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    df = pd.DataFrame({"anchor_id": preds["anchor_id"], "split": split_name})
    for t in binary_targets:
        mask = preds["binary"][t]["mask"] > 0
        prob = preds["binary"][t]["prob"].astype(float)
        true = preds["binary"][t]["true"].astype(float)
        thr = thresholds.get(t, 0.5)
        pred = (prob >= thr).astype(float)
        df[f"{t}__mask"] = preds["binary"][t]["mask"]
        df[f"{t}__true"] = true
        df[f"{t}__prob"] = prob
        df[f"{t}__pred"] = pred
    for t in regression_targets:
        df[f"{t}__mask"] = preds["regression"][t]["mask"]
        df[f"{t}__true"] = preds["regression"][t]["true"]
        df[f"{t}__pred"] = preds["regression"][t]["pred"]
    return df


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    binary_targets: List[str],
    regression_targets: List[str],
    pos_weight_map: Dict[str, float],
    use_amp: bool,
    grad_clip: float,
    binary_loss_weight: float,
    regression_loss_weight: float,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    n_batches = 0

    for batch_cpu in loader:
        batch = move_batch_to_device(batch_cpu, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            outputs = model(batch)
            loss, parts = compute_loss(
                outputs=outputs,
                batch=batch,
                binary_targets=binary_targets,
                regression_targets=regression_targets,
                pos_weight_map=pos_weight_map,
                binary_loss_weight=binary_loss_weight,
                regression_loss_weight=regression_loss_weight,
                device=device,
            )
        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total += float(loss.detach().cpu().item())
        n_batches += 1

    return {"train_loss": total / max(n_batches, 1)}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    history: List[Dict],
    thresholds: Dict[str, float],
    config: Dict,
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "history": history,
        "thresholds": thresholds,
        "config": config,
    }
    torch.save(ckpt, path)


def maybe_load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, float, List[Dict], Dict[str, float]]:
    if not path.exists():
        return 0, -1e9, [], {}
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    try:
        scaler.load_state_dict(ckpt["scaler_state"])
    except Exception:
        pass
    return int(ckpt["epoch"]) + 1, float(ckpt["best_score"]), list(ckpt["history"]), dict(ckpt.get("thresholds", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train temporal multi-resolution models on meals/days/weeks sequences.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--dataset-dir", default="training/multires_sequence_dataset", help="Relative path to the multires dataset directory.")
    parser.add_argument("--run-name", default="", help="Optional run name. Auto-generated if omitted.")
    parser.add_argument("--model-family", choices=["gru", "tcn", "transformer"], default="gru")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--binary-targets", default=",".join(DEFAULT_BINARY_TARGETS))
    parser.add_argument("--regression-targets", default=",".join(DEFAULT_REGRESSION_TARGETS))
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--regression-loss-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(force_cpu=args.force_cpu)
    use_amp = bool(args.amp)

    project_root = Path(args.project_root).expanduser().resolve()
    dataset_dir = project_root / args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    binary_targets = [t.strip() for t in args.binary_targets.split(",") if t.strip()]
    regression_targets = [t.strip() for t in args.regression_targets.split(",") if t.strip()]
    if not binary_targets and not regression_targets:
        raise ValueError("At least one binary or regression target is required.")

    dataset, data_meta = build_datasets(
        dataset_dir=dataset_dir,
        binary_targets=binary_targets,
        regression_targets=regression_targets,
    )

    train_loader, val_loader, test_loader = make_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    run_name = args.run_name or f"{args.model_family}_hd{args.hidden_dim}_nl{args.num_layers}"
    model_dir = project_root / "models" / "temporal_multires" / run_name
    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    ensure_dir(model_dir)
    ensure_dir(report_dir)

    model = TemporalMultiResModel(
        day_input_dim=len(data_meta["day_feature_names"]),
        meal_input_dim=len(data_meta["meal_feature_names"]),
        week_input_dim=len(data_meta["week_feature_names"]),
        static_input_dim=6,
        model_family=args.model_family,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        binary_heads=binary_targets,
        regression_heads=regression_targets,
        day_seq_len=data_meta["seq_shapes"]["days_X"][1],
        meal_seq_len=data_meta["seq_shapes"]["meals_X"][1] if len(data_meta["seq_shapes"]["meals_X"]) >= 2 else 0,
        week_seq_len=data_meta["seq_shapes"]["weeks_X"][1] if len(data_meta["seq_shapes"]["weeks_X"]) >= 2 else 0,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            log("torch.compile() failed; continuing without compile.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=use_amp and device.type == "cuda")

    config = {
        "project_root": str(project_root),
        "dataset_dir": str(dataset_dir),
        "run_name": run_name,
        "device": str(device),
        "model_family": args.model_family,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "binary_targets": binary_targets,
        "regression_targets": regression_targets,
        "binary_loss_weight": args.binary_loss_weight,
        "regression_loss_weight": args.regression_loss_weight,
        "grad_clip": args.grad_clip,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "amp": args.amp,
        "compile": args.compile,
        "force_cpu": args.force_cpu,
    }

    save_json(report_dir / "config.json", config)
    save_json(report_dir / "data_manifest.json", data_meta)

    checkpoint_last = model_dir / "last_checkpoint.pt"
    checkpoint_best = model_dir / "best_checkpoint.pt"

    start_epoch = 0
    best_score = -1e9
    history: List[Dict] = []
    best_thresholds: Dict[str, float] = {}

    if args.resume:
        start_epoch, best_score, history, best_thresholds = maybe_load_checkpoint(
            checkpoint_last, model, optimizer, scaler, device
        )
        if start_epoch > 0:
            log(f"Resumed from epoch {start_epoch} with best_score={best_score:.4f}")

    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0

    for epoch in range(start_epoch, args.max_epochs):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            binary_targets=binary_targets,
            regression_targets=regression_targets,
            pos_weight_map=data_meta["pos_weight_map"],
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            binary_loss_weight=args.binary_loss_weight,
            regression_loss_weight=args.regression_loss_weight,
        )

        # Raw validation at 0.5 then tuned thresholds on val itself.
        val_preds = collect_predictions(
            model=model,
            loader=val_loader,
            device=device,
            binary_targets=binary_targets,
            regression_targets=regression_targets,
            regression_stats=data_meta["regression_stats"],
            use_amp=use_amp,
        )

        val_thresholds = {}
        for t in binary_targets:
            mask = val_preds["binary"][t]["mask"] > 0
            if mask.sum() == 0:
                val_thresholds[t] = 0.5
                continue
            y_true = val_preds["binary"][t]["true"][mask].astype(int)
            prob = val_preds["binary"][t]["prob"][mask].astype(float)
            val_thresholds[t] = choose_threshold(y_true, prob, metric="balanced_accuracy")

        val_metrics = evaluate_predictions(
            preds=val_preds,
            binary_targets=binary_targets,
            regression_targets=regression_targets,
            tuned_thresholds=val_thresholds,
        )
        score = composite_score(val_metrics, data_meta["regression_stats"])
        scheduler.step(score)

        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_composite_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        for t, m in val_metrics["binary"].items():
            row[f"val__{t}__balanced_accuracy"] = m.get("balanced_accuracy")
            row[f"val__{t}__roc_auc"] = m.get("roc_auc")
            row[f"val__{t}__f1"] = m.get("f1")
            row[f"val__{t}__threshold"] = m.get("threshold")
        for t, m in val_metrics["regression"].items():
            row[f"val__{t}__mae"] = m.get("mae")
            row[f"val__{t}__rmse"] = m.get("rmse")
            row[f"val__{t}__r2"] = m.get("r2")
        history.append(row)
        pd.DataFrame(history).to_csv(report_dir / "training_history.csv", index=False)

        improved = score > best_score
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_thresholds = dict(val_thresholds)
            epochs_without_improve = 0
            save_checkpoint(
                path=checkpoint_best,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                history=history,
                thresholds=best_thresholds,
                config=config,
            )
        else:
            epochs_without_improve += 1

        save_checkpoint(
            path=checkpoint_last,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            history=history,
            thresholds=best_thresholds,
            config=config,
        )

        log(f"{run_name} epoch {epoch+1}/{args.max_epochs} train={train_stats['train_loss']:.4f} val_score={score:.4f} best={best_score:.4f}")

        if epochs_without_improve >= args.patience:
            log("Early stopping triggered.")
            break

    # Evaluate best state.
    model.load_state_dict(best_state)
    torch.save(best_state, model_dir / "best_model_state.pt")
    save_json(report_dir / "selected_thresholds.json", best_thresholds)

    val_preds = collect_predictions(
        model=model,
        loader=val_loader,
        device=device,
        binary_targets=binary_targets,
        regression_targets=regression_targets,
        regression_stats=data_meta["regression_stats"],
        use_amp=use_amp,
    )
    test_preds = collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
        binary_targets=binary_targets,
        regression_targets=regression_targets,
        regression_stats=data_meta["regression_stats"],
        use_amp=use_amp,
    )

    val_metrics = evaluate_predictions(
        preds=val_preds,
        binary_targets=binary_targets,
        regression_targets=regression_targets,
        tuned_thresholds=best_thresholds,
    )
    test_metrics = evaluate_predictions(
        preds=test_preds,
        binary_targets=binary_targets,
        regression_targets=regression_targets,
        tuned_thresholds=best_thresholds,
    )

    save_json(report_dir / "val_metrics.json", val_metrics)
    save_json(report_dir / "test_metrics.json", test_metrics)

    build_prediction_frame(val_preds, "val", binary_targets, regression_targets, best_thresholds).to_csv(
        report_dir / "val_predictions.csv", index=False
    )
    build_prediction_frame(test_preds, "test", binary_targets, regression_targets, best_thresholds).to_csv(
        report_dir / "test_predictions.csv", index=False
    )

    summary_rows = []
    for t, m in test_metrics["binary"].items():
        summary_rows.append({
            "target": t,
            "kind": "binary_classification",
            "threshold": best_thresholds.get(t, 0.5),
            **m,
        })
    for t, m in test_metrics["regression"].items():
        summary_rows.append({
            "target": t,
            "kind": "regression",
            **m,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(report_dir / "test_summary.csv", index=False)

    final_summary = {
        "run_name": run_name,
        "device": str(device),
        "best_val_composite_score": best_score,
        "binary_targets": binary_targets,
        "regression_targets": regression_targets,
        "selected_thresholds": best_thresholds,
        "test_summary_rows": len(summary_df),
        "test_metrics": test_metrics,
    }
    save_json(report_dir / "final_summary.json", final_summary)

    log("Done.")
    log(f"Model artifacts written under: {model_dir}")
    log(f"Backtests written under: {report_dir}")


if __name__ == "__main__":
    main()
