from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


RANDOM_SEED = 42


SPACE_CONFIGS = {
    "meal_state_context": {
        "mode": "supervised_context",
        "source_csv": "training/predictive_views/meal_prediction_view.csv",
        "manifest_json": "models/retrieval_v3/meal_state_context/manifest.json",
        "preprocessor_joblib": "models/retrieval_v3/meal_state_context/preprocessor.joblib",
        "id_col": "meal_id",
        "time_col": "decision_time",
        "class_targets": [
            "y_next_restaurant_meal",
            "y_post_meal_budget_breach",
        ],
        "reg_targets": [
            "y_next_meal_kcal_log1p",
        ],
        "purity_labels": [
            "y_next_restaurant_meal",
            "y_post_meal_budget_breach",
            "target_meal_archetype_primary",
            "target_cuisine_primary",
        ],
        "default_hidden_dim": 512,
        "default_latent_dim": 64,
        "exclude_input_cols": [],
        "latent_eval_exclude_cols": [],
    },
    "meal_target_semantics": {
        "mode": "masked_semantic",
        "source_csv": "meal_db/final_repaired/meal_semantic_features.csv",
        "manifest_json": "models/retrieval_v3/meal_target_semantics/manifest.json",
        "preprocessor_joblib": "models/retrieval_v3/meal_target_semantics/preprocessor.joblib",
        "id_col": "meal_id",
        "time_col": "datetime_local_approx",
        "purity_labels": [
            "meal_archetype_primary",
            "cuisine_primary",
            "service_form_primary",
            "principal_protein",
            "principal_starch",
            "principal_veg",
        ],
        "numeric_labels": [
            "calories_kcal",
            "protein_g",
            "carbs_g",
            "fat_g",
        ],
        "default_hidden_dim": 512,
        "default_latent_dim": 64,
        "exclude_input_cols": [],
        "latent_eval_exclude_cols": [],
    },
    "weeks_structure": {
        "mode": "supervised_structure",
        "source_csv": "training/week_summary_matrix.csv",
        "manifest_json": "models/retrieval_v3/weeks_structure/manifest.json",
        "preprocessor_joblib": "models/retrieval_v3/weeks_structure/preprocessor.joblib",
        "id_col": "week_id",
        "time_col": "week_start",
        "class_targets": [
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
        ],
        "reg_targets": [
            "weight_delta_lb",
            "meal_events_per_day_week",
            "restaurant_meal_fraction_week",
            "budget_minus_logged_food_kcal_week",
        ],
        "purity_labels": [
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
            "dominant_service_form_week",
            "dominant_prep_profile_week",
            "dominant_protein_week",
            "dominant_starch_week",
        ],
        "numeric_labels": [
            "weight_delta_lb",
            "meal_events_per_day_week",
            "restaurant_meal_fraction_week",
            "budget_minus_logged_food_kcal_week",
        ],
        "default_hidden_dim": 192,
        "default_latent_dim": 24,
        "exclude_input_cols": [
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
            "dominant_service_form_week",
            "dominant_prep_profile_week",
            "dominant_protein_week",
            "dominant_starch_week",
            "dominant_energy_density_week",
            "dominant_satiety_style_week",
        ],
        "latent_eval_exclude_cols": [
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
            "dominant_service_form_week",
            "dominant_prep_profile_week",
            "dominant_protein_week",
            "dominant_starch_week",
        ],
    },
    "weekends_structure": {
        "mode": "supervised_structure",
        "source_csv": "training/weekend_summary_matrix.csv",
        "manifest_json": "models/retrieval_v3/weekends_structure/manifest.json",
        "preprocessor_joblib": "models/retrieval_v3/weekends_structure/preprocessor.joblib",
        "id_col": "weekend_id",
        "time_col": "weekend_start",
        "class_targets": [
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
        ],
        "reg_targets": [
            "weight_delta_lb",
            "meal_events_per_day_weekend",
            "restaurant_meal_fraction_weekend",
            "budget_minus_logged_food_kcal_weekend",
        ],
        "purity_labels": [
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
            "dominant_service_form_weekend",
            "dominant_prep_profile_weekend",
            "dominant_protein_weekend",
            "dominant_starch_weekend",
        ],
        "numeric_labels": [
            "weight_delta_lb",
            "meal_events_per_day_weekend",
            "restaurant_meal_fraction_weekend",
            "budget_minus_logged_food_kcal_weekend",
        ],
        "default_hidden_dim": 192,
        "default_latent_dim": 24,
        "exclude_input_cols": [
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
            "dominant_service_form_weekend",
            "dominant_prep_profile_weekend",
            "dominant_protein_weekend",
            "dominant_starch_weekend",
            "dominant_energy_density_weekend",
            "dominant_satiety_style_weekend",
        ],
        "latent_eval_exclude_cols": [
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
            "dominant_service_form_weekend",
            "dominant_prep_profile_weekend",
            "dominant_protein_weekend",
            "dominant_starch_weekend",
        ],
    },
}


def log(msg: str) -> None:
    print(f"[repr-v3.2] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_time_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def build_temporal_split_labels(df: pd.DataFrame, time_col: str) -> np.ndarray:
    times = parse_time_col(df, time_col)
    if times.notna().sum() == 0:
        n = len(df)
        labels = np.array(["train"] * n, dtype=object)
        val_start = int(n * 0.8)
        test_start = int(n * 0.9)
        labels[val_start:test_start] = "val"
        labels[test_start:] = "test"
        return labels

    order = pd.DataFrame({"i": np.arange(len(df)), "t": times}).sort_values("t")
    valid = order["t"].notna()
    idx = order.loc[valid, "i"].to_numpy()
    n = len(idx)

    labels = np.array(["train"] * len(df), dtype=object)
    if n < 20:
        val_start = max(1, int(n * 0.7))
        test_start = max(val_start + 1, int(n * 0.85))
    else:
        val_start = int(n * 0.8)
        test_start = int(n * 0.9)

    labels[idx[val_start:test_start]] = "val"
    labels[idx[test_start:]] = "test"
    return labels


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            try:
                return torch.amp.GradScaler(enabled=enabled)
            except TypeError:
                pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


class AutocastContext:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.ctx = None

    def __enter__(self):
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            try:
                self.ctx = torch.amp.autocast("cuda", enabled=self.enabled)
                return self.ctx.__enter__()
            except TypeError:
                pass
        self.ctx = torch.cuda.amp.autocast(enabled=self.enabled)
        return self.ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.ctx.__exit__(exc_type, exc_val, exc_tb)


@dataclass
class EncodedTargets:
    class_arrays: Dict[str, np.ndarray]
    class_meta: Dict[str, Dict]
    reg_arrays_raw: Dict[str, np.ndarray]
    reg_arrays_norm: Dict[str, np.ndarray]
    reg_stats: Dict[str, Dict[str, float]]


def encode_supervised_targets(
    df: pd.DataFrame,
    split_labels: np.ndarray,
    class_targets: List[str],
    reg_targets: List[str],
    min_class_count: int = 5,
) -> EncodedTargets:
    class_arrays: Dict[str, np.ndarray] = {}
    class_meta: Dict[str, Dict] = {}
    reg_arrays_raw: Dict[str, np.ndarray] = {}
    reg_arrays_norm: Dict[str, np.ndarray] = {}
    reg_stats: Dict[str, Dict[str, float]] = {}

    train_mask = split_labels == "train"

    for col in class_targets:
        if col not in df.columns:
            continue

        raw = df[col].astype("object")
        raw = raw.where(raw.notna(), other=None)

        if raw.dropna().isin([0, 1, True, False, "0", "1", "True", "False"]).all():
            raw = raw.map(
                lambda x: None if x is None else str(int(bool(x))) if isinstance(x, (bool, np.bool_)) else str(x)
            )

        vc = pd.Series(raw[train_mask]).dropna().astype(str).value_counts()
        common = set(vc[vc >= min_class_count].index.tolist())

        collapsed = []
        for x in raw:
            if x is None:
                collapsed.append(None)
            else:
                sx = str(x)
                collapsed.append(sx if sx in common else "OTHER")

        labels = sorted(pd.Series(collapsed).dropna().unique().tolist())
        if len(labels) < 2:
            continue

        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        enc = np.full(len(collapsed), -100, dtype=np.int64)
        for i, x in enumerate(collapsed):
            if x is not None:
                enc[i] = label_to_idx[x]

        class_arrays[col] = enc
        class_meta[col] = {
            "labels": labels,
            "label_to_idx": label_to_idx,
            "num_classes": len(labels),
        }

    for col in reg_targets:
        if col not in df.columns:
            continue
        arr = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
        if np.isfinite(arr).sum() < 20:
            continue

        train_vals = arr[train_mask]
        train_vals = train_vals[np.isfinite(train_vals)]
        if len(train_vals) < 5:
            continue

        mean = float(np.mean(train_vals))
        std = float(np.std(train_vals))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

        arr_norm = np.full_like(arr, np.nan, dtype=float)
        finite_mask = np.isfinite(arr)
        arr_norm[finite_mask] = (arr[finite_mask] - mean) / std

        reg_arrays_raw[col] = arr
        reg_arrays_norm[col] = arr_norm
        reg_stats[col] = {"mean": mean, "std": std}

    return EncodedTargets(
        class_arrays=class_arrays,
        class_meta=class_meta,
        reg_arrays_raw=reg_arrays_raw,
        reg_arrays_norm=reg_arrays_norm,
        reg_stats=reg_stats,
    )


class BaseTabularDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        ids: List[str],
        split_labels: np.ndarray,
        class_targets: Optional[Dict[str, np.ndarray]] = None,
        reg_targets_norm: Optional[Dict[str, np.ndarray]] = None,
        reg_targets_raw: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.X = X.astype(np.float32)
        self.ids = ids
        self.split_labels = split_labels
        self.class_targets = class_targets or {}
        self.reg_targets_norm = reg_targets_norm or {}
        self.reg_targets_raw = reg_targets_raw or {}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        out = {
            "x": torch.from_numpy(self.X[idx]),
            "id": self.ids[idx],
            "split": self.split_labels[idx],
        }
        for name, arr in self.class_targets.items():
            out[f"class::{name}"] = torch.tensor(arr[idx], dtype=torch.long)
        for name, arr in self.reg_targets_norm.items():
            val = arr[idx]
            out[f"reg_norm::{name}"] = torch.tensor(val if np.isfinite(val) else np.nan, dtype=torch.float32)
        for name, arr in self.reg_targets_raw.items():
            val = arr[idx]
            out[f"reg_raw::{name}"] = torch.tensor(val if np.isfinite(val) else np.nan, dtype=torch.float32)
        return out


def collate_fn(batch_list: List[Dict]) -> Dict:
    out = {}
    keys = batch_list[0].keys()
    for k in keys:
        if k in ("id", "split"):
            out[k] = [b[k] for b in batch_list]
        else:
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            d = hidden_dim
        layers.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        d = latent_dim
        for _ in range(depth):
            layers.extend([
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class SupervisedModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_meta: Dict[str, Dict],
        reg_targets: List[str],
        hidden_dim: int = 512,
        latent_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, latent_dim, depth, dropout)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, input_dim, depth, dropout)
        self.class_heads = nn.ModuleDict({k: nn.Linear(latent_dim, v["num_classes"]) for k, v in class_meta.items()})
        self.reg_heads = nn.ModuleDict({k: nn.Linear(latent_dim, 1) for k in reg_targets})

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        class_logits = {k: head(z) for k, head in self.class_heads.items()}
        reg_out = {k: head(z).squeeze(-1) for k, head in self.reg_heads.items()}
        return {"z": z, "recon": recon, "class_logits": class_logits, "reg_out": reg_out}


class MaskedSemanticModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim, latent_dim, depth, dropout)
        self.decoder = MLPDecoder(latent_dim, hidden_dim, input_dim, depth, dropout)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return {"z": z, "recon": recon}


def corrupt_features(x: torch.Tensor, mask_prob: float = 0.15, noise_std: float = 0.01) -> torch.Tensor:
    if mask_prob <= 0 and noise_std <= 0:
        return x
    out = x.clone()
    if mask_prob > 0:
        mask = torch.rand_like(out) < mask_prob
        out = out.masked_fill(mask, 0.0)
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if z1.shape[0] < 2:
        return torch.tensor(0.0, device=z1.device, dtype=torch.float32)

    z1 = nn.functional.normalize(z1.float(), dim=-1)
    z2 = nn.functional.normalize(z2.float(), dim=-1)
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    eye = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, torch.finfo(sim.dtype).min)
    targets = torch.arange(n, device=z.device)
    targets = torch.cat([targets + n, targets], dim=0)
    return nn.functional.cross_entropy(sim, targets)


def compute_supervised_loss(
    batch_gpu: Dict,
    outputs: Dict,
    class_meta: Dict[str, Dict],
    recon_weight: float,
    class_weight: float,
    reg_weight: float,
    regression_loss: str = "smoothl1",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = torch.tensor(0.0, device=outputs["recon"].device)
    parts: Dict[str, float] = {}

    recon_loss = nn.functional.mse_loss(outputs["recon"], batch_gpu["x"])
    loss = loss + recon_weight * recon_loss
    parts["recon_loss"] = float(recon_loss.detach().cpu())

    class_loss_total = torch.tensor(0.0, device=outputs["recon"].device)
    class_count = 0
    for name in class_meta.keys():
        y = batch_gpu[f"class::{name}"]
        mask = y >= 0
        if mask.sum() == 0:
            continue
        ce = nn.functional.cross_entropy(outputs["class_logits"][name][mask], y[mask])
        class_loss_total = class_loss_total + ce
        class_count += 1
    if class_count > 0:
        class_loss_total = class_loss_total / class_count
        loss = loss + class_weight * class_loss_total
        parts["class_loss"] = float(class_loss_total.detach().cpu())

    reg_loss_total = torch.tensor(0.0, device=outputs["recon"].device)
    reg_count = 0
    for name in outputs["reg_out"].keys():
        y = batch_gpu[f"reg_norm::{name}"]
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            continue
        pred = outputs["reg_out"][name][mask]
        yy = y[mask]
        if regression_loss == "mse":
            reg_l = nn.functional.mse_loss(pred, yy)
        else:
            reg_l = nn.functional.smooth_l1_loss(pred, yy, beta=1.0)
        reg_loss_total = reg_loss_total + reg_l
        reg_count += 1
    if reg_count > 0:
        reg_loss_total = reg_loss_total / reg_count
        loss = loss + reg_weight * reg_loss_total
        parts["reg_loss"] = float(reg_loss_total.detach().cpu())

    parts["total_loss"] = float(loss.detach().cpu())
    return loss, parts


def compute_semantic_loss(
    clean_x: torch.Tensor,
    out1: Dict,
    out2: Dict,
    recon_weight: float,
    contrastive_weight: float,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = torch.tensor(0.0, device=clean_x.device)
    parts: Dict[str, float] = {}

    recon1 = nn.functional.mse_loss(out1["recon"], clean_x)
    recon2 = nn.functional.mse_loss(out2["recon"], clean_x)
    recon_loss = 0.5 * (recon1 + recon2)
    loss = loss + recon_weight * recon_loss
    parts["recon_loss"] = float(recon_loss.detach().cpu())

    cont_loss = nt_xent_loss(out1["z"], out2["z"], temperature=temperature)
    loss = loss + contrastive_weight * cont_loss
    parts["contrastive_loss"] = float(cont_loss.detach().cpu())

    parts["total_loss"] = float(loss.detach().cpu())
    return loss, parts


@torch.no_grad()
def extract_latents(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, List[str], List[str]]:
    model.eval()
    zs: List[np.ndarray] = []
    ids: List[str] = []
    splits: List[str] = []
    for batch in loader:
        x = batch["x"].to(device)
        z = model.encoder(x).detach().cpu().numpy()
        zs.append(z)
        ids.extend(batch["id"])
        splits.extend(batch["split"])
    return np.vstack(zs), ids, splits


def cosine_knn(query: np.ndarray, ref: np.ndarray, k: int = 5) -> np.ndarray:
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    r = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    sims = q @ r.T
    idx = np.argsort(-sims, axis=1)[:, :k]
    return idx


def neighbor_purity_metrics(
    train_z: np.ndarray,
    train_df: pd.DataFrame,
    eval_z: np.ndarray,
    eval_df: pd.DataFrame,
    label_cols: List[str],
    k: int = 5,
) -> Dict:
    idx = cosine_knn(eval_z, train_z, k=k)
    out: Dict[str, float] = {}

    for col in label_cols:
        if col not in train_df.columns or col not in eval_df.columns:
            continue
        scores = []
        for i in range(len(eval_df)):
            true_val = eval_df.iloc[i][col]
            if pd.isna(true_val):
                continue
            nbr_vals = train_df.iloc[idx[i]][col]
            nbr_vals = nbr_vals.dropna().astype(str).tolist()
            if len(nbr_vals) == 0:
                continue
            scores.append(sum(v == str(true_val) for v in nbr_vals) / len(nbr_vals))
        if scores:
            out[f"{col}_purity_at_{k}"] = float(np.mean(scores))
    return out


def numeric_neighbor_gap_metrics(
    train_z: np.ndarray,
    train_df: pd.DataFrame,
    eval_z: np.ndarray,
    eval_df: pd.DataFrame,
    numeric_cols: List[str],
    k: int = 5,
) -> Dict:
    idx = cosine_knn(eval_z, train_z, k=k)
    out: Dict[str, float] = {}

    for col in numeric_cols:
        if col not in train_df.columns or col not in eval_df.columns:
            continue
        gaps = []
        for i in range(len(eval_df)):
            true_val = pd.to_numeric(pd.Series([eval_df.iloc[i][col]]), errors="coerce").iloc[0]
            if not np.isfinite(true_val):
                continue
            nbr_vals = pd.to_numeric(train_df.iloc[idx[i]][col], errors="coerce")
            nbr_vals = nbr_vals[np.isfinite(nbr_vals)]
            if len(nbr_vals) == 0:
                continue
            gaps.append(float(abs(true_val - nbr_vals.mean())))
        if gaps:
            out[f"{col}_neighbor_mean_abs_gap_at_{k}"] = float(np.mean(gaps))
    return out


@torch.no_grad()
def evaluate_supervised_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_meta: Dict[str, Dict],
    reg_stats: Dict[str, Dict[str, float]],
    recon_weight: float,
    class_weight: float,
    reg_weight: float,
    regression_loss: str = "smoothl1",
) -> Dict:
    model.eval()
    losses = []

    class_y_true: Dict[str, List[int]] = {k: [] for k in class_meta.keys()}
    class_probs: Dict[str, List[np.ndarray]] = {k: [] for k in class_meta.keys()}
    class_y_pred: Dict[str, List[int]] = {k: [] for k in class_meta.keys()}

    reg_y_true_raw: Dict[str, List[float]] = {k: [] for k in model.reg_heads.keys()}
    reg_y_pred_raw: Dict[str, List[float]] = {k: [] for k in model.reg_heads.keys()}

    for batch in loader:
        x = batch["x"].to(device)
        outputs = model(x)
        batch_gpu = {"x": x}
        for k, v in batch.items():
            if k.startswith("class::") or k.startswith("reg_norm::") or k.startswith("reg_raw::"):
                batch_gpu[k] = v.to(device)
        _, parts = compute_supervised_loss(batch_gpu, outputs, class_meta, recon_weight, class_weight, reg_weight, regression_loss=regression_loss)
        losses.append(parts)

        for name in class_meta.keys():
            y = batch_gpu[f"class::{name}"]
            mask = y >= 0
            if mask.sum() == 0:
                continue
            logits = outputs["class_logits"][name][mask]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            pred = probs.argmax(axis=1)
            true = y[mask].detach().cpu().numpy()
            class_y_true[name].extend(true.tolist())
            class_y_pred[name].extend(pred.tolist())
            class_probs[name].extend(probs.tolist())

        for name in model.reg_heads.keys():
            y_norm = batch_gpu[f"reg_norm::{name}"]
            y_raw = batch_gpu[f"reg_raw::{name}"]
            mask = torch.isfinite(y_norm)
            if mask.sum() == 0:
                continue
            pred_norm = outputs["reg_out"][name][mask].detach().cpu().numpy()
            true_raw = y_raw[mask].detach().cpu().numpy()
            stats = reg_stats[name]
            pred_raw = pred_norm * stats["std"] + stats["mean"]
            reg_y_true_raw[name].extend(true_raw.tolist())
            reg_y_pred_raw[name].extend(pred_raw.tolist())

    out = {"loss": {}, "classification": {}, "regression": {}}
    if losses:
        for key in losses[0].keys():
            out["loss"][key] = float(np.mean([d[key] for d in losses if key in d]))

    for name in class_meta.keys():
        yt = np.array(class_y_true[name], dtype=int)
        yp = np.array(class_y_pred[name], dtype=int)
        if len(yt) == 0:
            continue
        probs = np.array(class_probs[name], dtype=float)
        metrics = {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        }
        if probs.ndim == 2 and probs.shape[1] == 2 and len(np.unique(yt)) >= 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(yt, probs[:, 1]))
            except Exception:
                pass
        out["classification"][name] = metrics

    for name in model.reg_heads.keys():
        yt = np.array(reg_y_true_raw[name], dtype=float)
        yp = np.array(reg_y_pred_raw[name], dtype=float)
        if len(yt) == 0:
            continue
        rmse = math.sqrt(mean_squared_error(yt, yp))
        out["regression"][name] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(rmse),
            "r2": float(r2_score(yt, yp)),
        }

    return out


@torch.no_grad()
def evaluate_semantic_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    recon_weight: float,
    contrastive_weight: float,
    temperature: float,
) -> Dict:
    model.eval()
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        x1 = corrupt_features(x, mask_prob=0.20, noise_std=0.01)
        x2 = corrupt_features(x, mask_prob=0.20, noise_std=0.01)
        out1 = model(x1)
        out2 = model(x2)
        _, parts = compute_semantic_loss(x, out1, out2, recon_weight, contrastive_weight, temperature)
        losses.append(parts)

    out = {"loss": {}}
    if losses:
        for key in losses[0].keys():
            out["loss"][key] = float(np.mean([d[key] for d in losses if key in d]))
    return out


def save_checkpoint(path: Path, model: nn.Module, optimizer, scaler, epoch: int, best_val: float, history: List[Dict], config: Dict):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "history": history,
        "config": config,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf")), ckpt.get("history", []), ckpt.get("config", {})


def export_latents(model: nn.Module, dataset: BaseTabularDataset, device: torch.device, out_csv: Path):
    loader = DataLoader(dataset, batch_size=min(512, max(1, len(dataset))), shuffle=False, num_workers=0, collate_fn=collate_fn)
    z, ids, splits = extract_latents(model, loader, device=device)
    rows = []
    for i in range(len(ids)):
        row = {"id": ids[i], "split": splits[i]}
        for j, val in enumerate(z[i]):
            row[f"z_{j:03d}"] = float(val)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def select_feature_columns(manifest_cols: List[str], df_cols: List[str], exclude_cols: List[str]) -> List[str]:
    exclude = set(exclude_cols)
    return [c for c in manifest_cols if c in df_cols and c not in exclude]


def filtered_purity_labels(all_labels: List[str], excluded_from_input: List[str]) -> List[str]:
    excluded = set(excluded_from_input)
    return [c for c in all_labels if c not in excluded]


def train_space(
    project_root: Path,
    space: str,
    batch_size: int,
    hidden_dim: int,
    latent_dim: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    recon_weight: float,
    class_weight: float,
    reg_weight: float,
    contrastive_weight: float,
    temperature: float,
    mask_prob: float,
    noise_std: float,
    resume: str,
    checkpoint_every: int,
    num_workers: int,
    amp: bool,
    compile_model: bool,
    regression_loss: str,
) -> None:
    cfg = SPACE_CONFIGS[space]
    mode = cfg["mode"]
    source_csv = project_root / cfg["source_csv"]
    manifest_json = project_root / cfg["manifest_json"]
    preprocessor_joblib = project_root / cfg["preprocessor_joblib"]

    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV for {space}: {source_csv}")
    if not manifest_json.exists():
        raise FileNotFoundError(f"Missing manifest for {space}: {manifest_json}")
    if not preprocessor_joblib.exists():
        raise FileNotFoundError(f"Missing preprocessor for {space}: {preprocessor_joblib}")

    run_dir = project_root / "models" / "representation_v3_2" / space
    ensure_dir(run_dir)

    log(f"Loading source data for {space} ...")
    df = pd.read_csv(source_csv, low_memory=False)
    manifest = load_json(manifest_json)
    feature_cols = select_feature_columns(
        manifest_cols=manifest["feature_columns"],
        df_cols=list(df.columns),
        exclude_cols=cfg.get("exclude_input_cols", []),
    )
    id_col = cfg["id_col"]
    time_col = cfg["time_col"]

    preprocessor = joblib.load(preprocessor_joblib)
    X_raw = df[feature_cols].copy()
    X_transformed = preprocessor.transform(X_raw)
    if hasattr(X_transformed, "toarray"):
        X_np = X_transformed.toarray().astype(np.float32)
    else:
        X_np = np.asarray(X_transformed, dtype=np.float32)

    split_labels = build_temporal_split_labels(df, time_col=time_col)
    ids = df[id_col].astype(str).tolist()

    if mode in {"supervised_context", "supervised_structure"}:
        encoded = encode_supervised_targets(df, split_labels, cfg["class_targets"], cfg["reg_targets"], min_class_count=5)
        model = SupervisedModel(
            input_dim=X_np.shape[1],
            class_meta=encoded.class_meta,
            reg_targets=list(encoded.reg_arrays_norm.keys()),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout=dropout,
        )
        dataset = BaseTabularDataset(
            X_np,
            ids,
            split_labels,
            encoded.class_arrays,
            encoded.reg_arrays_norm,
            encoded.reg_arrays_raw,
        )
        class_meta = encoded.class_meta
        reg_names = list(encoded.reg_arrays_norm.keys())
        reg_stats = encoded.reg_stats
    else:
        model = MaskedSemanticModel(
            input_dim=X_np.shape[1],
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout=dropout,
        )
        dataset = BaseTabularDataset(X_np, ids, split_labels, None, None, None)
        class_meta = {}
        reg_names = []
        reg_stats = {}

    device = get_device()
    use_amp = amp and device.type == "cuda"
    model = model.to(device)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            log("Enabled torch.compile")
        except Exception as e:
            log(f"torch.compile not enabled: {e}")

    train_idx = np.where(split_labels == "train")[0]
    val_idx = np.where(split_labels == "val")[0]
    test_idx = np.where(split_labels == "test")[0]

    train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_ds = torch.utils.data.Subset(dataset, test_idx.tolist())

    effective_batch_size = min(batch_size, max(1, len(train_idx)))
    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(1, len(val_idx))), shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=min(batch_size, max(1, len(test_idx))), shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = make_grad_scaler(enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")
    history: List[Dict] = []

    latest_ckpt = run_dir / "latest.pt"
    best_ckpt = run_dir / "best.pt"
    interrupt_ckpt = run_dir / "interrupt.pt"

    run_config = {
        "space": space,
        "mode": mode,
        "source_csv": str(source_csv),
        "manifest_json": str(manifest_json),
        "preprocessor_joblib": str(preprocessor_joblib),
        "feature_cols": feature_cols,
        "excluded_input_cols": cfg.get("exclude_input_cols", []),
        "input_dim": int(X_np.shape[1]),
        "batch_size_requested": batch_size,
        "batch_size_effective_train": effective_batch_size,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "depth": depth,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "recon_weight": recon_weight,
        "class_weight": class_weight,
        "reg_weight": reg_weight,
        "contrastive_weight": contrastive_weight,
        "temperature": temperature,
        "mask_prob": mask_prob,
        "noise_std": noise_std,
        "device": str(device),
        "amp": bool(use_amp),
        "class_targets": list(class_meta.keys()),
        "reg_targets": reg_names,
        "reg_stats": reg_stats,
        "regression_loss": regression_loss,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_idx)),
        "rows_val": int(len(val_idx)),
        "rows_test": int(len(test_idx)),
    }

    if resume:
        ckpt_path = latest_ckpt if resume == "latest" else Path(resume)
        if ckpt_path.exists():
            start_epoch, best_val, history, _old_cfg = load_checkpoint(ckpt_path, model, optimizer, scaler)
            log(f"Resumed {space} from {ckpt_path} at epoch {start_epoch}")
        else:
            log(f"Resume requested but checkpoint not found: {ckpt_path}")

    save_json(run_dir / "config.json", run_config)
    if class_meta:
        save_json(run_dir / "label_meta.json", class_meta)

    epochs_since_improve = 0

    try:
        for epoch in range(start_epoch + 1, max_epochs + 1):
            model.train()
            epoch_losses = []
            t0 = time.time()

            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                if mode in {"supervised_context", "supervised_structure"}:
                    x_noisy = corrupt_features(x, mask_prob=mask_prob, noise_std=noise_std)
                    batch_gpu = {"x": x}
                    for k, v in batch.items():
                        if k.startswith("class::") or k.startswith("reg_norm::") or k.startswith("reg_raw::"):
                            batch_gpu[k] = v.to(device, non_blocking=True)

                    with AutocastContext(enabled=use_amp):
                        outputs = model(x_noisy)
                        loss, parts = compute_supervised_loss(
                            batch_gpu,
                            outputs,
                            class_meta=class_meta,
                            recon_weight=recon_weight,
                            class_weight=class_weight,
                            reg_weight=reg_weight,
                            regression_loss=regression_loss,
                        )
                else:
                    x1 = corrupt_features(x, mask_prob=mask_prob, noise_std=noise_std)
                    x2 = corrupt_features(x, mask_prob=mask_prob, noise_std=noise_std)
                    with AutocastContext(enabled=use_amp):
                        out1 = model(x1)
                        out2 = model(x2)
                        loss, parts = compute_semantic_loss(
                            x,
                            out1,
                            out2,
                            recon_weight=recon_weight,
                            contrastive_weight=contrastive_weight,
                            temperature=temperature,
                        )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(parts)

            train_total = float(np.mean([p["total_loss"] for p in epoch_losses])) if epoch_losses else float("nan")

            if mode in {"supervised_context", "supervised_structure"}:
                val_metrics = evaluate_supervised_model(
                    model=model,
                    loader=val_loader,
                    device=device,
                    class_meta=class_meta,
                    reg_stats=reg_stats,
                    recon_weight=recon_weight,
                    class_weight=class_weight,
                    reg_weight=reg_weight,
                    regression_loss=regression_loss,
                )
            else:
                val_metrics = evaluate_semantic_model(
                    model=model,
                    loader=val_loader,
                    device=device,
                    recon_weight=recon_weight,
                    contrastive_weight=contrastive_weight,
                    temperature=temperature,
                )

            val_total = val_metrics["loss"].get("total_loss", float("inf"))

            row = {
                "epoch": epoch,
                "train_total_loss": train_total,
                "val_total_loss": val_total,
                "seconds": round(time.time() - t0, 2),
            }

            if mode in {"supervised_context", "supervised_structure"}:
                for head, metrics in val_metrics.get("classification", {}).items():
                    row[f"val_cls_{head}_acc"] = metrics.get("accuracy")
                    row[f"val_cls_{head}_f1"] = metrics.get("macro_f1")
                    if "roc_auc" in metrics:
                        row[f"val_cls_{head}_auc"] = metrics.get("roc_auc")
                for head, metrics in val_metrics.get("regression", {}).items():
                    row[f"val_reg_{head}_mae"] = metrics.get("mae")
                    row[f"val_reg_{head}_r2"] = metrics.get("r2")
            else:
                row["val_recon_loss"] = val_metrics["loss"].get("recon_loss")
                row["val_contrastive_loss"] = val_metrics["loss"].get("contrastive_loss")

            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            improved = val_total < best_val
            if improved:
                best_val = val_total
                epochs_since_improve = 0
                save_checkpoint(best_ckpt, model, optimizer, scaler, epoch, best_val, history, run_config)
            else:
                epochs_since_improve += 1

            if epoch % checkpoint_every == 0 or improved:
                save_checkpoint(latest_ckpt, model, optimizer, scaler, epoch, best_val, history, run_config)

            log(f"{space} epoch {epoch}/{max_epochs} train={train_total:.4f} val={val_total:.4f} best={best_val:.4f}")

            if patience > 0 and epochs_since_improve >= patience:
                log(f"Early stopping triggered for {space} after {epoch} epochs.")
                break

    except KeyboardInterrupt:
        save_checkpoint(interrupt_ckpt, model, optimizer, scaler, epoch if 'epoch' in locals() else 0, best_val, history, run_config)
        log(f"Interrupted. Saved checkpoint to: {interrupt_ckpt}")
        return

    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model)

    if mode in {"supervised_context", "supervised_structure"}:
        test_metrics = evaluate_supervised_model(
            model=model,
            loader=test_loader,
            device=device,
            class_meta=class_meta,
            reg_stats=reg_stats,
            recon_weight=recon_weight,
            class_weight=class_weight,
            reg_weight=reg_weight,
            regression_loss=regression_loss,
        )
    else:
        test_metrics = evaluate_semantic_model(
            model=model,
            loader=test_loader,
            device=device,
            recon_weight=recon_weight,
            contrastive_weight=contrastive_weight,
            temperature=temperature,
        )

    full_loader = DataLoader(dataset, batch_size=min(512, max(1, len(dataset))), shuffle=False, num_workers=0, collate_fn=collate_fn)
    z_all, ids_all, split_all = extract_latents(model, full_loader, device=device)
    split_arr = np.array(split_all, dtype=object)

    train_mask = split_arr == "train"
    val_mask = split_arr == "val"
    test_mask = split_arr == "test"

    purity_labels = filtered_purity_labels(
        cfg.get("purity_labels", []),
        cfg.get("latent_eval_exclude_cols", []),
    )

    latent_eval = {}
    if train_mask.sum() > 0 and val_mask.sum() > 0:
        latent_eval["val"] = neighbor_purity_metrics(
            train_z=z_all[train_mask],
            train_df=df.iloc[np.where(train_mask)[0]].reset_index(drop=True),
            eval_z=z_all[val_mask],
            eval_df=df.iloc[np.where(val_mask)[0]].reset_index(drop=True),
            label_cols=purity_labels,
            k=5,
        )
        if "numeric_labels" in cfg:
            latent_eval["val"].update(
                numeric_neighbor_gap_metrics(
                    train_z=z_all[train_mask],
                    train_df=df.iloc[np.where(train_mask)[0]].reset_index(drop=True),
                    eval_z=z_all[val_mask],
                    eval_df=df.iloc[np.where(val_mask)[0]].reset_index(drop=True),
                    numeric_cols=cfg.get("numeric_labels", []),
                    k=5,
                )
            )
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        latent_eval["test"] = neighbor_purity_metrics(
            train_z=z_all[train_mask],
            train_df=df.iloc[np.where(train_mask)[0]].reset_index(drop=True),
            eval_z=z_all[test_mask],
            eval_df=df.iloc[np.where(test_mask)[0]].reset_index(drop=True),
            label_cols=purity_labels,
            k=5,
        )
        if "numeric_labels" in cfg:
            latent_eval["test"].update(
                numeric_neighbor_gap_metrics(
                    train_z=z_all[train_mask],
                    train_df=df.iloc[np.where(train_mask)[0]].reset_index(drop=True),
                    eval_z=z_all[test_mask],
                    eval_df=df.iloc[np.where(test_mask)[0]].reset_index(drop=True),
                    numeric_cols=cfg.get("numeric_labels", []),
                    k=5,
                )
            )

    test_metrics["latent_neighbor_eval"] = latent_eval
    save_json(run_dir / "test_metrics.json", test_metrics)
    export_latents(model, dataset, device=device, out_csv=run_dir / "embeddings_latent.csv")
    save_checkpoint(latest_ckpt, model, optimizer, scaler, epoch if 'epoch' in locals() else 0, best_val, history, run_config)

    log(f"Finished space: {space}")
    log(f"Wrote outputs to: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train representation encoders v3.2 with leakage-safe regime inputs.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=["meal_state_context", "meal_target_semantics"],
        choices=list(SPACE_CONFIGS.keys()),
        help="Spaces to train.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=0, help="0 uses space-specific defaults.")
    parser.add_argument("--latent-dim", type=int, default=0, help="0 uses space-specific defaults.")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mask-prob", type=float, default=0.20)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--resume", default="", help="Use 'latest' or provide checkpoint path.")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows unless you know your setup is stable.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile if available.")
    parser.add_argument("--regression-loss", choices=["smoothl1", "mse"], default="smoothl1")
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)
    project_root = Path(args.project_root).expanduser().resolve()
    ensure_dir(project_root / "models" / "representation_v3_2")

    for space in args.spaces:
        cfg = SPACE_CONFIGS[space]
        hidden_dim = args.hidden_dim if args.hidden_dim > 0 else cfg["default_hidden_dim"]
        latent_dim = args.latent_dim if args.latent_dim > 0 else cfg["default_latent_dim"]

        train_space(
            project_root=project_root,
            space=space,
            batch_size=args.batch_size,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            depth=args.depth,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            recon_weight=args.recon_weight,
            class_weight=args.class_weight,
            reg_weight=args.reg_weight,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
            mask_prob=args.mask_prob,
            noise_std=args.noise_std,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            num_workers=args.num_workers,
            amp=args.amp,
            compile_model=args.compile,
            regression_loss=args.regression_loss,
        )


if __name__ == "__main__":
    main()
