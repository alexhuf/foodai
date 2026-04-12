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


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

SPACE_CONFIGS = {
    "meal_state_context": {
        "source_csv": "training/predictive_views/meal_prediction_view.csv",
        "manifest_json": "models/retrieval_v3/meal_state_context/manifest.json",
        "preprocessor_joblib": "models/retrieval_v3/meal_state_context/preprocessor.joblib",
        "id_col": "meal_id",
        "time_col": "decision_time",
        "class_targets": [
            "y_next_meal_family_coarse",
            "y_next_restaurant_meal",
            "y_post_meal_budget_breach",
        ],
        "reg_targets": [
            "y_next_meal_kcal_log1p",
        ],
    },
    "meal_target_semantics": {
        "source_csv": "meal_db/final_repaired/meal_semantic_features.csv",
        "manifest_json": "models/retrieval_v3/meal_target_semantics/manifest.json",
        "preprocessor_joblib": "models/retrieval_v3/meal_target_semantics/preprocessor.joblib",
        "id_col": "meal_id",
        "time_col": "datetime_local_approx",
        "class_targets": [
            "meal_archetype_primary",
            "cuisine_primary",
            "service_form_primary",
            "principal_protein",
        ],
        "reg_targets": [
            "calories_kcal",
            "protein_g",
            "carbs_g",
            "fat_g",
            "comfort_food_score",
            "fresh_light_score",
            "indulgence_score",
        ],
    },
    "weeks_structure": {
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
    },
    "weekends_structure": {
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
    },
}

RANDOM_SEED = 42


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[repr] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_time_col(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def build_temporal_split_indices(df: pd.DataFrame, time_col: str) -> pd.Series:
    # Train / val / test by chronological unique time buckets
    times = parse_time_col(df, time_col)
    if times.notna().sum() == 0:
        # fallback to row-order split
        n = len(df)
        idx = pd.Series(["train"] * n)
        val_start = int(n * 0.8)
        test_start = int(n * 0.9)
        idx.iloc[val_start:test_start] = "val"
        idx.iloc[test_start:] = "test"
        return idx

    tmp = pd.DataFrame({"i": np.arange(len(df)), "t": times}).sort_values("t")
    valid = tmp["t"].notna()
    ordered = tmp.loc[valid, "i"].tolist()
    n = len(ordered)

    split = pd.Series(["train"] * len(df))
    if n < 20:
        val_start = max(1, int(n * 0.7))
        test_start = max(val_start + 1, int(n * 0.85))
    else:
        val_start = int(n * 0.8)
        test_start = int(n * 0.9)

    for j in ordered[val_start:test_start]:
        split.iloc[j] = "val"
    for j in ordered[test_start:]:
        split.iloc[j] = "test"
    return split


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 3) -> float:
    if probs.ndim != 2:
        return float("nan")
    top_idx = np.argsort(-probs, axis=1)[:, :k]
    hits = [(y_true[i] in top_idx[i]) for i in range(len(y_true))]
    return float(np.mean(hits)) if hits else float("nan")


# ------------------------------------------------------------
# Encoding targets
# ------------------------------------------------------------

@dataclass
class EncodedTargets:
    class_arrays: Dict[str, np.ndarray]
    class_meta: Dict[str, Dict]
    reg_arrays: Dict[str, np.ndarray]


def encode_targets(df: pd.DataFrame, class_targets: List[str], reg_targets: List[str], min_class_count: int = 10) -> EncodedTargets:
    class_arrays: Dict[str, np.ndarray] = {}
    class_meta: Dict[str, Dict] = {}
    reg_arrays: Dict[str, np.ndarray] = {}

    for col in class_targets:
        if col not in df.columns:
            continue
        raw = df[col].astype("object")
        raw = raw.where(raw.notna(), other=None)

        # Normalize booleans / binary-ish numerics
        if raw.dropna().isin([0, 1, True, False, "0", "1", "True", "False"]).all():
            raw = raw.map(lambda x: None if x is None else str(int(bool(x))) if isinstance(x, (bool, np.bool_)) else str(x))

        vc = pd.Series(raw).dropna().astype(str).value_counts()
        common = set(vc[vc >= min_class_count].index.tolist())

        collapsed = []
        for x in raw:
            if x is None:
                collapsed.append(None)
            else:
                sx = str(x)
                collapsed.append(sx if sx in common else "OTHER")

        labels = sorted(pd.Series(collapsed).dropna().unique().tolist())
        label_to_idx = {lab: i for i, lab in enumerate(labels)}
        encoded = np.full(len(collapsed), -100, dtype=np.int64)
        for i, x in enumerate(collapsed):
            if x is not None:
                encoded[i] = label_to_idx[x]

        if len(labels) >= 2:
            class_arrays[col] = encoded
            class_meta[col] = {
                "labels": labels,
                "label_to_idx": label_to_idx,
                "num_classes": len(labels),
            }

    for col in reg_targets:
        if col not in df.columns:
            continue
        arr = pd.to_numeric(df[col], errors="coerce").astype(float).to_numpy()
        if np.isfinite(arr).sum() >= 20:
            reg_arrays[col] = arr

    return EncodedTargets(class_arrays=class_arrays, class_meta=class_meta, reg_arrays=reg_arrays)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class TabularMultitaskDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        ids: List[str],
        split_labels: np.ndarray,
        class_targets: Dict[str, np.ndarray],
        reg_targets: Dict[str, np.ndarray],
    ) -> None:
        self.X = X.astype(np.float32)
        self.ids = ids
        self.split_labels = split_labels
        self.class_targets = class_targets
        self.reg_targets = reg_targets

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        item = {
            "x": torch.from_numpy(self.X[idx]),
            "id": self.ids[idx],
            "split": self.split_labels[idx],
        }
        for name, arr in self.class_targets.items():
            item[f"class::{name}"] = torch.tensor(arr[idx], dtype=torch.long)
        for name, arr in self.reg_targets.items():
            val = arr[idx]
            item[f"reg::{name}"] = torch.tensor(val if np.isfinite(val) else np.nan, dtype=torch.float32)
        return item


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

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


class MultiTaskEncoderModel(nn.Module):
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
        self.class_heads = nn.ModuleDict({
            name: nn.Linear(latent_dim, meta["num_classes"]) for name, meta in class_meta.items()
        })
        self.reg_heads = nn.ModuleDict({
            name: nn.Linear(latent_dim, 1) for name in reg_targets
        })

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        class_logits = {name: head(z) for name, head in self.class_heads.items()}
        reg_out = {name: head(z).squeeze(-1) for name, head in self.reg_heads.items()}
        return {
            "z": z,
            "recon": recon,
            "class_logits": class_logits,
            "reg_out": reg_out,
        }


# ------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------

def compute_loss(
    batch: Dict,
    outputs: Dict,
    class_meta: Dict[str, Dict],
    recon_weight: float,
    class_weight: float,
    reg_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss = torch.tensor(0.0, device=outputs["recon"].device)
    parts: Dict[str, float] = {}

    # Reconstruction
    recon_loss = nn.functional.mse_loss(outputs["recon"], batch["x"])
    loss = loss + recon_weight * recon_loss
    parts["recon_loss"] = float(recon_loss.detach().cpu())

    # Classification
    class_loss_total = torch.tensor(0.0, device=outputs["recon"].device)
    class_count = 0
    for name in class_meta.keys():
        y = batch[f"class::{name}"]
        mask = y >= 0
        if mask.sum() == 0:
            continue
        logits = outputs["class_logits"][name][mask]
        yy = y[mask]
        ce = nn.functional.cross_entropy(logits, yy)
        class_loss_total = class_loss_total + ce
        class_count += 1
    if class_count > 0:
        class_loss_total = class_loss_total / class_count
        loss = loss + class_weight * class_loss_total
        parts["class_loss"] = float(class_loss_total.detach().cpu())

    # Regression
    reg_loss_total = torch.tensor(0.0, device=outputs["recon"].device)
    reg_count = 0
    for name in outputs["reg_out"].keys():
        y = batch[f"reg::{name}"]
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            continue
        pred = outputs["reg_out"][name][mask]
        yy = y[mask]
        mse = nn.functional.mse_loss(pred, yy)
        reg_loss_total = reg_loss_total + mse
        reg_count += 1
    if reg_count > 0:
        reg_loss_total = reg_loss_total / reg_count
        loss = loss + reg_weight * reg_loss_total
        parts["reg_loss"] = float(reg_loss_total.detach().cpu())

    parts["total_loss"] = float(loss.detach().cpu())
    return loss, parts


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_meta: Dict[str, Dict],
    recon_weight: float,
    class_weight: float,
    reg_weight: float,
) -> Dict:
    model.eval()
    losses = []

    class_y_true: Dict[str, List[int]] = {k: [] for k in class_meta.keys()}
    class_probs: Dict[str, List[np.ndarray]] = {k: [] for k in class_meta.keys()}
    class_y_pred: Dict[str, List[int]] = {k: [] for k in class_meta.keys()}

    reg_y_true: Dict[str, List[float]] = {k: [] for k in model.reg_heads.keys()}
    reg_y_pred: Dict[str, List[float]] = {k: [] for k in model.reg_heads.keys()}

    for batch in loader:
        x = batch["x"].to(device)
        outputs = model(x)
        batch_gpu = {"x": x}
        for k, v in batch.items():
            if k.startswith("class::") or k.startswith("reg::"):
                batch_gpu[k] = v.to(device)
        _, parts = compute_loss(batch_gpu, outputs, class_meta, recon_weight, class_weight, reg_weight)
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
            y = batch_gpu[f"reg::{name}"]
            mask = torch.isfinite(y)
            if mask.sum() == 0:
                continue
            pred = outputs["reg_out"][name][mask].detach().cpu().numpy()
            true = y[mask].detach().cpu().numpy()
            reg_y_true[name].extend(true.tolist())
            reg_y_pred[name].extend(pred.tolist())

    # Aggregate losses
    out = {
        "loss": {},
        "classification": {},
        "regression": {},
    }
    if losses:
        for key in losses[0].keys():
            out["loss"][key] = float(np.mean([d[key] for d in losses if key in d]))

    # Classification metrics
    for name, meta in class_meta.items():
        yt = np.array(class_y_true[name], dtype=int)
        yp = np.array(class_y_pred[name], dtype=int)
        if len(yt) == 0:
            continue
        probs = np.array(class_probs[name], dtype=float)
        metrics = {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        }
        if probs.ndim == 2 and probs.shape[1] >= 3:
            metrics["top3_accuracy"] = float(topk_accuracy(yt, probs, k=3))
        if probs.ndim == 2 and probs.shape[1] == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(yt, probs[:, 1]))
            except Exception:
                pass
        out["classification"][name] = metrics

    # Regression metrics
    for name in model.reg_heads.keys():
        yt = np.array(reg_y_true[name], dtype=float)
        yp = np.array(reg_y_pred[name], dtype=float)
        if len(yt) == 0:
            continue
        rmse = math.sqrt(mean_squared_error(yt, yp))
        out["regression"][name] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(rmse),
            "r2": float(r2_score(yt, yp)),
        }

    return out


def collate_fn(batch_list: List[Dict]) -> Dict:
    out: Dict[str, object] = {}
    keys = batch_list[0].keys()
    for k in keys:
        if k in ("id", "split"):
            out[k] = [b[k] for b in batch_list]
        else:
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_val: float,
    history: List[Dict],
    config: Dict,
) -> None:
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


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[int, float, List[Dict], Dict]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf")), ckpt.get("history", []), ckpt.get("config", {})


def export_latents(
    model: nn.Module,
    dataset: TabularMultitaskDataset,
    device: torch.device,
    out_csv: Path,
) -> None:
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn)
    all_rows = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            x = batch["x"].to(device)
            z = model.encoder(x).detach().cpu().numpy()
            for i in range(z.shape[0]):
                row = {"id": batch["id"][i], "split": batch["split"][i]}
                for j, val in enumerate(z[i]):
                    row[f"z_{j:03d}"] = float(val)
                all_rows.append(row)
            offset += z.shape[0]
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)


# ------------------------------------------------------------
# Main per-space runner
# ------------------------------------------------------------

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
    resume: str,
    checkpoint_every: int,
    num_workers: int,
    amp: bool,
    compile_model: bool,
) -> None:
    cfg = SPACE_CONFIGS[space]
    source_csv = project_root / cfg["source_csv"]
    manifest_json = project_root / cfg["manifest_json"]
    preprocessor_joblib = project_root / cfg["preprocessor_joblib"]

    if not source_csv.exists():
        raise FileNotFoundError(f"Missing source CSV for {space}: {source_csv}")
    if not manifest_json.exists():
        raise FileNotFoundError(f"Missing manifest for {space}: {manifest_json}")
    if not preprocessor_joblib.exists():
        raise FileNotFoundError(f"Missing preprocessor for {space}: {preprocessor_joblib}")

    run_dir = project_root / "models" / "representation" / space
    ensure_dir(run_dir)

    log(f"Loading source data for {space} ...")
    df = pd.read_csv(source_csv, low_memory=False)
    manifest = load_json(manifest_json)
    feature_cols = [c for c in manifest["feature_columns"] if c in df.columns]
    id_col = cfg["id_col"]
    time_col = cfg["time_col"]

    preprocessor = joblib.load(preprocessor_joblib)
    X_raw = df[feature_cols].copy()
    X_transformed = preprocessor.transform(X_raw)
    if hasattr(X_transformed, "toarray"):
        X_np = X_transformed.toarray().astype(np.float32)
    else:
        X_np = np.asarray(X_transformed, dtype=np.float32)

    split_labels = build_temporal_split_indices(df, time_col=time_col).astype(str).to_numpy()
    ids = df[id_col].astype(str).tolist()

    encoded = encode_targets(df, cfg["class_targets"], cfg["reg_targets"], min_class_count=10)

    dataset = TabularMultitaskDataset(
        X=X_np,
        ids=ids,
        split_labels=split_labels,
        class_targets=encoded.class_arrays,
        reg_targets=encoded.reg_arrays,
    )

    train_idx = np.where(split_labels == "train")[0]
    val_idx = np.where(split_labels == "val")[0]
    test_idx = np.where(split_labels == "test")[0]

    train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
    test_ds = torch.utils.data.Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device.type == "cuda"

    model = MultiTaskEncoderModel(
        input_dim=X_np.shape[1],
        class_meta=encoded.class_meta,
        reg_targets=list(encoded.reg_arrays.keys()),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        depth=depth,
        dropout=dropout,
    ).to(device)

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            log("Enabled torch.compile")
        except Exception as e:
            log(f"torch.compile not enabled: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")
    history: List[Dict] = []

    latest_ckpt = run_dir / "latest.pt"
    best_ckpt = run_dir / "best.pt"
    interrupt_ckpt = run_dir / "interrupt.pt"

    run_config = {
        "space": space,
        "source_csv": str(source_csv),
        "manifest_json": str(manifest_json),
        "preprocessor_joblib": str(preprocessor_joblib),
        "feature_cols": feature_cols,
        "class_targets": list(encoded.class_arrays.keys()),
        "reg_targets": list(encoded.reg_arrays.keys()),
        "input_dim": int(X_np.shape[1]),
        "batch_size": batch_size,
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
        "device": str(device),
        "amp": bool(use_amp),
    }

    if resume:
        ckpt_path = latest_ckpt if resume == "latest" else Path(resume)
        if ckpt_path.exists():
            start_epoch, best_val, history, old_cfg = load_checkpoint(ckpt_path, model, optimizer, scaler)
            log(f"Resumed {space} from {ckpt_path} at epoch {start_epoch}")
        else:
            log(f"Resume requested but checkpoint not found: {ckpt_path}")

    save_json(run_dir / "config.json", run_config)
    save_json(run_dir / "label_meta.json", encoded.class_meta)

    epochs_since_improve = 0

    try:
        for epoch in range(start_epoch + 1, max_epochs + 1):
            model.train()
            epoch_losses = []

            t0 = time.time()
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)

                batch_gpu = {"x": x}
                for k, v in batch.items():
                    if k.startswith("class::") or k.startswith("reg::"):
                        batch_gpu[k] = v.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(x)
                    loss, parts = compute_loss(
                        batch_gpu,
                        outputs,
                        encoded.class_meta,
                        recon_weight=recon_weight,
                        class_weight=class_weight,
                        reg_weight=reg_weight,
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(parts)

            train_loss = float(np.mean([x["total_loss"] for x in epoch_losses])) if epoch_losses else float("nan")
            val_metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                class_meta=encoded.class_meta,
                recon_weight=recon_weight,
                class_weight=class_weight,
                reg_weight=reg_weight,
            )
            val_total = val_metrics["loss"].get("total_loss", float("inf"))

            row = {
                "epoch": epoch,
                "train_total_loss": train_loss,
                "val_total_loss": val_total,
                "seconds": round(time.time() - t0, 2),
            }

            # Flatten a subset of metrics into history
            for head, metrics in val_metrics["classification"].items():
                row[f"val_cls_{head}_acc"] = metrics.get("accuracy")
                row[f"val_cls_{head}_f1"] = metrics.get("macro_f1")
                if "top3_accuracy" in metrics:
                    row[f"val_cls_{head}_top3"] = metrics.get("top3_accuracy")
            for head, metrics in val_metrics["regression"].items():
                row[f"val_reg_{head}_mae"] = metrics.get("mae")
                row[f"val_reg_{head}_r2"] = metrics.get("r2")

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

            log(f"{space} epoch {epoch}/{max_epochs} train={train_loss:.4f} val={val_total:.4f} best={best_val:.4f}")

            if patience > 0 and epochs_since_improve >= patience:
                log(f"Early stopping triggered for {space} after {epoch} epochs.")
                break

    except KeyboardInterrupt:
        save_checkpoint(interrupt_ckpt, model, optimizer, scaler, epoch if 'epoch' in locals() else 0, best_val, history, run_config)
        log(f"Interrupted. Saved checkpoint to: {interrupt_ckpt}")
        return

    # Final evaluation from best checkpoint
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        class_meta=encoded.class_meta,
        recon_weight=recon_weight,
        class_weight=class_weight,
        reg_weight=reg_weight,
    )
    save_json(run_dir / "test_metrics.json", test_metrics)
    export_latents(model, dataset, device=device, out_csv=run_dir / "embeddings_latent.csv")
    save_checkpoint(latest_ckpt, model, optimizer, scaler, epoch if 'epoch' in locals() else 0, best_val, history, run_config)

    log(f"Finished space: {space}")
    log(f"Wrote outputs to: {run_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPU-heavy multi-task representation encoders with checkpoint/resume.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=["meal_state_context", "meal_target_semantics"],
        choices=list(SPACE_CONFIGS.keys()),
        help="One or more spaces to train.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25, help="0 disables early stopping.")
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--resume", default="", help="Use 'latest' or provide a checkpoint path.")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on Windows unless you know your setup is stable.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile if available.")
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)
    project_root = Path(args.project_root).expanduser().resolve()
    ensure_dir(project_root / "models" / "representation")

    for space in args.spaces:
        train_space(
            project_root=project_root,
            space=space,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            depth=args.depth,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            recon_weight=args.recon_weight,
            class_weight=args.class_weight,
            reg_weight=args.reg_weight,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            num_workers=args.num_workers,
            amp=args.amp,
            compile_model=args.compile,
        )


if __name__ == "__main__":
    main()
