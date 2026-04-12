from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42

MEAL_SAFE_DECISION_FEATURES = [
    "decision_hour",
    "time_slot",
    "time_slot_label",
    "meal_order_in_day",
    "is_first_meal_of_day",
    "hours_since_prior_meal",
    "cumulative_meal_calories_before_meal",
    "remaining_budget_before_meal_kcal",
]

MEAL_STATE_EXCLUDE = {
    "meal_id",
    "date",
    "decision_time",
    "is_last_meal_of_day",
    "hours_until_next_meal",
    "day_meal_count",
    "state_prior_meal_id",
    "state_prior_meal_text",
}


def log(msg: str) -> None:
    print(f"[retrieval] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def read_json_if_exists(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for c in X.columns:
        s = X[c]
        if is_datetime64_any_dtype(s):
            categorical_cols.append(c)
        elif is_bool_dtype(s):
            categorical_cols.append(c)
        elif is_numeric_dtype(s):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return pre, numeric_cols, categorical_cols


def prepare_X(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in X.columns:
        if is_bool_dtype(X[c]):
            X[c] = X[c].astype("object")
        elif is_datetime64_any_dtype(X[c]):
            X[c] = X[c].astype(str)
        elif not is_numeric_dtype(X[c]):
            X[c] = X[c].astype("object")
    return X


def choose_svd_components(n_rows: int, n_features: int, max_components: int = 64) -> int:
    upper = min(max_components, n_rows - 1, n_features - 1)
    return max(0, upper)


def fit_embedding_pipeline(X: pd.DataFrame):
    preprocessor, numeric_cols, categorical_cols = make_preprocessor(X)
    Xt = preprocessor.fit_transform(X)

    n_rows = X.shape[0]
    n_features = Xt.shape[1]
    n_components = choose_svd_components(n_rows, n_features)

    svd = None
    embedding = Xt
    if sparse.issparse(Xt) and n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        embedding = svd.fit_transform(Xt)
    elif not sparse.issparse(Xt):
        if hasattr(Xt, "toarray"):
            embedding = Xt.toarray()
        else:
            embedding = np.asarray(Xt)

    if sparse.issparse(embedding):
        embedding = embedding.toarray()

    return preprocessor, svd, embedding, numeric_cols, categorical_cols


def build_neighbor_table(
    df: pd.DataFrame,
    id_col: str,
    label_cols: List[str],
    embedding: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, len(df)), metric="cosine")
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)

    records = []
    for i in range(len(df)):
        src = df.iloc[i]
        rank = 0
        for d, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            rank += 1
            nbr = df.iloc[j]
            rec = {
                "source_id": src[id_col],
                "neighbor_rank": rank,
                "neighbor_id": nbr[id_col],
                "cosine_distance": float(d),
            }
            for col in label_cols:
                if col in df.columns:
                    rec[f"source_{col}"] = src[col]
                    rec[f"neighbor_{col}"] = nbr[col]
            records.append(rec)
            if rank >= top_k:
                break
    return pd.DataFrame(records), nn


def save_embedding_csv(df: pd.DataFrame, id_col: str, embedding: np.ndarray, out_path: Path) -> None:
    emb_cols = [f"emb_{i:03d}" for i in range(embedding.shape[1])]
    emb_df = pd.DataFrame(embedding, columns=emb_cols)
    emb_df.insert(0, id_col, df[id_col].values)
    emb_df.to_csv(out_path, index=False)


def meal_space_spec(project_root: Path) -> Dict:
    manifest = read_json_if_exists(project_root / "models" / "baselines" / "meal" / "run_manifest.json")
    feature_cols = None
    if manifest is not None:
        feature_cols = manifest.get("feature_columns")

    if not feature_cols:
        # fallback if manifest unavailable
        df = read_csv_required(project_root / "training" / "predictive_views" / "meal_prediction_view.csv")
        feature_cols = []
        for col in df.columns:
            if col in MEAL_STATE_EXCLUDE:
                continue
            if col.startswith("state_"):
                feature_cols.append(col)
            elif col in MEAL_SAFE_DECISION_FEATURES:
                feature_cols.append(col)

    return {
        "name": "meal_states",
        "path": project_root / "training" / "predictive_views" / "meal_prediction_view.csv",
        "id_col": "meal_id",
        "feature_cols": feature_cols,
        "label_cols": [
            "decision_time",
            "time_slot_label",
            "target_meal_text",
            "target_meal_archetype_primary",
            "target_cuisine_primary",
            "target_service_form_primary",
            "target_calories_kcal",
            "y_next_meal_family_coarse",
        ],
    }


def generic_space_spec(name: str, path: Path, id_col: str, exclude_cols: List[str], label_cols: List[str]) -> Dict:
    df = read_csv_required(path)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return {
        "name": name,
        "path": path,
        "id_col": id_col,
        "feature_cols": feature_cols,
        "label_cols": label_cols,
    }


def build_space(project_root: Path, spec: Dict, top_k: int = 10) -> None:
    name = spec["name"]
    path = spec["path"]
    id_col = spec["id_col"]
    feature_cols = spec["feature_cols"]
    label_cols = spec["label_cols"]

    out_dir = project_root / "models" / "retrieval" / name
    ensure_dir(out_dir)

    log(f"Building retrieval space: {name}")
    df = read_csv_required(path)

    if id_col not in df.columns:
        raise ValueError(f"{name}: missing id column '{id_col}' in {path}")

    # convert key date-like labels for readability
    for c in ["date", "decision_time", "week_start", "week_end", "weekend_start", "weekend_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    X = prepare_X(df, feature_cols)
    preprocessor, svd, embedding, numeric_cols, categorical_cols = fit_embedding_pipeline(X)

    neighbors_df, nn_model = build_neighbor_table(df, id_col=id_col, label_cols=label_cols, embedding=embedding, top_k=top_k)

    # Save artifacts
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")
    joblib.dump(nn_model, out_dir / "nearest_neighbors.joblib")
    if svd is not None:
        joblib.dump(svd, out_dir / "svd.joblib")

    # Save transformed matrix
    save_embedding_csv(df, id_col=id_col, embedding=embedding, out_path=out_dir / "embeddings.csv")
    neighbors_df.to_csv(out_dir / "neighbors_topk.csv", index=False)

    index_cols = [id_col] + [c for c in label_cols if c in df.columns]
    df[index_cols].to_csv(out_dir / "index_rows.csv", index=False)

    manifest = {
        "name": name,
        "source_table": str(path.name),
        "rows": int(len(df)),
        "id_col": id_col,
        "feature_count": int(len(feature_cols)),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "embedding_dim": int(embedding.shape[1]),
        "top_k_neighbors": int(top_k),
        "feature_columns": feature_cols,
        "label_columns": [c for c in label_cols if c in df.columns],
        "notes": "Cosine nearest-neighbor retrieval baseline built from standardized numeric + one-hot categorical features with optional TruncatedSVD.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(f"Wrote retrieval artifacts to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval baselines for meals, days, weeks, and weekends.")
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument("--top-k", type=int, default=10, help="Neighbors to save per row.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    ensure_dir(project_root / "models" / "retrieval")

    # Space specs
    meal_spec = meal_space_spec(project_root)

    day_spec = generic_space_spec(
        name="days",
        path=project_root / "training" / "day_feature_matrix.csv",
        id_col="date",
        exclude_cols=["date"],
        label_cols=[
            "day_of_week",
            "season",
            "meal_event_count",
            "dominant_meal_archetype",
            "dominant_cuisine",
            "true_weight_lb",
        ],
    )

    week_spec = generic_space_spec(
        name="weeks",
        path=project_root / "training" / "week_summary_matrix.csv",
        id_col="week_id",
        exclude_cols=["week_id", "week_start", "week_end", "week_label"],
        label_cols=[
            "week_start",
            "week_end",
            "week_label",
            "dominant_meal_archetype_week",
            "dominant_cuisine_week",
            "weight_delta_lb",
            "meal_events_per_day_week",
        ],
    )

    weekend_spec = generic_space_spec(
        name="weekends",
        path=project_root / "training" / "weekend_summary_matrix.csv",
        id_col="weekend_id",
        exclude_cols=["weekend_id", "weekend_start", "weekend_end", "weekend_label"],
        label_cols=[
            "weekend_start",
            "weekend_end",
            "weekend_label",
            "dominant_meal_archetype_weekend",
            "dominant_cuisine_weekend",
            "weight_delta_lb",
            "meal_events_per_day_weekend",
        ],
    )

    for spec in [meal_spec, day_spec, week_spec, weekend_spec]:
        build_space(project_root, spec, top_k=args.top_k)

    log("Done.")


if __name__ == "__main__":
    main()
