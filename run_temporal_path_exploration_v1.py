from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_RUN_NAME = "temporal_path_explore_v1"
TARGET_NAME = "y_next_weight_loss_flag"
REFERENCE_RUNS = [
    "simple_loss_daysweeks_v2",
    "gru_loss_daysweeks_smoke_v4_1",
    "tcn_loss_daysweeks_compare_smoke_v1_check",
    "tcn_loss_daysweeks_compare_pilot_v1",
]


def log(msg: str) -> None:
    print(f"[temporal-path-explore] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def report_dir_for(project_root: Path, run_name: str) -> Path:
    return project_root / "reports" / "backtests" / "temporal_multires" / run_name


def load_binary_metrics(project_root: Path, run_name: str) -> Dict:
    report_dir = report_dir_for(project_root, run_name)
    payload = read_json(report_dir / "test_metrics_tuned.json")
    metrics = payload.get("binary", {}).get(TARGET_NAME)
    if not metrics:
        raise ValueError(f"Run {run_name} is missing tuned binary metrics for {TARGET_NAME}")
    return metrics


def load_flattened_candidates(project_root: Path, run_name: str) -> pd.DataFrame:
    path = report_dir_for(project_root, run_name) / "model_comparison.csv"
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No candidate rows found in {path}")
    return df


def is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def evaluate_status(bal_acc: float, roc_auc: float, prob_std: float, positive_rate_pred: float) -> Dict[str, str]:
    reasons: List[str] = []
    promoted = True
    if is_nan(roc_auc) or is_nan(bal_acc):
        promoted = False
        reasons.append("missing core metrics")
    if positive_rate_pred <= 0.02 or positive_rate_pred >= 0.98:
        promoted = False
        reasons.append("effectively one-class predictions")
    if not is_nan(prob_std) and prob_std < 0.01:
        promoted = False
        reasons.append("under-dispersed probabilities")
    if not is_nan(roc_auc) and roc_auc < 0.55:
        promoted = False
        reasons.append("ranking too weak")
    if not is_nan(bal_acc) and bal_acc < 0.55:
        promoted = False
        reasons.append("balanced accuracy below promotion floor")
    return {
        "status": "promote" if promoted else "reject",
        "reason": "; ".join(reasons) if reasons else "clears bounded smoke criteria",
    }


def build_plan_markdown() -> str:
    return "\n".join(
        [
            f"# Temporal Path Exploration Plan: {PROJECT_RUN_NAME}",
            "",
            "## Mission",
            "",
            "- Explore bounded, directly comparable paths for `y_next_weight_loss_flag` on `days,weeks`.",
            "- Prune collapsed or weak paths quickly.",
            "- Promote at most the top 1-2 surviving directions to the next pilot stage.",
            "",
            "## Reference Runs",
            "",
            "- `simple_loss_daysweeks_v2`",
            "- `gru_loss_daysweeks_smoke_v4_1`",
            "- `tcn_loss_daysweeks_compare_smoke_v1_check`",
            "- `tcn_loss_daysweeks_compare_pilot_v1`",
            "",
            "## Matrix",
            "",
            "1. Path A/B: stronger flattened baselines on the current lag-window representation.",
            "2. Path C/D: cheap neural smoke variants only on `days,weeks` and the loss target.",
            "3. No meals, no regression, no multi-head runs in this loop.",
            "",
            "## Candidate Runs",
            "",
            "- `flat_loss_daysweeks_explore_v1`: logistic, random forest, extra trees, histogram boosting, small MLPs on flattened windows.",
            "- `gru_loss_daysweeks_bce_smoke_v1`: GRU smoke with BCE instead of focal.",
            "- `tcn_loss_daysweeks_bce_smoke_v1`: TCN smoke with BCE instead of focal.",
            "- `tcn_loss_daysweeks_bce_deep_smoke_v1`: TCN smoke with BCE plus a slightly larger hidden state.",
            "",
            "## Promotion Criteria",
            "",
            "- no obvious collapse",
            "- interpretable metrics",
            "- balanced accuracy at least `0.55`",
            "- ROC AUC at least `0.55`",
            "- materially stronger than the current weak neural ceiling",
            "",
            "## Stop Rules",
            "",
            "- reject any run with effectively one-class predictions",
            "- reject any run with `prob_std < 0.01`",
            "- reject any run with missing or non-finite metrics",
            "- do not escalate to a longer pilot unless a smoke run survives the criteria above",
            "",
        ]
    )


def build_command_matrix(project_root: Path) -> List[Dict]:
    return [
        {
            "name": "flat_loss_daysweeks_explore_v1",
            "path_family": "flattened_explore",
            "command": [
                sys.executable,
                str(project_root / "train_temporal_multires_flattened_explore_v1.py"),
                "--project-root",
                str(project_root),
                "--run-name",
                "flat_loss_daysweeks_explore_v1",
            ],
        },
        {
            "name": "gru_loss_daysweeks_bce_smoke_v1",
            "path_family": "neural_sequence",
            "command": [
                sys.executable,
                str(project_root / "train_temporal_multires_models_v4_1.py"),
                "--project-root",
                str(project_root),
                "--run-name",
                "gru_loss_daysweeks_bce_smoke_v1",
                "--model-family",
                "gru",
                "--modalities",
                "days,weeks",
                "--single-binary-target",
                TARGET_NAME,
                "--regression-targets",
                "none",
                "--hidden-dim",
                "64",
                "--num-layers",
                "1",
                "--dropout",
                "0.05",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.0002",
                "--weight-decay",
                "0.0001",
                "--warmup-epochs",
                "2",
                "--max-epochs",
                "12",
                "--patience",
                "4",
                "--binary-loss-mode",
                "bce",
                "--grad-clip",
                "0.5",
                "--seed",
                "42",
                "--balanced-sampler",
                "--smoke-test",
            ],
        },
        {
            "name": "tcn_loss_daysweeks_bce_smoke_v1",
            "path_family": "neural_sequence",
            "command": [
                sys.executable,
                str(project_root / "train_temporal_multires_models_v4_1.py"),
                "--project-root",
                str(project_root),
                "--run-name",
                "tcn_loss_daysweeks_bce_smoke_v1",
                "--model-family",
                "tcn",
                "--modalities",
                "days,weeks",
                "--single-binary-target",
                TARGET_NAME,
                "--regression-targets",
                "none",
                "--hidden-dim",
                "64",
                "--num-layers",
                "1",
                "--dropout",
                "0.05",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.0002",
                "--weight-decay",
                "0.0001",
                "--warmup-epochs",
                "2",
                "--max-epochs",
                "12",
                "--patience",
                "4",
                "--binary-loss-mode",
                "bce",
                "--grad-clip",
                "0.5",
                "--seed",
                "42",
                "--balanced-sampler",
                "--smoke-test",
            ],
        },
        {
            "name": "tcn_loss_daysweeks_bce_deep_smoke_v1",
            "path_family": "neural_sequence",
            "command": [
                sys.executable,
                str(project_root / "train_temporal_multires_models_v4_1.py"),
                "--project-root",
                str(project_root),
                "--run-name",
                "tcn_loss_daysweeks_bce_deep_smoke_v1",
                "--model-family",
                "tcn",
                "--modalities",
                "days,weeks",
                "--single-binary-target",
                TARGET_NAME,
                "--regression-targets",
                "none",
                "--hidden-dim",
                "96",
                "--num-layers",
                "2",
                "--dropout",
                "0.10",
                "--batch-size",
                "64",
                "--learning-rate",
                "0.0002",
                "--weight-decay",
                "0.0001",
                "--warmup-epochs",
                "2",
                "--max-epochs",
                "12",
                "--patience",
                "4",
                "--binary-loss-mode",
                "bce",
                "--grad-clip",
                "0.5",
                "--seed",
                "42",
                "--balanced-sampler",
                "--smoke-test",
            ],
        },
    ]


def run_command(spec: Dict, output_root: Path) -> Dict:
    existing_final_summary = report_dir_for(Path(".").resolve(), spec["name"]) / "final_summary.json"
    existing_comparison = report_dir_for(Path(".").resolve(), spec["name"]) / "model_comparison.csv"
    if existing_final_summary.exists() or existing_comparison.exists():
        log(f"Skipping {spec['name']} because report artifacts already exist")
        return {
            "name": spec["name"],
            "path_family": spec["path_family"],
            "stdout_log": None,
            "stderr_log": None,
            "skipped_existing": True,
        }
    log(f"Running {spec['name']}")
    proc = subprocess.run(spec["command"], capture_output=True, text=True)
    log_path = output_root / f"{spec['name']}.log"
    err_path = output_root / f"{spec['name']}.stderr.log"
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    err_path.write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{spec['name']} failed with exit code {proc.returncode}. See {log_path} and {err_path}.")
    return {
        "name": spec["name"],
        "path_family": spec["path_family"],
        "stdout_log": str(log_path),
        "stderr_log": str(err_path),
    }


def build_reference_rows(project_root: Path, simple_floor: Dict, gru_ref: Dict, tcn_smoke: Dict, tcn_pilot: Dict) -> List[Dict]:
    refs = []
    for run_name in REFERENCE_RUNS:
        metrics = load_binary_metrics(project_root, run_name)
        status = evaluate_status(
            bal_acc=float(metrics.get("balanced_accuracy")),
            roc_auc=float(metrics.get("roc_auc")) if not is_nan(metrics.get("roc_auc")) else float("nan"),
            prob_std=float(metrics.get("prob_std")) if not is_nan(metrics.get("prob_std")) else float("nan"),
            positive_rate_pred=float(metrics.get("positive_rate_pred")),
        )
        refs.append(
            {
                "candidate_name": run_name,
                "run_name": run_name,
                "path_family": "reference",
                "model_family": "reference",
                "source_kind": "reference",
                "balanced_accuracy_tuned": metrics.get("balanced_accuracy"),
                "roc_auc_tuned": metrics.get("roc_auc"),
                "f1_tuned": metrics.get("f1"),
                "prob_std": metrics.get("prob_std"),
                "positive_rate_pred": metrics.get("positive_rate_pred"),
                "status": status["status"],
                "status_reason": status["reason"],
                "delta_vs_simple_floor_balanced_accuracy": metrics.get("balanced_accuracy") - simple_floor.get("balanced_accuracy"),
                "delta_vs_simple_floor_roc_auc": metrics.get("roc_auc") - simple_floor.get("roc_auc"),
                "delta_vs_gru_v4_1_balanced_accuracy": metrics.get("balanced_accuracy") - gru_ref.get("balanced_accuracy"),
                "delta_vs_gru_v4_1_roc_auc": metrics.get("roc_auc") - gru_ref.get("roc_auc"),
                "delta_vs_tcn_smoke_balanced_accuracy": metrics.get("balanced_accuracy") - tcn_smoke.get("balanced_accuracy"),
                "delta_vs_tcn_smoke_roc_auc": metrics.get("roc_auc") - tcn_smoke.get("roc_auc"),
                "delta_vs_tcn_pilot_balanced_accuracy": metrics.get("balanced_accuracy") - tcn_pilot.get("balanced_accuracy"),
                "delta_vs_tcn_pilot_roc_auc": metrics.get("roc_auc") - tcn_pilot.get("roc_auc"),
            }
        )
    return refs


def main() -> None:
    project_root = Path(".").resolve()
    output_root = report_dir_for(project_root, PROJECT_RUN_NAME)
    ensure_dir(output_root)
    (output_root / "experiment_plan.md").write_text(build_plan_markdown(), encoding="utf-8")

    simple_floor = load_binary_metrics(project_root, "simple_loss_daysweeks_v2")
    gru_ref = load_binary_metrics(project_root, "gru_loss_daysweeks_smoke_v4_1")
    tcn_smoke = load_binary_metrics(project_root, "tcn_loss_daysweeks_compare_smoke_v1_check")
    tcn_pilot = load_binary_metrics(project_root, "tcn_loss_daysweeks_compare_pilot_v1")

    execution_rows: List[Dict] = []
    for spec in build_command_matrix(project_root):
        execution_rows.append(run_command(spec, output_root=output_root))

    rows: List[Dict] = build_reference_rows(project_root, simple_floor, gru_ref, tcn_smoke, tcn_pilot)

    flat_run_name = "flat_loss_daysweeks_explore_v1"
    flat_candidates = load_flattened_candidates(project_root, flat_run_name)
    for candidate in flat_candidates.itertuples(index=False):
        bal_acc = float(candidate.test_balanced_accuracy_tuned)
        roc_auc = float(candidate.test_roc_auc)
        prob_std = float(candidate.test_prob_std)
        positive_rate_pred = float(candidate.test_positive_rate_pred_tuned)
        status = evaluate_status(
            bal_acc=bal_acc,
            roc_auc=roc_auc,
            prob_std=prob_std,
            positive_rate_pred=positive_rate_pred,
        )
        rows.append(
            {
                "candidate_name": f"{flat_run_name}:{candidate.model_name}",
                "run_name": flat_run_name,
                "path_family": "flattened_explore",
                "model_family": str(candidate.model_name),
                "source_kind": "new_flattened_candidate",
                "balanced_accuracy_tuned": bal_acc,
                "roc_auc_tuned": roc_auc,
                "f1_tuned": float(candidate.test_f1_tuned),
                "prob_std": prob_std,
                "positive_rate_pred": positive_rate_pred,
                "status": status["status"],
                "status_reason": status["reason"],
                "delta_vs_simple_floor_balanced_accuracy": bal_acc - simple_floor.get("balanced_accuracy"),
                "delta_vs_simple_floor_roc_auc": roc_auc - simple_floor.get("roc_auc"),
                "delta_vs_gru_v4_1_balanced_accuracy": bal_acc - gru_ref.get("balanced_accuracy"),
                "delta_vs_gru_v4_1_roc_auc": roc_auc - gru_ref.get("roc_auc"),
                "delta_vs_tcn_smoke_balanced_accuracy": bal_acc - tcn_smoke.get("balanced_accuracy"),
                "delta_vs_tcn_smoke_roc_auc": roc_auc - tcn_smoke.get("roc_auc"),
                "delta_vs_tcn_pilot_balanced_accuracy": bal_acc - tcn_pilot.get("balanced_accuracy"),
                "delta_vs_tcn_pilot_roc_auc": roc_auc - tcn_pilot.get("roc_auc"),
            }
        )

    for run_name in [
        "gru_loss_daysweeks_bce_smoke_v1",
        "tcn_loss_daysweeks_bce_smoke_v1",
        "tcn_loss_daysweeks_bce_deep_smoke_v1",
    ]:
        metrics = load_binary_metrics(project_root, run_name)
        status = evaluate_status(
            bal_acc=float(metrics.get("balanced_accuracy")),
            roc_auc=float(metrics.get("roc_auc")) if not is_nan(metrics.get("roc_auc")) else float("nan"),
            prob_std=float(metrics.get("prob_std")) if not is_nan(metrics.get("prob_std")) else float("nan"),
            positive_rate_pred=float(metrics.get("positive_rate_pred")),
        )
        rows.append(
            {
                "candidate_name": run_name,
                "run_name": run_name,
                "path_family": "neural_sequence",
                "model_family": run_name.split("_")[0],
                "source_kind": "new_neural_run",
                "balanced_accuracy_tuned": metrics.get("balanced_accuracy"),
                "roc_auc_tuned": metrics.get("roc_auc"),
                "f1_tuned": metrics.get("f1"),
                "prob_std": metrics.get("prob_std"),
                "positive_rate_pred": metrics.get("positive_rate_pred"),
                "status": status["status"],
                "status_reason": status["reason"],
                "delta_vs_simple_floor_balanced_accuracy": metrics.get("balanced_accuracy") - simple_floor.get("balanced_accuracy"),
                "delta_vs_simple_floor_roc_auc": metrics.get("roc_auc") - simple_floor.get("roc_auc"),
                "delta_vs_gru_v4_1_balanced_accuracy": metrics.get("balanced_accuracy") - gru_ref.get("balanced_accuracy"),
                "delta_vs_gru_v4_1_roc_auc": metrics.get("roc_auc") - gru_ref.get("roc_auc"),
                "delta_vs_tcn_smoke_balanced_accuracy": metrics.get("balanced_accuracy") - tcn_smoke.get("balanced_accuracy"),
                "delta_vs_tcn_smoke_roc_auc": metrics.get("roc_auc") - tcn_smoke.get("roc_auc"),
                "delta_vs_tcn_pilot_balanced_accuracy": metrics.get("balanced_accuracy") - tcn_pilot.get("balanced_accuracy"),
                "delta_vs_tcn_pilot_roc_auc": metrics.get("roc_auc") - tcn_pilot.get("roc_auc"),
            }
        )

    ranking = pd.DataFrame(rows)
    ranking["status_sort"] = ranking["status"].map({"promote": 0, "reject": 1}).fillna(2)
    ranking = ranking.sort_values(
        by=["status_sort", "roc_auc_tuned", "balanced_accuracy_tuned", "prob_std"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    ranking.to_csv(output_root / "aggregate_ranking.csv", index=False)

    promoted = ranking[(ranking["status"] == "promote") & (~ranking["source_kind"].eq("reference"))]
    top_promoted = promoted.head(2)

    ranking_table = ranking[
        [
            "candidate_name",
            "path_family",
            "source_kind",
            "balanced_accuracy_tuned",
            "roc_auc_tuned",
            "f1_tuned",
            "prob_std",
            "positive_rate_pred",
            "status",
            "status_reason",
            "delta_vs_simple_floor_balanced_accuracy",
            "delta_vs_simple_floor_roc_auc",
            "delta_vs_tcn_smoke_balanced_accuracy",
            "delta_vs_tcn_smoke_roc_auc",
        ]
    ].to_csv(index=False).strip()
    report_lines = [
        f"# Temporal Path Exploration Results: {PROJECT_RUN_NAME}",
        "",
        "## Ranked Results",
        "",
        "```csv",
        ranking_table,
        "```",
        "",
        "## Surviving Paths",
        "",
    ]
    if top_promoted.empty:
        report_lines.extend(["- none", ""])
    else:
        for row in top_promoted.itertuples(index=False):
            report_lines.append(
                f"- `{row.candidate_name}`: ROC AUC `{row.roc_auc_tuned:.4f}`, balanced accuracy `{row.balanced_accuracy_tuned:.4f}`, prob_std `{row.prob_std:.4f}`"
            )
        report_lines.append("")

    (output_root / "aggregate_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    save_json(
        output_root / "final_summary.json",
        {
            "project_run_name": PROJECT_RUN_NAME,
            "target": TARGET_NAME,
            "reference_runs": REFERENCE_RUNS,
            "executed_runs": execution_rows,
            "n_ranked_rows": int(len(ranking)),
            "n_promoted_non_reference": int(len(promoted)),
            "top_promoted_candidates": top_promoted["candidate_name"].tolist(),
        },
    )
    log(f"Wrote aggregate exploration artifacts to {output_root}")


if __name__ == "__main__":
    main()
