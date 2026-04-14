from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


TARGET_NAME = "y_next_weight_loss_flag"
MODALITIES = ["days", "weeks"]
BASELINE_RUNS = [
    "simple_loss_daysweeks_v2",
    "gru_loss_daysweeks_smoke_v4_1",
]
SUPPORTED_FAMILIES = ["gru", "tcn", "transformer"]


def log(msg: str) -> None:
    print(f"[temporal-neural-compare] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_summary(project_root: Path, run_name: str) -> Dict:
    report_dir = project_root / "reports" / "backtests" / "temporal_multires" / run_name
    final_summary_path = report_dir / "final_summary.json"
    if not final_summary_path.exists():
        raise FileNotFoundError(f"Missing required report artifact: {final_summary_path}")
    final_summary = read_json(final_summary_path)
    test_metrics_tuned = read_json(report_dir / "test_metrics_tuned.json")
    target_metrics = test_metrics_tuned.get("binary", {}).get(TARGET_NAME)
    if not target_metrics:
        raise ValueError(f"Run {run_name} does not contain tuned binary metrics for {TARGET_NAME}")
    return {
        "report_dir": str(report_dir),
        "final_summary": final_summary,
        "target_metrics": target_metrics,
    }


def build_comparison_row(
    run_name: str,
    source_kind: str,
    model_family: str,
    payload: Dict,
    floor_metrics: Dict,
    gru_reference_metrics: Dict,
) -> Dict:
    metrics = payload["target_metrics"]
    return {
        "run_name": run_name,
        "source_kind": source_kind,
        "model_family": model_family,
        "target": TARGET_NAME,
        "modalities": ",".join(MODALITIES),
        "report_dir": payload["report_dir"],
        "balanced_accuracy_tuned": metrics.get("balanced_accuracy"),
        "roc_auc_tuned": metrics.get("roc_auc"),
        "f1_tuned": metrics.get("f1"),
        "threshold": metrics.get("threshold"),
        "positive_rate_true": metrics.get("positive_rate_true"),
        "positive_rate_pred": metrics.get("positive_rate_pred"),
        "prob_mean": metrics.get("prob_mean"),
        "prob_std": metrics.get("prob_std"),
        "prob_q05": metrics.get("prob_q05"),
        "prob_q50": metrics.get("prob_q50"),
        "prob_q95": metrics.get("prob_q95"),
        "positive_rate_pred_at_0_5": metrics.get("positive_rate_pred_at_0_5"),
        "positive_rate_pred_at_threshold": metrics.get("positive_rate_pred_at_threshold"),
        "delta_vs_simple_floor_balanced_accuracy": (
            None
            if metrics.get("balanced_accuracy") is None
            else metrics.get("balanced_accuracy") - floor_metrics.get("balanced_accuracy")
        ),
        "delta_vs_simple_floor_roc_auc": (
            None if metrics.get("roc_auc") is None else metrics.get("roc_auc") - floor_metrics.get("roc_auc")
        ),
        "delta_vs_gru_v4_1_balanced_accuracy": (
            None
            if metrics.get("balanced_accuracy") is None
            else metrics.get("balanced_accuracy") - gru_reference_metrics.get("balanced_accuracy")
        ),
        "delta_vs_gru_v4_1_roc_auc": (
            None
            if metrics.get("roc_auc") is None
            else metrics.get("roc_auc") - gru_reference_metrics.get("roc_auc")
        ),
    }


def build_markdown_summary(
    comparison_run_name: str,
    executed_run_names: List[str],
    comparison_rows: List[Dict],
    smoke_test: bool,
) -> str:
    df = pd.DataFrame(comparison_rows)
    cols = [
        "run_name",
        "source_kind",
        "model_family",
        "balanced_accuracy_tuned",
        "roc_auc_tuned",
        "f1_tuned",
        "prob_std",
        "delta_vs_simple_floor_balanced_accuracy",
        "delta_vs_simple_floor_roc_auc",
        "delta_vs_gru_v4_1_balanced_accuracy",
        "delta_vs_gru_v4_1_roc_auc",
    ]
    display_df = df[cols].copy()
    table_lines = [
        "| " + " | ".join(display_df.columns.astype(str).tolist()) + " |",
        "| " + " | ".join(["---"] * len(display_df.columns)) + " |",
    ]
    for row in display_df.itertuples(index=False, name=None):
        table_lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join([
        f"# Temporal Neural Comparison: {comparison_run_name}",
        "",
        f"- target: `{TARGET_NAME}`",
        f"- modalities: `{','.join(MODALITIES)}`",
        "- binary-only: `true`",
        "- regression-heads: `none`",
        f"- smoke_test: `{str(smoke_test).lower()}`",
        f"- executed_neural_runs: `{', '.join(executed_run_names)}`",
        "- required_reference_runs: `simple_loss_daysweeks_v2`, `gru_loss_daysweeks_smoke_v4_1`",
        "",
        "## Comparison",
        "",
        *table_lines,
        "",
        "## Interpretation",
        "",
        "- `delta_vs_simple_floor_*` shows whether a neural run cleared the current conservative floor.",
        "- `delta_vs_gru_v4_1_*` shows whether a neural run improved on the best existing GRU loss smoke reference.",
        "- `prob_std` remains in the comparison because under-dispersed probabilities were the main prior neural failure mode.",
        "",
    ])


def build_trainer_command(
    trainer_script: Path,
    project_root: Path,
    dataset_dir: str,
    run_name: str,
    model_family: str,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable,
        str(trainer_script),
        "--project-root",
        str(project_root),
        "--dataset-dir",
        dataset_dir,
        "--run-name",
        run_name,
        "--model-family",
        model_family,
        "--modalities",
        ",".join(MODALITIES),
        "--single-binary-target",
        TARGET_NAME,
        "--regression-targets",
        "none",
        "--hidden-dim",
        str(args.hidden_dim),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--binary-loss-mode",
        args.binary_loss_mode,
        "--focal-gamma",
        str(args.focal_gamma),
        "--grad-clip",
        str(args.grad_clip),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--balanced-sampler",
    ]
    if args.smoke_test:
        cmd.append("--smoke-test")
    if args.amp:
        cmd.append("--amp")
    if args.force_amp:
        cmd.append("--force-amp")
    if args.compile:
        cmd.append("--compile")
    if args.force_cpu:
        cmd.append("--force-cpu")
    if args.resume:
        cmd.append("--resume")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a focused days+weeks neural comparison for y_next_weight_loss_flag and compare it to the current temporal floor."
    )
    parser.add_argument("--project-root", default=".", help="Path to project root.")
    parser.add_argument(
        "--dataset-dir",
        default="training/multires_sequence_dataset",
        help="Relative path to the multires dataset directory.",
    )
    parser.add_argument(
        "--comparison-run-name",
        default="",
        help="Optional aggregate comparison run name. Auto-generated if omitted.",
    )
    parser.add_argument(
        "--families",
        default="gru,tcn,transformer",
        help="Comma-separated neural families to evaluate.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="",
        help="Optional prefix for per-family run names. Auto-generated if omitted.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--binary-loss-mode", choices=["bce", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--force-amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    trainer_script = project_root / "train_temporal_multires_models_v4_1.py"
    if not trainer_script.exists():
        raise FileNotFoundError(f"Missing trainer dependency: {trainer_script}")

    families = parse_csv_list(args.families)
    if not families:
        raise ValueError("At least one neural family is required.")
    unknown = [family for family in families if family not in SUPPORTED_FAMILIES]
    if unknown:
        raise ValueError(f"Unsupported family values: {unknown}. Supported: {SUPPORTED_FAMILIES}")

    run_suffix = "smoke" if args.smoke_test else "pilot"
    prefix = args.run_name_prefix or f"loss_daysweeks_compare_{run_suffix}_v1"
    comparison_run_name = args.comparison_run_name or prefix

    aggregate_report_dir = project_root / "reports" / "backtests" / "temporal_multires" / comparison_run_name
    ensure_dir(aggregate_report_dir)

    commands = []
    executed_run_names = []
    for family in families:
        run_name = f"{family}_{prefix}"
        executed_run_names.append(run_name)
        cmd = build_trainer_command(
            trainer_script=trainer_script,
            project_root=project_root,
            dataset_dir=args.dataset_dir,
            run_name=run_name,
            model_family=family,
            args=args,
        )
        commands.append({"family": family, "run_name": run_name, "command": cmd})

    save_json(
        aggregate_report_dir / "comparison_config.json",
        {
            "comparison_run_name": comparison_run_name,
            "target": TARGET_NAME,
            "modalities": MODALITIES,
            "binary_only": True,
            "regression_targets": [],
            "families": families,
            "executed_run_names": executed_run_names,
            "baseline_runs": BASELINE_RUNS,
            "trainer_script": str(trainer_script),
            "dataset_dir": args.dataset_dir,
            "smoke_test": args.smoke_test,
            "trainer_commands": commands,
        },
    )

    for item in commands:
        log(f"Running {item['family']} as {item['run_name']}")
        subprocess.run(item["command"], check=True, cwd=str(project_root))

    baseline_payload = load_run_summary(project_root, "simple_loss_daysweeks_v2")
    gru_reference_payload = load_run_summary(project_root, "gru_loss_daysweeks_smoke_v4_1")
    floor_metrics = baseline_payload["target_metrics"]
    gru_reference_metrics = gru_reference_payload["target_metrics"]

    comparison_rows: List[Dict] = []
    comparison_rows.append(
        build_comparison_row(
            run_name="simple_loss_daysweeks_v2",
            source_kind="reference_baseline",
            model_family="simple_floor",
            payload=baseline_payload,
            floor_metrics=floor_metrics,
            gru_reference_metrics=gru_reference_metrics,
        )
    )
    comparison_rows.append(
        build_comparison_row(
            run_name="gru_loss_daysweeks_smoke_v4_1",
            source_kind="historical_neural_reference",
            model_family="gru",
            payload=gru_reference_payload,
            floor_metrics=floor_metrics,
            gru_reference_metrics=gru_reference_metrics,
        )
    )

    for item in commands:
        payload = load_run_summary(project_root, item["run_name"])
        comparison_rows.append(
            build_comparison_row(
                run_name=item["run_name"],
                source_kind="new_neural_run",
                model_family=item["family"],
                payload=payload,
                floor_metrics=floor_metrics,
                gru_reference_metrics=gru_reference_metrics,
            )
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(aggregate_report_dir / "comparison_summary.csv", index=False)

    markdown = build_markdown_summary(
        comparison_run_name=comparison_run_name,
        executed_run_names=executed_run_names,
        comparison_rows=comparison_rows,
        smoke_test=args.smoke_test,
    )
    (aggregate_report_dir / "comparison_report.md").write_text(markdown, encoding="utf-8")

    final_summary = {
        "comparison_run_name": comparison_run_name,
        "target": TARGET_NAME,
        "modalities": MODALITIES,
        "binary_only": True,
        "regression_targets": [],
        "families": families,
        "executed_run_names": executed_run_names,
        "baseline_runs": BASELINE_RUNS,
        "comparison_report": str(aggregate_report_dir / "comparison_report.md"),
        "comparison_summary_csv": str(aggregate_report_dir / "comparison_summary.csv"),
    }
    save_json(aggregate_report_dir / "final_summary.json", final_summary)

    log(f"Comparison report written to {aggregate_report_dir / 'comparison_report.md'}")
    log(f"Comparison table written to {aggregate_report_dir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
