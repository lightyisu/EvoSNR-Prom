#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_PER_BASE_PATH = Path("/root/autodl-tmp/evosnr_0605/Experiment/results/genome_scan_plus_two_stage_full/agro_linear.per_base.tsv")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/evosnr_0605/Experiment/results/fixed_window_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate per-base scan results on fixed genomic windows.")
    parser.add_argument("--per-base-tsv", type=Path, default=DEFAULT_PER_BASE_PATH)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_per_base_tsv(path: Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    sequence_id: str | None = None
    probabilities: list[float] = []
    pred_labels: list[int] = []
    true_labels: list[int] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if sequence_id is None:
                sequence_id = row["sequence_id"]
            probabilities.append(float(row["positive_probability"]))
            pred_labels.append(int(row["pred_label"]))
            true_labels.append(int(row["true_label"]))
    if sequence_id is None:
        raise ValueError(f"No rows found in {path}")
    return sequence_id, np.asarray(probabilities, dtype=np.float32), np.asarray(pred_labels, dtype=np.int8), np.asarray(true_labels, dtype=np.int8)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    tp = int(np.logical_and(y_true_bool, y_pred_bool).sum())
    tn = int(np.logical_and(~y_true_bool, ~y_pred_bool).sum())
    fp = int(np.logical_and(~y_true_bool, y_pred_bool).sum())
    fn = int(np.logical_and(y_true_bool, ~y_pred_bool).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "jaccard": jaccard,
        "mcc": mcc,
    }


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true_bool = y_true.astype(bool)
    positives = int(y_true_bool.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true_bool[order]
    true_positives = np.cumsum(sorted_true)
    ranks = np.arange(1, len(sorted_true) + 1)
    precisions = true_positives / ranks
    return float(precisions[sorted_true].sum() / positives)


def window_rows(sequence_id: str, probabilities: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray, window_size: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for start in range(0, len(probabilities), window_size):
        end = min(start + window_size, len(probabilities))
        chunk_probs = probabilities[start:end]
        chunk_pred = pred_labels[start:end]
        chunk_true = true_labels[start:end]
        metrics = binary_metrics(chunk_true, chunk_pred)
        rows.append(
            {
                "sequence_id": sequence_id,
                "start_1based": start + 1,
                "end_1based": end,
                "length": end - start,
                "true_positive_bp": int(chunk_true.sum()),
                "pred_positive_bp": int(chunk_pred.sum()),
                "max_probability": float(chunk_probs.max()),
                "mean_probability": float(chunk_probs.mean()),
                "auprc": average_precision(chunk_true, chunk_probs),
                **metrics,
            }
        )
    return rows


def mean_metric(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def aggregate_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        "tp": sum(int(row["tp"]) for row in rows),
        "tn": sum(int(row["tn"]) for row in rows),
        "fp": sum(int(row["fp"]) for row in rows),
        "fn": sum(int(row["fn"]) for row in rows),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    y_true = np.array([1] * counts["tp"] + [1] * counts["fn"] + [0] * counts["fp"] + [0] * counts["tn"], dtype=np.int8)
    y_pred = np.array([1] * counts["tp"] + [0] * counts["fn"] + [1] * counts["fp"] + [0] * counts["tn"], dtype=np.int8)
    return binary_metrics(y_true, y_pred)


def write_window_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "start_1based",
        "end_1based",
        "length",
        "true_positive_bp",
        "pred_positive_bp",
        "max_probability",
        "mean_probability",
        "auprc",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "jaccard",
        "mcc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(rows: list[dict[str, object]], window_size: int, per_base_path: Path) -> dict[str, object]:
    active_rows = [row for row in rows if int(row["true_positive_bp"]) > 0 or int(row["pred_positive_bp"]) > 0]
    true_rows = [row for row in rows if int(row["true_positive_bp"]) > 0]
    all_counts = aggregate_counts(rows)
    active_counts = aggregate_counts(active_rows)
    return {
        "per_base_tsv": str(per_base_path),
        "window_size": window_size,
        "window_count": len(rows),
        "active_window_count": len(active_rows),
        "true_window_count": len(true_rows),
        "micro_all": metrics_from_counts(all_counts),
        "micro_active": metrics_from_counts(active_counts) if active_rows else metrics_from_counts({"tp": 0, "tn": 0, "fp": 0, "fn": 0}),
        "macro_all": {
            "accuracy": mean_metric(rows, "accuracy"),
            "precision": mean_metric(rows, "precision"),
            "recall": mean_metric(rows, "recall"),
            "f1": mean_metric(rows, "f1"),
            "jaccard": mean_metric(rows, "jaccard"),
            "mcc": mean_metric(rows, "mcc"),
            "auprc": mean_metric(rows, "auprc"),
        },
        "macro_active": {
            "accuracy": mean_metric(active_rows, "accuracy"),
            "precision": mean_metric(active_rows, "precision"),
            "recall": mean_metric(active_rows, "recall"),
            "f1": mean_metric(active_rows, "f1"),
            "jaccard": mean_metric(active_rows, "jaccard"),
            "mcc": mean_metric(active_rows, "mcc"),
            "auprc": mean_metric(active_rows, "auprc"),
        },
        "macro_true_windows": {
            "accuracy": mean_metric(true_rows, "accuracy"),
            "precision": mean_metric(true_rows, "precision"),
            "recall": mean_metric(true_rows, "recall"),
            "f1": mean_metric(true_rows, "f1"),
            "jaccard": mean_metric(true_rows, "jaccard"),
            "mcc": mean_metric(true_rows, "mcc"),
            "auprc": mean_metric(true_rows, "auprc"),
        },
    }


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    sequence_id, probabilities, pred_labels, true_labels = read_per_base_tsv(args.per_base_tsv)
    rows = window_rows(sequence_id, probabilities, pred_labels, true_labels, args.window_size)

    output_prefix = args.output_dir / args.per_base_tsv.stem.replace('.per_base', '')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_prefix.with_name(f"{output_prefix.name}.window_{args.window_size}.tsv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}.window_{args.window_size}.summary.json")

    write_window_table(table_path, rows)
    summary = build_summary(rows, args.window_size, args.per_base_tsv)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Window table written to: {table_path}")
    print(f"Window summary written to: {summary_path}")
    print(
        f"micro_all: F1={summary['micro_all']['f1']:.6f} MCC={summary['micro_all']['mcc']:.6f}; "
        f"macro_active: F1={summary['macro_active']['f1']:.6f} MCC={summary['macro_active']['mcc']:.6f}"
    )


if __name__ == "__main__":
    main()
