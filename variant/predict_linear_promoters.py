#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import models.model_evosnr as model_evosnr
from utils.fasttext_load import get_direct_vector
from utils.lexicon import count_kmers_in_dataset_with_vocab, match_and_embed_with_kmers_batch, readLexicon
from utils.save_load import load_model_evosnr


DEFAULT_MODEL_DIR = Path("/root/autodl-tmp/evosnr_0605/Weights/best_evosnr_Agro_500bp_seed1")
DEFAULT_SPLIT_DIR = Path("/root/autodl-tmp/evosnr_0605/data/Agro/500_split")
DEFAULT_LEXICON_PATH = Path("/root/autodl-tmp/evosnr_0605/data/Agro/motifs/lexicon.txt")
DEFAULT_FASTTEXT_PATH = Path("/root/autodl-tmp/evosnr_0605/data/Agro/fasttext_model/model.bin")
DEFAULT_INPUT_FILES = [
    Path("/root/autodl-tmp/evosnr_0605/data/Agro/genome/agro_linear_promoter_500bp.csv"),
    Path("/root/autodl-tmp/evosnr_0605/data/Agro/genome/agro_linear_promoter_1000bp.csv"),
]
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/evosnr_0605/Experiment/results/linear_promoter_predictions")
SPLIT_KMER_FILES = ("split_train.csv", "split_test.csv")
PLOT_DPI = 330
ARIAL_FONT_PATH = Path("/root/autodl-tmp/ARIAL.TTF")
PREDICTION_LINE_COLOR = "#0B57C0"
THRESHOLD_LINE_COLOR = "#9AA3AF"
ANNOTATION_BASE_COLOR = "#E6E6E8"
ANNOTATION_HIGHLIGHT_COLOR = "#D97757"
GRID_COLOR = "#D5D8DE"
SPINE_COLOR = "#2F343B"
ANNOTATION_TRACK_Y = 0.5
ANNOTATION_TRACK_WIDTH = 8.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict per-base promoter labels for linear promoter windows with EvoSNR.")
    parser.add_argument("--input-files", type=Path, nargs="+", default=DEFAULT_INPUT_FILES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--fasttext-model-path", type=Path, default=DEFAULT_FASTTEXT_PATH)
    parser.add_argument("--mode", choices=("auto", "direct", "scan"), default="auto")
    parser.add_argument("--window-size", type=int, default=500, help="Window size for scan mode.")
    parser.add_argument("--stride", type=int, default=50, help="Stride for scan mode on sequences longer than window size.")
    parser.add_argument("--aggregation", choices=("max", "mean"), default="max")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--post-process-promoters", action="store_true", help="Merge nearby predicted promoter segments and remove short fragments.")
    parser.add_argument("--merge-gap", type=int, default=10, help="Maximum 0-run length to fill between predicted promoter segments during post-processing.")
    parser.add_argument("--min-promoter-length", type=int, default=30, help="Minimum predicted promoter segment length kept after post-processing.")
    parser.add_argument("--save-probs", action="store_true", help="Write per-base positive probabilities into the TSV output.")
    parser.add_argument("--plot-top-k", type=int, default=50, help="Save visualizations for randomly sampled rows per input file.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directory for per-base prediction signal plots.")
    parser.add_argument("--redraw-existing-plot-dir", type=Path, default=None, help="Redraw plots using the existing file names in this directory.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional debug limit per input file.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_labeled_tsv(path: Path, limit_rows: int | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"{path} line {line_no}: expected 2 tab-separated columns, got {len(row)}")
            sequence = row[0].strip().upper()
            labels = row[1].strip()
            if len(sequence) != len(labels):
                raise ValueError(f"{path} line {line_no}: sequence length {len(sequence)} != label length {len(labels)}")
            if any(base not in {"A", "C", "G", "T", "N"} for base in sequence):
                raise ValueError(f"{path} line {line_no}: sequence contains unsupported bases")
            if any(label not in {"0", "1"} for label in labels):
                raise ValueError(f"{path} line {line_no}: labels must contain only 0/1")
            rows.append((sequence, labels))
            if limit_rows is not None and len(rows) >= limit_rows:
                break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_split_sequences(split_dir: Path, split_files: Iterable[str]) -> list[str]:
    sequences: list[str] = []
    for split_name in split_files:
        split_path = split_dir / split_name
        with split_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if row:
                    sequences.append(row[0].strip().upper())
    return sequences


def build_kmer_resources(split_dir: Path, lexicon_path: Path, fasttext_model_path: Path) -> tuple[dict[str, int], dict[str, int], dict[str, torch.Tensor]]:
    print(f"Loading lexicon: {lexicon_path}")
    kmer_vocab = readLexicon(str(lexicon_path))
    print(f"Counting k-mers from: {split_dir}")
    total_kmer_counts: dict[str, int] = defaultdict(int)
    split_sequences = read_split_sequences(split_dir, SPLIT_KMER_FILES)
    for kmer, count in count_kmers_in_dataset_with_vocab(split_sequences, kmer_vocab).items():
        total_kmer_counts[kmer] += count

    print(f"Building FastText cache for {len(total_kmer_counts)} observed k-mers...")
    kmer_embedding_cache = {
        kmer: get_direct_vector(kmer, str(fasttext_model_path))
        for kmer in tqdm(kmer_vocab, desc="FastText cache")
        if kmer in total_kmer_counts
    }
    return kmer_vocab, total_kmer_counts, kmer_embedding_cache


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def predict_direct(
    model: torch.nn.Module,
    sequences: list[str],
    kmer_vocab: dict[str, int],
    total_kmer_counts: dict[str, int],
    fasttext_model_path: Path,
    kmer_embedding_cache: dict[str, torch.Tensor],
    batch_size: int,
    desc: str,
) -> list[np.ndarray]:
    probability_chunks: list[np.ndarray] = []
    batches = list(batched(sequences, batch_size))
    for batch_sequences in tqdm(batches, desc=desc, unit="batch"):
        kmer_embeddings = match_and_embed_with_kmers_batch(
            batch_sequences,
            kmer_vocab,
            total_kmer_counts,
            str(fasttext_model_path),
            kmer_embedding_cache,
        )
        with torch.inference_mode():
            outputs, _ = model(batch_sequences, label_ids=None, kmer_embeddings=kmer_embeddings)
            logits = outputs[0].float()
            probabilities = F.softmax(logits, dim=-1)[..., 1].detach().cpu().numpy()
        probability_chunks.extend(probabilities[index, : len(sequence)].astype(np.float32) for index, sequence in enumerate(batch_sequences))
    return probability_chunks


def scan_starts(sequence_length: int, window_size: int, stride: int) -> list[int]:
    if sequence_length <= window_size:
        return [0]
    starts = list(range(0, sequence_length - window_size + 1, stride))
    final_start = sequence_length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_one_by_scan(
    model: torch.nn.Module,
    sequence: str,
    kmer_vocab: dict[str, int],
    total_kmer_counts: dict[str, int],
    fasttext_model_path: Path,
    kmer_embedding_cache: dict[str, torch.Tensor],
    window_size: int,
    stride: int,
    aggregation: str,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    if len(sequence) <= window_size:
        return predict_direct(
            model,
            [sequence],
            kmer_vocab,
            total_kmer_counts,
            fasttext_model_path,
            kmer_embedding_cache,
            batch_size=1,
            desc=desc,
        )[0]

    starts = scan_starts(len(sequence), window_size, stride)
    windows = [sequence[start : start + window_size] for start in starts]
    window_probabilities = predict_direct(
        model,
        windows,
        kmer_vocab,
        total_kmer_counts,
        fasttext_model_path,
        kmer_embedding_cache,
        batch_size=batch_size,
        desc=desc,
    )

    if aggregation == "max":
        scores = np.full(len(sequence), -np.inf, dtype=np.float32)
    else:
        scores = np.zeros(len(sequence), dtype=np.float64)
    counts = np.zeros(len(sequence), dtype=np.int32)

    for start, probs in zip(starts, window_probabilities):
        end = start + window_size
        if aggregation == "max":
            scores[start:end] = np.maximum(scores[start:end], probs)
        else:
            scores[start:end] += probs
            counts[start:end] += 1

    if aggregation == "max":
        scores[~np.isfinite(scores)] = 0.0
        return scores.astype(np.float32)

    covered = counts > 0
    scores[covered] = scores[covered] / counts[covered]
    scores[~covered] = 0.0
    return scores.astype(np.float32)


def choose_mode(sequence_length: int, requested_mode: str, window_size: int) -> str:
    if requested_mode == "auto":
        return "direct" if sequence_length <= window_size else "scan"
    return requested_mode


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


def probability_summary(probabilities: list[np.ndarray]) -> dict[str, float]:
    flat = np.concatenate(probabilities)
    return {"min": float(flat.min()), "max": float(flat.max()), "mean": float(flat.mean())}


def labels_to_array(labels: str) -> np.ndarray:
    return np.fromiter((int(char) for char in labels), dtype=np.int8)


def positive_intervals(labels: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(labels, start=1):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(labels)))
    return intervals


def interval_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    return [(start, end - start + 1) for start, end in positive_intervals(labels)]


def post_process_pred_labels(labels: np.ndarray, merge_gap: int, min_length: int) -> np.ndarray:
    processed = labels.copy()
    intervals = positive_intervals(processed)
    if intervals:
        merged_intervals: list[tuple[int, int]] = []
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            gap = start - current_end - 1
            if gap <= merge_gap:
                current_end = end
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        merged_intervals.append((current_start, current_end))

        processed.fill(0)
        for start, end in merged_intervals:
            if end - start + 1 >= min_length:
                processed[start - 1:end] = 1
    return processed


def prediction_labels(probabilities: np.ndarray, threshold: float, post_process: bool, merge_gap: int, min_length: int) -> np.ndarray:
    labels = (probabilities >= threshold).astype(np.int8)
    if post_process:
        labels = post_process_pred_labels(labels, merge_gap, min_length)
    return labels


def row_plot_score(labels: str, probabilities: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    y_true = labels_to_array(labels)
    y_pred = (probabilities >= threshold).astype(np.int8)
    metrics = binary_metrics(y_true, y_pred)
    return (
        float(metrics["f1"]),
        float(metrics["mcc"]),
        float(metrics["recall"]),
        float(probabilities.max()),
    )

def draw_annotation_track(axis: object, sequence_length: int, intervals: list[tuple[int, int]]) -> None:
    axis.set_ylabel("")
    axis.set_yticks([ANNOTATION_TRACK_Y])
    axis.set_yticklabels(["Annotation"])
    axis.set_ylim(0, 1)
    axis.set_xlim(1, sequence_length)
    axis.plot(
        [1, sequence_length],
        [ANNOTATION_TRACK_Y, ANNOTATION_TRACK_Y],
        color=ANNOTATION_BASE_COLOR,
        linewidth=ANNOTATION_TRACK_WIDTH,
        solid_capstyle="round",
        zorder=1,
    )
    for start, end in intervals:
        axis.plot(
            [start, end],
            [ANNOTATION_TRACK_Y, ANNOTATION_TRACK_Y],
            color=ANNOTATION_HIGHLIGHT_COLOR,
            linewidth=ANNOTATION_TRACK_WIDTH,
            solid_capstyle="round",
            zorder=2,
        )
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.tick_params(axis="y", length=0, labelsize=18, pad=14)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_prediction_signal(
    plot_path: Path,
    source_name: str,
    row_id: int,
    sequence: str,
    labels: str,
    probabilities: np.ndarray,
    threshold: float,
    mode: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(ARIAL_FONT_PATH))
    font_name = font_manager.FontProperties(fname=str(ARIAL_FONT_PATH)).get_name()
    mpl.rcParams.update({
        "font.family": [font_name],
        "font.sans-serif": [font_name],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 1.2,
    })

    x = np.arange(1, len(sequence) + 1)
    annotation_intervals = positive_intervals(labels_to_array(labels))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [0.26, 1.54]},
    )
    draw_annotation_track(axes[0], len(sequence), annotation_intervals)

    axes[1].plot(
        x,
        probabilities,
        color=PREDICTION_LINE_COLOR,
        linewidth=1.9,
        label="Prediction signal",
    )
    axes[1].axhline(
        threshold,
        color=THRESHOLD_LINE_COLOR,
        linestyle="--",
        linewidth=1.2,
        label="Threshold",
    )
    axes[1].set_xlim(1, len(sequence))
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Sequence position", fontsize=22)
    axes[1].set_ylabel("Probability", fontsize=22)
    axes[1].tick_params(axis="both", labelsize=18, width=1.0, length=6, pad=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].spines["left"].set_color(SPINE_COLOR)
    axes[1].spines["bottom"].set_color(SPINE_COLOR)
    axes[1].spines["left"].set_linewidth(1.2)
    axes[1].spines["bottom"].set_linewidth(1.2)
    legend = axes[1].legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        borderpad=0.8,
        labelspacing=0.8,
        handlelength=2.0,
        fontsize=18,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#C8CDD5")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_alpha(0.95)
    axes[1].grid(True, which="major", linestyle=(0, (1.5, 3.0)), color=GRID_COLOR, linewidth=0.8)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=PLOT_DPI)
    fig.savefig(plot_path.with_suffix(".pdf"), dpi=PLOT_DPI)
    plt.close(fig)


def collect_existing_plot_requests(plot_subdir: Path) -> list[tuple[str, int]]:
    requests: list[tuple[str, int]] = []
    for plot_path in sorted(plot_subdir.glob("*.png")):
        matched = re.fullmatch(r"random\d+_row(\d+)\.png", plot_path.name)
        if matched is None:
            raise ValueError(f"Unexpected plot filename: {plot_path.name}")
        requests.append((plot_path.name, int(matched.group(1))))
    if not requests:
        raise ValueError(f"No PNG plots found in {plot_subdir}")
    return requests


def write_requested_prediction_plots(
    plot_subdir: Path,
    rows: list[tuple[str, str]],
    probabilities: list[np.ndarray],
    modes: list[str],
    threshold: float,
) -> list[Path]:
    written_paths: list[Path] = []
    for filename, row_id in collect_existing_plot_requests(plot_subdir):
        sequence, labels = rows[row_id - 1]
        probs = probabilities[row_id - 1]
        mode = modes[row_id - 1]
        plot_path = plot_subdir / filename
        plot_prediction_signal(
            plot_path=plot_path,
            source_name=plot_subdir.name,
            row_id=row_id,
            sequence=sequence,
            labels=labels,
            probabilities=probs,
            threshold=threshold,
            mode=mode,
        )
        written_paths.append(plot_path)
    return written_paths


def write_top_prediction_plots(
    plot_dir: Path,
    source_path: Path,
    rows: list[tuple[str, str]],
    probabilities: list[np.ndarray],
    modes: list[str],
    threshold: float,
    top_k: int,
) -> list[Path]:
    if top_k <= 0:
        return []

    row_items = list(enumerate(zip(rows, probabilities, modes), start=1))
    sample_count = min(top_k, len(row_items))
    sampled_indices = np.random.default_rng().choice(len(row_items), size=sample_count, replace=False)
    sampled_rows = [row_items[index] for index in sampled_indices]

    written_paths: list[Path] = []
    for rank, (row_id, ((sequence, labels), probs, mode)) in enumerate(sampled_rows, start=1):
        plot_path = plot_dir / source_path.stem / f"random{rank:02d}_row{row_id:04d}.png"
        plot_prediction_signal(
            plot_path=plot_path,
            source_name=source_path.name,
            row_id=row_id,
            sequence=sequence,
            labels=labels,
            probabilities=probs,
            threshold=threshold,
            mode=mode,
        )
        written_paths.append(plot_path)
    return written_paths


def write_predictions(
    output_path: Path,
    source_path: Path,
    rows: list[tuple[str, str]],
    probabilities: list[np.ndarray],
    modes: list[str],
    threshold: float,
    save_probs: bool,
    post_process: bool,
    merge_gap: int,
    min_length: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "source_file",
        "length",
        "mode",
        "true_positive_bp",
        "pred_positive_bp",
        "max_probability",
        "mean_probability",
        "sequence",
        "annotation",
        "pred_annotation",
    ]
    if save_probs:
        fieldnames.append("positive_probs")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row_id, ((sequence, labels), probs, mode) in enumerate(zip(rows, probabilities, modes), start=1):
            pred_labels = prediction_labels(probs, threshold, post_process, merge_gap, min_length)
            output_row = {
                "row_id": row_id,
                "source_file": str(source_path),
                "length": len(sequence),
                "mode": mode,
                "true_positive_bp": labels.count("1"),
                "pred_positive_bp": int(pred_labels.sum()),
                "max_probability": f"{float(probs.max()):.6f}",
                "mean_probability": f"{float(probs.mean()):.6f}",
                "sequence": sequence,
                "annotation": labels,
                "pred_annotation": "".join(str(int(value)) for value in pred_labels),
            }
            if save_probs:
                output_row["positive_probs"] = ",".join(f"{float(value):.6f}" for value in probs)
            writer.writerow(output_row)


def write_predicted_promoter_fasta(
    output_path: Path,
    rows: list[tuple[str, str]],
    probabilities: list[np.ndarray],
    threshold: float,
    post_process: bool,
    merge_gap: int,
    min_length: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sequence_id = 1
    with output_path.open("w", encoding="utf-8") as handle:
        for sequence, probs in zip((sequence for sequence, _ in rows), probabilities):
            pred_labels = prediction_labels(probs, threshold, post_process, merge_gap, min_length)
            for start, end in positive_intervals(pred_labels):
                handle.write(f">seq_{sequence_id:03d}\n")
                handle.write(f"{sequence[start - 1:end].upper()}\n")
                sequence_id += 1
    return sequence_id - 1


def predict_file(
    input_path: Path,
    output_dir: Path,
    model: torch.nn.Module,
    kmer_vocab: dict[str, int],
    total_kmer_counts: dict[str, int],
    fasttext_model_path: Path,
    kmer_embedding_cache: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, object]:
    rows = read_labeled_tsv(input_path, args.limit_rows)
    sequences = [sequence for sequence, _ in rows]
    labels = [label for _, label in rows]
    modes = [choose_mode(len(sequence), args.mode, args.window_size) for sequence in sequences]

    probabilities: list[np.ndarray] = []
    if len(set(modes)) == 1 and modes[0] == "direct":
        probabilities = predict_direct(
            model,
            sequences,
            kmer_vocab,
            total_kmer_counts,
            fasttext_model_path,
            kmer_embedding_cache,
            batch_size=args.batch_size,
            desc=f"Predict {input_path.name}",
        )
    else:
        row_items = list(enumerate(zip(sequences, modes), start=1))
        for row_index, (row_sequence, row_mode) in tqdm(row_items, desc=f"Rows {input_path.name}"):
            if row_mode == "direct":
                probs = predict_direct(
                    model,
                    [row_sequence],
                    kmer_vocab,
                    total_kmer_counts,
                    fasttext_model_path,
                    kmer_embedding_cache,
                    batch_size=1,
                    desc=f"row {row_index} direct",
                )[0]
            else:
                probs = predict_one_by_scan(
                    model,
                    row_sequence,
                    kmer_vocab,
                    total_kmer_counts,
                    fasttext_model_path,
                    kmer_embedding_cache,
                    window_size=args.window_size,
                    stride=args.stride,
                    aggregation=args.aggregation,
                    batch_size=args.batch_size,
                    desc=f"row {row_index} scan",
                )
            probabilities.append(probs)

    y_true = np.concatenate([np.fromiter((int(char) for char in label), dtype=np.int8) for label in labels])
    y_pred = np.concatenate([
        prediction_labels(
            probs,
            args.threshold,
            args.post_process_promoters,
            args.merge_gap,
            args.min_promoter_length,
        )
        for probs in probabilities
    ])
    y_prob = np.concatenate(probabilities)
    metrics = {
        "input_file": str(input_path),
        "rows": len(rows),
        "mode_request": args.mode,
        "modes_used": {mode: modes.count(mode) for mode in sorted(set(modes))},
        "window_size": args.window_size,
        "stride": args.stride,
        "aggregation": args.aggregation,
        "threshold": args.threshold,
        "post_process_promoters": args.post_process_promoters,
        "merge_gap": args.merge_gap if args.post_process_promoters else None,
        "min_promoter_length": args.min_promoter_length if args.post_process_promoters else None,
        "base_level": binary_metrics(y_true, y_pred),
        "auprc": average_precision(y_true, y_prob),
        "probability_summary": probability_summary(probabilities),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{input_path.stem}.evosnr_predictions.tsv"
    metrics_path = output_dir / f"{input_path.stem}.evosnr_metrics.json"
    promoter_fasta_path = output_dir / f"{input_path.stem}.predicted_promoters.fasta"
    write_predictions(
        prediction_path,
        input_path,
        rows,
        probabilities,
        modes,
        args.threshold,
        args.save_probs,
        args.post_process_promoters,
        args.merge_gap,
        args.min_promoter_length,
    )
    promoter_count = write_predicted_promoter_fasta(
        promoter_fasta_path,
        rows,
        probabilities,
        args.threshold,
        args.post_process_promoters,
        args.merge_gap,
        args.min_promoter_length,
    )
    plot_dir = args.plot_dir if args.plot_dir is not None else output_dir / "plots"
    if args.redraw_existing_plot_dir is not None and input_path.stem == args.redraw_existing_plot_dir.name:
        plot_paths = write_requested_prediction_plots(
            args.redraw_existing_plot_dir,
            rows,
            probabilities,
            modes,
            args.threshold,
        )
    else:
        plot_paths = write_top_prediction_plots(plot_dir, input_path, rows, probabilities, modes, args.threshold, args.plot_top_k)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Predictions written to: {prediction_path}")
    print(f"Metrics written to: {metrics_path}")
    print(f"Predicted promoter FASTA written to: {promoter_fasta_path} ({promoter_count} sequences)")
    if plot_paths:
        print(f"Top prediction plots written to: {plot_paths[0].parent}")
    print(
        f"{input_path.name}: Acc={metrics['base_level']['accuracy']:.4f} "
        f"Precision={metrics['base_level']['precision']:.4f} Recall={metrics['base_level']['recall']:.4f} "
        f"F1={metrics['base_level']['f1']:.4f} Jaccard={metrics['base_level']['jaccard']:.4f} "
        f"MCC={metrics['base_level']['mcc']:.4f} AUPRC={metrics['auprc']:.4f}"
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.merge_gap < 0:
        raise ValueError("--merge-gap must be >= 0")
    if args.min_promoter_length <= 0:
        raise ValueError("--min-promoter-length must be positive")

    device = torch.device(args.device)
    model_evosnr.device = device

    kmer_vocab, total_kmer_counts, kmer_embedding_cache = build_kmer_resources(
        args.split_dir,
        args.lexicon_path,
        args.fasttext_model_path,
    )

    print(f"Loading EvoSNR model from: {args.model_dir}")
    model = load_model_evosnr(model_evosnr.EvoSNR, str(args.model_dir), device)
    model.to(device)
    model.eval()

    all_metrics = []
    for input_path in args.input_files:
        all_metrics.append(
            predict_file(
                input_path=input_path,
                output_dir=args.output_dir,
                model=model,
                kmer_vocab=kmer_vocab,
                total_kmer_counts=total_kmer_counts,
                fasttext_model_path=args.fasttext_model_path,
                kmer_embedding_cache=kmer_embedding_cache,
                args=args,
            )
        )

    combined_path = args.output_dir / "combined_evosnr_metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Combined metrics written to: {combined_path}")


if __name__ == "__main__":
    main()
