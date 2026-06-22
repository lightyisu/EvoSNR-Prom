#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import models.model_evosnr as model_evosnr
from predict_linear_promoters import (
    DEFAULT_FASTTEXT_PATH,
    DEFAULT_LEXICON_PATH,
    DEFAULT_MODEL_DIR,
    DEFAULT_SPLIT_DIR,
    average_precision,
    binary_metrics,
    build_kmer_resources,
    positive_intervals,
    post_process_pred_labels,
    predict_direct,
    prediction_labels,
    scan_starts,
)
from utils.save_load import load_model_evosnr


DEFAULT_FASTA_PATH = Path("/root/autodl-tmp/evosnr_0605/data/Agro/genome/agro_linear.fasta")
DEFAULT_ANNOTATION_PATH = Path("/root/autodl-tmp/evosnr_0605/data/Agro/genome/agro_annotation_promoters.csv")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/evosnr_0605/Experiment/results/genome_scan")
DEFAULT_ANNOTATION_TYPE = "linear-Chr"
DEFAULT_SCAN_STRAND = "plus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan the plus strand of a genome FASTA with the Agro EvoSNR checkpoint.")
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA_PATH)
    parser.add_argument("--annotation-path", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--annotation-type", default=DEFAULT_ANNOTATION_TYPE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--lexicon-path", type=Path, default=DEFAULT_LEXICON_PATH)
    parser.add_argument("--fasttext-model-path", type=Path, default=DEFAULT_FASTTEXT_PATH)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--center-window-size", type=int, default=None)
    parser.add_argument("--aggregation", choices=("max", "mean"), default="mean")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--post-process-promoters", action="store_true")
    parser.add_argument("--merge-gap", type=int, default=10)
    parser.add_argument("--min-promoter-length", type=int, default=30)
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--stage1-threshold", type=float, default=None)
    parser.add_argument("--stage1-merge-gap", type=int, default=10)
    parser.add_argument("--stage1-min-length", type=int, default=1)
    parser.add_argument("--stage2-min-peak-prob", type=float, default=0.75)
    parser.add_argument("--stage2-min-mean-prob", type=float, default=0.55)
    parser.add_argument("--stage2-min-region-length", type=int, default=30)
    parser.add_argument("--report-every-bp", type=int, default=2000)
    parser.add_argument("--max-bp", type=int, default=None)
    parser.add_argument("--save-per-base-probabilities", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_single_fasta(path: Path) -> tuple[str, str]:
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    raise ValueError(f"Expected single-record FASTA, found another header in {path}")
                header = line[1:].strip()
                continue
            sequence_parts.append(line.upper())

    if header is None:
        raise ValueError(f"No FASTA header found in {path}")
    sequence = "".join(sequence_parts)
    if not sequence:
        raise ValueError(f"No sequence found in {path}")

    invalid_bases = sorted(set(sequence) - {"A", "C", "G", "T", "N"})
    if invalid_bases:
        raise ValueError(f"FASTA contains unsupported bases: {invalid_bases}")
    return header, sequence


def sanitize_stem(header: str) -> str:
    token = header.split()[0]
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in token)


def reverse_complement(sequence: str) -> str:
    complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(complement)[::-1].upper()


def find_uppercase_index(sequence: str) -> int:
    for index, base in enumerate(sequence):
        if base.isupper():
            return index
    raise ValueError(f"Sequence does not contain an uppercase TSS marker: {sequence}")


def resolve_promoter_interval(row: dict[str, str], genome_sequence: str) -> tuple[int, int]:
    promoter_sequence = row["PromoterSeq"].strip()
    tss_position = int(row["TSSPosition"])
    strand = row["Strand"].strip()
    uppercase_index = find_uppercase_index(promoter_sequence)
    promoter_length = len(promoter_sequence)

    if strand == "+":
        start = tss_position - uppercase_index
        end = start + promoter_length - 1
        expected_sequence = promoter_sequence.upper()
    elif strand == "-":
        start = tss_position - (promoter_length - 1 - uppercase_index)
        end = tss_position + uppercase_index
        expected_sequence = reverse_complement(promoter_sequence)
    else:
        raise ValueError(f"Unsupported strand value: {strand}")

    if start < 1 or end > len(genome_sequence):
        raise ValueError(f"Promoter {row['PromoterName']} interval {start}-{end} is outside the FASTA")

    genome_slice = genome_sequence[start - 1 : end]
    if genome_slice != expected_sequence:
        raise ValueError(f"Promoter sequence mismatch for {row['PromoterName']} ({strand}) at {start}-{end}")
    return start, end


def load_annotation_records(annotation_path: Path, genome_sequence: str, annotation_type: str, strand: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    expected_strand = "+" if strand == "plus" else "-"
    with annotation_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Type"] != annotation_type:
                continue
            if row["Strand"].strip() != expected_strand:
                continue
            promoter_sequence = row["PromoterSeq"].strip()
            tss_position = int(row["TSSPosition"])
            uppercase_index = find_uppercase_index(promoter_sequence)
            promoter_length = len(promoter_sequence)
            if expected_strand == "+":
                start = tss_position - uppercase_index
                end = start + promoter_length - 1
            else:
                start = tss_position - (promoter_length - 1 - uppercase_index)
                end = tss_position + uppercase_index
            if start < 1 or end > len(genome_sequence):
                continue
            start, end = resolve_promoter_interval(row, genome_sequence)
            records.append(
                {
                    "promoter_name": row["PromoterName"],
                    "strand": row["Strand"].strip(),
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "tss_position": int(row["TSSPosition"]),
                    "locus_tag": row["Locus_tag"],
                }
            )
    if not records:
        raise ValueError(f"No annotation records matched Type={annotation_type!r} and strand={expected_strand!r}")
    return records


def build_reference_labels(sequence_length: int, records: list[dict[str, object]]) -> np.ndarray:
    labels = np.zeros(sequence_length, dtype=np.int8)
    for record in records:
        start = int(record["start"])
        end = int(record["end"])
        labels[start - 1 : end] = 1
    return labels


def resolve_center_window_size(window_size: int, stride: int, requested_center_size: int | None) -> int:
    if requested_center_size is not None:
        if requested_center_size <= 0 or requested_center_size > window_size:
            raise ValueError("--center-window-size must be in (0, window_size]")
        return requested_center_size
    inferred = window_size - 2 * stride
    if inferred <= 0:
        return window_size
    return inferred


def predict_one_by_scan_center(
    model: torch.nn.Module,
    sequence: str,
    kmer_vocab: dict[str, int],
    total_kmer_counts: dict[str, int],
    fasttext_model_path: Path,
    kmer_embedding_cache: dict[str, torch.Tensor],
    window_size: int,
    stride: int,
    aggregation: str,
    center_window_size: int,
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
    center_margin = max(0, (window_size - center_window_size) // 2)

    for index, (start, probs) in enumerate(zip(starts, window_probabilities)):
        emit_start_offset = 0 if index == 0 else center_margin
        emit_end_offset = window_size if index == len(starts) - 1 else window_size - center_margin
        emit_global_start = start + emit_start_offset
        emit_global_end = min(start + emit_end_offset, len(sequence))
        emit_local_start = emit_start_offset
        emit_local_end = emit_local_start + (emit_global_end - emit_global_start)
        emit_probs = probs[emit_local_start:emit_local_end]

        if aggregation == "max":
            scores[emit_global_start:emit_global_end] = np.maximum(scores[emit_global_start:emit_global_end], emit_probs)
        else:
            scores[emit_global_start:emit_global_end] += emit_probs
            counts[emit_global_start:emit_global_end] += 1

    if aggregation == "max":
        scores[~np.isfinite(scores)] = 0.0
        return scores.astype(np.float32)

    covered = counts > 0
    scores[covered] = scores[covered] / counts[covered]
    scores[~covered] = 0.0
    return scores.astype(np.float32)


def apply_two_stage_filter(probabilities: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    stage1_threshold = args.stage1_threshold if args.stage1_threshold is not None else args.threshold
    stage1_labels = (probabilities >= stage1_threshold).astype(np.int8)
    stage1_labels = post_process_pred_labels(stage1_labels, args.stage1_merge_gap, args.stage1_min_length)
    candidate_intervals = positive_intervals(stage1_labels)

    final_labels = np.zeros_like(stage1_labels)
    candidate_stats: list[dict[str, float | int]] = []
    for candidate_id, (start, end) in enumerate(candidate_intervals, start=1):
        region_probs = probabilities[start - 1 : end]
        length = end - start + 1
        peak_prob = float(region_probs.max())
        mean_prob = float(region_probs.mean())
        keep = (
            length >= args.stage2_min_region_length
            and peak_prob >= args.stage2_min_peak_prob
            and mean_prob >= args.stage2_min_mean_prob
        )
        candidate_stats.append(
            {
                "candidate_id": candidate_id,
                "start_1based": start,
                "end_1based": end,
                "length": length,
                "peak_prob": peak_prob,
                "mean_prob": mean_prob,
                "kept": int(keep),
            }
        )
        if keep:
            final_labels[start - 1 : end] = 1

    return final_labels, candidate_stats


def report_chunk_metrics(true_labels: np.ndarray, pred_labels: np.ndarray, chunk_size: int) -> list[dict[str, object]]:
    if chunk_size <= 0:
        return []

    chunk_metrics: list[dict[str, object]] = []
    for start in range(0, len(true_labels), chunk_size):
        end = min(start + chunk_size, len(true_labels))
        metrics = binary_metrics(true_labels[start:end], pred_labels[start:end])
        item = {
            "start_1based": start + 1,
            "end_1based": end,
            "length": end - start,
            "true_positive_bp": int(true_labels[start:end].sum()),
            "pred_positive_bp": int(pred_labels[start:end].sum()),
            "metrics": metrics,
        }
        chunk_metrics.append(item)
        print(
            f"Chunk {start + 1}-{end}: "
            f"Acc={metrics['accuracy']:.6f} "
            f"Precision={metrics['precision']:.6f} "
            f"Recall={metrics['recall']:.6f} "
            f"F1={metrics['f1']:.6f} "
            f"Jaccard={metrics['jaccard']:.6f} "
            f"MCC={metrics['mcc']:.6f} "
            f"true_bp={item['true_positive_bp']} "
            f"pred_bp={item['pred_positive_bp']}"
        )
    return chunk_metrics


def write_per_base_probabilities(
    path: Path,
    sequence_id: str,
    probabilities: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "sequence_id",
            "position_1based",
            "positive_probability",
            "pred_label",
            "true_label",
        ])
        for index, values in enumerate(zip(probabilities, pred_labels, true_labels), start=1):
            writer.writerow([sequence_id, index, f"{float(values[0]):.6f}", int(values[1]), int(values[2])])


def write_region_table(
    path: Path,
    sequence_id: str,
    sequence: str,
    probabilities: np.ndarray,
    intervals: list[tuple[int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["sequence_id", "strand", "region_id", "start_1based", "end_1based", "length", "max_probability", "mean_probability", "sequence"])
        for region_id, (start, end) in enumerate(intervals, start=1):
            region_probs = probabilities[start - 1 : end]
            writer.writerow(
                [
                    sequence_id,
                    "+",
                    region_id,
                    start,
                    end,
                    end - start + 1,
                    f"{float(region_probs.max()):.6f}",
                    f"{float(region_probs.mean()):.6f}",
                    sequence[start - 1 : end],
                ]
            )


def write_region_fasta(path: Path, sequence_id: str, sequence: str, intervals: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for region_id, (start, end) in enumerate(intervals, start=1):
            handle.write(f">{sequence_id}|strand_+|region_{region_id:05d}|{start}-{end}\n")
            handle.write(f"{sequence[start - 1 : end]}\n")


def write_annotation_table(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["promoter_name", "locus_tag", "strand", "tss_position", "start_1based", "end_1based", "length"])
        for record in records:
            writer.writerow([
                record["promoter_name"],
                record["locus_tag"],
                record["strand"],
                record["tss_position"],
                record["start"],
                record["end"],
                record["length"],
            ])


def write_candidate_table(path: Path, candidate_stats: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["candidate_id", "start_1based", "end_1based", "length", "peak_prob", "mean_prob", "kept"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in candidate_stats:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "start_1based": row["start_1based"],
                "end_1based": row["end_1based"],
                "length": row["length"],
                "peak_prob": f"{float(row['peak_prob']):.6f}",
                "mean_prob": f"{float(row['mean_prob']):.6f}",
                "kept": row["kept"],
            })


def build_summary(
    fasta_path: Path,
    annotation_path: Path,
    annotation_type: str,
    sequence_id: str,
    sequence_length: int,
    records: list[dict[str, object]],
    probabilities: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    intervals: list[tuple[int, int]],
    chunk_metrics: list[dict[str, object]],
    center_window_size: int,
    args: argparse.Namespace,
    candidate_stats: list[dict[str, float | int]],
) -> dict[str, object]:
    summary = {
        "input_fasta": str(fasta_path),
        "annotation_path": str(annotation_path),
        "annotation_type": annotation_type,
        "scan_strand": DEFAULT_SCAN_STRAND,
        "sequence_id": sequence_id,
        "sequence_length": sequence_length,
        "model_dir": str(args.model_dir),
        "window_size": args.window_size,
        "stride": args.stride,
        "center_window_size": center_window_size,
        "aggregation": args.aggregation,
        "threshold": args.threshold,
        "post_process_promoters": args.post_process_promoters,
        "merge_gap": args.merge_gap if args.post_process_promoters else None,
        "min_promoter_length": args.min_promoter_length if args.post_process_promoters else None,
        "two_stage": args.two_stage,
        "report_every_bp": args.report_every_bp,
        "max_bp": args.max_bp,
        "true_region_count": len(records),
        "predicted_region_count": len(intervals),
        "true_positive_bp": int(true_labels.sum()),
        "predicted_positive_bp": int(pred_labels.sum()),
        "base_level": binary_metrics(true_labels, pred_labels),
        "auprc": average_precision(true_labels, probabilities),
        "probability_summary": {
            "min": float(probabilities.min()),
            "max": float(probabilities.max()),
            "mean": float(probabilities.mean()),
        },
        "chunk_metrics": chunk_metrics,
    }
    if args.two_stage:
        summary["two_stage_config"] = {
            "stage1_threshold": args.stage1_threshold if args.stage1_threshold is not None else args.threshold,
            "stage1_merge_gap": args.stage1_merge_gap,
            "stage1_min_length": args.stage1_min_length,
            "stage2_min_peak_prob": args.stage2_min_peak_prob,
            "stage2_min_mean_prob": args.stage2_min_mean_prob,
            "stage2_min_region_length": args.stage2_min_region_length,
            "candidate_count": len(candidate_stats),
            "kept_candidate_count": int(sum(int(item["kept"]) for item in candidate_stats)),
        }
    return summary


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
    if args.stage1_merge_gap < 0:
        raise ValueError("--stage1-merge-gap must be >= 0")
    if args.stage1_min_length <= 0:
        raise ValueError("--stage1-min-length must be positive")
    if args.stage2_min_region_length <= 0:
        raise ValueError("--stage2-min-region-length must be positive")

    header, sequence = load_single_fasta(args.fasta)
    if args.max_bp is not None:
        if args.max_bp <= 0:
            raise ValueError("--max-bp must be positive")
        sequence = sequence[: args.max_bp]
    sequence_id = sanitize_stem(header)
    records = load_annotation_records(args.annotation_path, sequence, args.annotation_type, strand=DEFAULT_SCAN_STRAND)
    true_labels = build_reference_labels(len(sequence), records)
    center_window_size = resolve_center_window_size(args.window_size, args.stride, args.center_window_size)

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

    print(
        f"Scanning plus strand of {args.fasta} ({len(sequence)} bp) "
        f"with aggregation={args.aggregation} center_window_size={center_window_size}"
    )
    probabilities = predict_one_by_scan_center(
        model=model,
        sequence=sequence,
        kmer_vocab=kmer_vocab,
        total_kmer_counts=total_kmer_counts,
        fasttext_model_path=args.fasttext_model_path,
        kmer_embedding_cache=kmer_embedding_cache,
        window_size=args.window_size,
        stride=args.stride,
        aggregation=args.aggregation,
        center_window_size=center_window_size,
        batch_size=args.batch_size,
        desc=f"plus scan {args.fasta.name}",
    )

    candidate_stats: list[dict[str, float | int]] = []
    if args.two_stage:
        pred_labels, candidate_stats = apply_two_stage_filter(probabilities, args)
        print(
            f"Two-stage filter kept {sum(int(item['kept']) for item in candidate_stats)} "
            f"of {len(candidate_stats)} candidate regions"
        )
    else:
        pred_labels = prediction_labels(
            probabilities,
            args.threshold,
            args.post_process_promoters,
            args.merge_gap,
            args.min_promoter_length,
        )

    intervals = positive_intervals(pred_labels)
    chunk_metrics = report_chunk_metrics(true_labels, pred_labels, args.report_every_bp)

    output_prefix = args.output_dir / sanitize_stem(args.fasta.stem)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_base_path = output_prefix.with_suffix(".per_base.tsv")
    region_path = output_prefix.with_suffix(".predicted_regions.tsv")
    fasta_path = output_prefix.with_suffix(".predicted_promoters.fasta")
    annotation_table_path = output_prefix.with_suffix(".reference_regions.tsv")
    candidate_table_path = output_prefix.with_suffix(".candidate_regions.tsv")
    summary_path = output_prefix.with_suffix(".summary.json")

    if args.save_per_base_probabilities:
        write_per_base_probabilities(per_base_path, sequence_id, probabilities, pred_labels, true_labels)
        print(f"Per-base probabilities written to: {per_base_path}")

    write_region_table(region_path, sequence_id, sequence, probabilities, intervals)
    write_region_fasta(fasta_path, sequence_id, sequence, intervals)
    write_annotation_table(annotation_table_path, records)
    if args.two_stage:
        write_candidate_table(candidate_table_path, candidate_stats)
        print(f"Candidate regions written to: {candidate_table_path}")

    summary = build_summary(
        args.fasta,
        args.annotation_path,
        args.annotation_type,
        sequence_id,
        len(sequence),
        records,
        probabilities,
        pred_labels,
        true_labels,
        intervals,
        chunk_metrics,
        center_window_size,
        args,
        candidate_stats,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Predicted regions written to: {region_path}")
    print(f"Predicted promoter FASTA written to: {fasta_path}")
    print(f"Reference regions written to: {annotation_table_path}")
    print(f"Summary written to: {summary_path}")
    print(
        f"Acc={summary['base_level']['accuracy']:.6f} "
        f"Precision={summary['base_level']['precision']:.6f} "
        f"Recall={summary['base_level']['recall']:.6f} "
        f"F1={summary['base_level']['f1']:.6f} "
        f"Jaccard={summary['base_level']['jaccard']:.6f} "
        f"MCC={summary['base_level']['mcc']:.6f} "
        f"AUPRC={summary['auprc']:.6f}"
    )


if __name__ == "__main__":
    main()
