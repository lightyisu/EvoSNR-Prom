#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_FASTA_PATH = Path("/root/autodl-tmp/evosnr_0605/data/Agro/genome/agro_linear.fasta")
DEFAULT_ANNOTATION_PATH = Path("/root/autodl-tmp/evosnr_0605/data/Agro/genome/agro_annotation_promoters.csv")
DEFAULT_OUTPUT_500 = Path("/root/autodl-tmp/evosnr_0605/data/Agro/genome/agro_linear_promoter_500bp.csv")
DEFAULT_OUTPUT_1000 = Path("/root/autodl-tmp/evosnr_0605/data/Agro/genome/agro_linear_promoter_1000bp.csv")
TYPE_FILTER = "linear-Chr"
WINDOW_SIZES = (500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate linear-chromosome promoter windows with per-base labels."
    )
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA_PATH)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION_PATH)
    parser.add_argument("--output-500", type=Path, default=DEFAULT_OUTPUT_500)
    parser.add_argument("--output-1000", type=Path, default=DEFAULT_OUTPUT_1000)
    parser.add_argument("--type-filter", default=TYPE_FILTER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--genomic-orientation",
        action="store_true",
        help="Keep every window in FASTA genomic orientation instead of reverse-complementing minus-strand windows.",
    )
    return parser.parse_args()


def reverse_complement(sequence: str) -> str:
    complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(complement)[::-1].upper()


def load_single_fasta(path: Path) -> str:
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(">"):
                continue
            sequence_parts.append(line.upper())

    sequence = "".join(sequence_parts)
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")
    invalid_bases = sorted(set(sequence) - {"A", "C", "G", "T", "N"})
    if invalid_bases:
        raise ValueError(f"FASTA contains unsupported bases: {invalid_bases}")
    return sequence


def find_uppercase_index(sequence: str) -> int:
    for index, base in enumerate(sequence):
        if base.isupper():
            return index
    raise ValueError(f"Sequence does not contain an uppercase TSS marker: {sequence}")


def resolve_promoter_interval(row: dict[str, str], genome_sequence: str) -> tuple[int, int, str]:
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
        raise ValueError(f"Promoter {row['PromoterName']} interval {start}-{end} is outside the linear FASTA")

    genome_slice = genome_sequence[start - 1 : end]
    if genome_slice != expected_sequence:
        raise ValueError(
            f"Promoter sequence mismatch for {row['PromoterName']} ({strand}) at {start}-{end}"
        )
    return start, end, promoter_sequence.upper()


def load_promoter_records(
    annotation_path: Path,
    genome_sequence: str,
    type_filter: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with annotation_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Type"] != type_filter:
                continue
            start, end, promoter_sequence = resolve_promoter_interval(row, genome_sequence)
            records.append(
                {
                    "promoter_name": row["PromoterName"],
                    "strand": row["Strand"].strip(),
                    "start": start,
                    "end": end,
                    "length": len(promoter_sequence),
                }
            )

    if not records:
        raise ValueError(f"No annotation records matched Type={type_filter!r}")
    return records


def build_window_row(
    record: dict[str, object],
    genome_sequence: str,
    window_size: int,
    rng: random.Random,
    orient_by_strand: bool,
) -> tuple[str, str]:
    promoter_start = int(record["start"])
    promoter_end = int(record["end"])
    promoter_length = int(record["length"])
    genome_length = len(genome_sequence)

    if promoter_length > window_size:
        raise ValueError(
            f"Promoter {record['promoter_name']} length {promoter_length} exceeds window size {window_size}"
        )

    min_window_start = max(1, promoter_end - window_size + 1)
    max_window_start = min(promoter_start, genome_length - window_size + 1)
    if min_window_start > max_window_start:
        raise ValueError(
            f"No valid {window_size}bp window can fully contain promoter {record['promoter_name']} "
            f"at {promoter_start}-{promoter_end}"
        )

    window_start = rng.randint(min_window_start, max_window_start)
    window_end = window_start + window_size - 1
    sequence = genome_sequence[window_start - 1 : window_end]
    labels = ["0"] * window_size

    label_start = promoter_start - window_start
    label_end = promoter_end - window_start
    for index in range(label_start, label_end + 1):
        labels[index] = "1"

    if orient_by_strand and record["strand"] == "-":
        sequence = reverse_complement(sequence)
        labels = list(reversed(labels))

    if len(sequence) != window_size:
        raise ValueError(f"Expected {window_size}bp sequence, got {len(sequence)}")
    if labels.count("1") != promoter_length:
        raise ValueError(f"Label length mismatch for promoter {record['promoter_name']}")

    return sequence, "".join(labels)


def write_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    orient_by_strand = not args.genomic_orientation

    genome_sequence = load_single_fasta(args.fasta)
    records = load_promoter_records(args.annotation, genome_sequence, args.type_filter)

    outputs = {500: args.output_500, 1000: args.output_1000}
    for window_size in WINDOW_SIZES:
        rows = [
            build_window_row(
                record=record,
                genome_sequence=genome_sequence,
                window_size=window_size,
                rng=rng,
                orient_by_strand=orient_by_strand,
            )
            for record in records
        ]
        write_rows(outputs[window_size], rows)
        print(f"{window_size}bp: records={len(rows)} written_to={outputs[window_size]}")


if __name__ == "__main__":
    main()
