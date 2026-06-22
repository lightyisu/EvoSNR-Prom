from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from Bio import SeqIO, motifs
from Bio.Seq import Seq
from matplotlib import font_manager

matplotlib.use("Agg")
import logomaker
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PROMOTER_SET_PATH = BASE_DIR / "PromoterSet.tsv"
PREDICTED_FASTA_PATH = Path(
    "/root/autodl-tmp/evosnr_0605/Experiment/results/linear_promoter_predictions/agro_linear_promoter_500bp.predicted_promoters.fasta"
)
OUTPUT_TABLE_PATH = BASE_DIR / "predicted_promoter_minus35_minus10_scan.tsv"
OUTPUT_FIGURE_PATH = BASE_DIR / "predicted_promoter_minus35_minus10_logo.png"
FONT_PATH = Path("/root/autodl-tmp/ARIAL.TTF")

SIGMA_FACTOR = "sigma70"
CONFIDENCE_LEVELS = {"C", "S"}
BOX_LENGTH = 6
SPACER_MIN = 15
SPACER_MAX = 19
SCORE_THRESHOLD_PERCENTILE = 10
BACKGROUND = {"A": 0.246, "T": 0.246, "G": 0.254, "C": 0.254}
FIGURE_FACE_COLOR = "#F3EEE7"
AXIS_FACE_COLOR = "#FCFAF6"
TITLE_COLOR = "#264653"
SPINE_COLOR = "#C8B8A6"
TICK_COLOR = "#5F5B57"


@dataclass(frozen=True)
class PromoterHit:
    record_id: str
    sequence_length: int
    minus35_start_1based: int
    minus35_seq: str
    minus35_score: float
    spacer_length: int
    minus10_start_1based: int
    minus10_seq: str
    minus10_score: float
    pair_score: float
    passes_minus35: bool
    passes_minus10: bool
    passes_sigma70_pattern: bool


def motif_counts_df(motif: motifs.Motif) -> pd.DataFrame:
    return pd.DataFrame({base: list(motif.counts[base]) for base in "ACGT"})


def clean_box_series(box_series: pd.Series) -> list[str]:
    boxes = box_series.dropna().str.strip().str.upper()
    boxes = boxes[boxes.str.len() == BOX_LENGTH]
    boxes = boxes[boxes.str.fullmatch("[ACGT]+")]
    return boxes.tolist()


def build_motif_and_pssm(boxes: list[str]) -> tuple[motifs.Motif, motifs.matrix.PositionSpecificScoringMatrix]:
    motif = motifs.create([Seq(box) for box in boxes])
    pssm = motif.counts.normalize(pseudocounts=0.5).log_odds(BACKGROUND)
    return motif, pssm


def load_sigma70_boxes() -> tuple[list[str], list[str]]:
    df = pd.read_csv(PROMOTER_SET_PATH, sep="\t", comment="#")
    df.columns = df.columns.str.strip()
    sigma70_df = df[
        (df["5)sigmaFactor"] == SIGMA_FACTOR)
        & (df["15)confidenceLevel"].isin(CONFIDENCE_LEVELS))
    ]
    minus10_boxes = clean_box_series(sigma70_df["10)boxMinus10seq"])
    minus35_boxes = clean_box_series(sigma70_df["12)boxMinus35seq"])
    return minus10_boxes, minus35_boxes


def percentile_score_threshold(pssm: motifs.matrix.PositionSpecificScoringMatrix, boxes: list[str]) -> float:
    known_scores = np.array([float(pssm.calculate(Seq(box))) for box in boxes])
    return float(np.percentile(known_scores, SCORE_THRESHOLD_PERCENTILE))


def scan_best_sigma70_pair(
    record_id: str,
    sequence: Seq,
    pssm10: motifs.matrix.PositionSpecificScoringMatrix,
    pssm35: motifs.matrix.PositionSpecificScoringMatrix,
    minus10_threshold: float,
    minus35_threshold: float,
) -> PromoterHit:
    sequence = sequence.upper()
    minus10_scores = np.asarray(pssm10.calculate(sequence), dtype=float)
    minus35_scores = np.asarray(pssm35.calculate(sequence), dtype=float)

    best_minus35_start = 0
    best_spacer = SPACER_MIN
    best_pair_score = -np.inf

    for minus35_start in range(len(minus35_scores)):
        for spacer in range(SPACER_MIN, SPACER_MAX + 1):
            minus10_start = minus35_start + BOX_LENGTH + spacer
            if minus10_start >= len(minus10_scores):
                continue
            pair_score = minus35_scores[minus35_start] + minus10_scores[minus10_start]
            if pair_score > best_pair_score:
                best_minus35_start = minus35_start
                best_spacer = spacer
                best_pair_score = pair_score

    best_minus10_start = best_minus35_start + BOX_LENGTH + best_spacer
    minus35_seq = str(sequence[best_minus35_start : best_minus35_start + BOX_LENGTH])
    minus10_seq = str(sequence[best_minus10_start : best_minus10_start + BOX_LENGTH])
    minus35_score = float(minus35_scores[best_minus35_start])
    minus10_score = float(minus10_scores[best_minus10_start])
    passes_minus35 = minus35_score >= minus35_threshold
    passes_minus10 = minus10_score >= minus10_threshold

    return PromoterHit(
        record_id=record_id,
        sequence_length=len(sequence),
        minus35_start_1based=best_minus35_start + 1,
        minus35_seq=minus35_seq,
        minus35_score=minus35_score,
        spacer_length=best_spacer,
        minus10_start_1based=best_minus10_start + 1,
        minus10_seq=minus10_seq,
        minus10_score=minus10_score,
        pair_score=float(best_pair_score),
        passes_minus35=passes_minus35,
        passes_minus10=passes_minus10,
        passes_sigma70_pattern=passes_minus35 and passes_minus10,
    )


def write_hits_table(hits: list[PromoterHit]) -> pd.DataFrame:
    hits_df = pd.DataFrame([hit.__dict__ for hit in hits])
    hits_df.to_csv(OUTPUT_TABLE_PATH, sep="\t", index=False)
    return hits_df


def build_logo_matrix(boxes: list[str]) -> pd.DataFrame:
    motif = motifs.create([Seq(box) for box in boxes])
    counts = motif_counts_df(motif)
    return logomaker.transform_matrix(counts, from_type="counts", to_type="information")


def configure_plot_font() -> None:
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()


def style_logo_axis(ax) -> None:
    ax.set_facecolor(AXIS_FACE_COLOR)
    ax.title.set_color(TITLE_COLOR)
    ax.title.set_fontweight("bold")
    ax.tick_params(axis="x", colors=TICK_COLOR, labelsize=9)
    ax.tick_params(axis="y", colors=TICK_COLOR, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.yaxis.label.set_color(TICK_COLOR)
    ax.xaxis.label.set_color(TICK_COLOR)


def plot_logo_comparison(
    minus35_reference_boxes: list[str],
    minus10_reference_boxes: list[str],
    passed_minus35_boxes: list[str],
    passed_minus10_boxes: list[str],
) -> None:
    configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 5), facecolor=FIGURE_FACE_COLOR)

    logomaker.Logo(build_logo_matrix(minus35_reference_boxes), ax=axes[0, 0])
    axes[0, 0].set_title("RegulonDB sigma70 -35 box")
    style_logo_axis(axes[0, 0])

    logomaker.Logo(build_logo_matrix(minus10_reference_boxes), ax=axes[0, 1])
    axes[0, 1].set_title("RegulonDB sigma70 -10 box")
    style_logo_axis(axes[0, 1])

    logomaker.Logo(build_logo_matrix(passed_minus35_boxes), ax=axes[1, 0])
    axes[1, 0].set_title("Predicted promoters passed -35")
    style_logo_axis(axes[1, 0])

    logomaker.Logo(build_logo_matrix(passed_minus10_boxes), ax=axes[1, 1])
    axes[1, 1].set_title("Predicted promoters passed -10")
    style_logo_axis(axes[1, 1])

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_PATH, dpi=150, facecolor=FIGURE_FACE_COLOR)
    plt.close(fig)


def main() -> None:
    minus10_boxes, minus35_boxes = load_sigma70_boxes()
    motif10, pssm10 = build_motif_and_pssm(minus10_boxes)
    motif35, pssm35 = build_motif_and_pssm(minus35_boxes)
    minus10_threshold = percentile_score_threshold(pssm10, minus10_boxes)
    minus35_threshold = percentile_score_threshold(pssm35, minus35_boxes)

    hits = [
        scan_best_sigma70_pair(
            record.id,
            record.seq,
            pssm10,
            pssm35,
            minus10_threshold,
            minus35_threshold,
        )
        for record in SeqIO.parse(PREDICTED_FASTA_PATH, "fasta")
    ]
    hits_df = write_hits_table(hits)

    passed_hits = hits_df[hits_df["passes_sigma70_pattern"]]
    plot_logo_comparison(
        minus35_reference_boxes=minus35_boxes,
        minus10_reference_boxes=minus10_boxes,
        passed_minus35_boxes=passed_hits["minus35_seq"].tolist(),
        passed_minus10_boxes=passed_hits["minus10_seq"].tolist(),
    )

    print(f"RegulonDB sigma70 -35 boxes: {len(minus35_boxes)}")
    print(f"RegulonDB sigma70 -10 boxes: {len(minus10_boxes)}")
    print(f"-35 score threshold ({SCORE_THRESHOLD_PERCENTILE}th percentile): {minus35_threshold:.3f}")
    print(f"-10 score threshold ({SCORE_THRESHOLD_PERCENTILE}th percentile): {minus10_threshold:.3f}")
    print(f"Predicted promoters scanned: {len(hits_df)}")
    print(f"Passed sigma70 -35/-10 pattern: {int(hits_df['passes_sigma70_pattern'].sum())}")
    print(f"Saved scan table: {OUTPUT_TABLE_PATH}")
    print(f"Saved logo figure: {OUTPUT_FIGURE_PATH}")


if __name__ == "__main__":
    main()
