from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager
import numpy as np
import pandas as pd
from Bio import motifs
from Bio.Seq import Seq

matplotlib.use("Agg")
import logomaker
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from matplotlib.transforms import ScaledTranslation


BASE_DIR = Path(__file__).resolve().parent
SCAN_TABLE_PATH = BASE_DIR / "predicted_promoter_minus35_minus10_scan.tsv"
PROMOTER_SET_PATH = BASE_DIR / "PromoterSet.tsv"
SUMMARY_PREFIX = BASE_DIR / "predicted_promoter_minus35_minus10_paper_summary"
SUMMARY_PANEL_PREFIX = BASE_DIR / "predicted_promoter_minus35_minus10_paper_panel"
LOGO_PREFIX = BASE_DIR / "predicted_promoter_minus35_minus10_logo_comparison"
FONT_PATH = Path("/root/autodl-tmp/ARIAL.TTF")

SIGMA_FACTOR = "sigma70"
CONFIDENCE_LEVELS = {"C", "S"}
BOX_LENGTH = 6
SPACER_MIN = 15
SPACER_MAX = 19
SCORE_THRESHOLD_PERCENTILE = 10
FIGURE_DPI = 600
BACKGROUND = {"A": 0.246, "T": 0.246, "G": 0.254, "C": 0.254}
BLUE = "#3B63D1"
LIGHT_BLUE = "#7A9CE6"
ORANGE = "#F28C3A"
LIGHT_ORANGE = "#FDBA74"
RED = "#FF5A59"
GREEN = ORANGE
PURPLE = "#CC79A7"
GRAY = "#000000"
LIGHT_GRAY = "#000000"


font_manager.fontManager.addfont(FONT_PATH)

plt.rcParams.update(
    {
        "font.family": font_manager.FontProperties(fname=FONT_PATH).get_name(),
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def clean_box_series(box_series: pd.Series) -> list[str]:
    boxes = box_series.dropna().str.strip().str.upper()
    boxes = boxes[boxes.str.len() == BOX_LENGTH]
    boxes = boxes[boxes.str.fullmatch("[ACGT]+")]
    return boxes.tolist()


def load_reference_boxes() -> tuple[list[str], list[str]]:
    df = pd.read_csv(PROMOTER_SET_PATH, sep="\t", comment="#")
    df.columns = df.columns.str.strip()
    sigma70_df = df[
        (df["5)sigmaFactor"] == SIGMA_FACTOR)
        & (df["15)confidenceLevel"].isin(CONFIDENCE_LEVELS))
    ]
    minus10_boxes = clean_box_series(sigma70_df["10)boxMinus10seq"])
    minus35_boxes = clean_box_series(sigma70_df["12)boxMinus35seq"])
    return minus35_boxes, minus10_boxes


def motif_counts_df(boxes: list[str]) -> pd.DataFrame:
    motif = motifs.create([Seq(box) for box in boxes])
    return pd.DataFrame({base: list(motif.counts[base]) for base in "ACGT"})


def logo_matrix(boxes: list[str]) -> pd.DataFrame:
    return logomaker.transform_matrix(motif_counts_df(boxes), from_type="counts", to_type="information")


def score_threshold(boxes: list[str]) -> float:
    motif = motifs.create([Seq(box) for box in boxes])
    pssm = motif.counts.normalize(pseudocounts=0.5).log_odds(BACKGROUND)
    scores = np.array([float(pssm.calculate(Seq(box))) for box in boxes])
    return float(np.percentile(scores, SCORE_THRESHOLD_PERCENTILE))


def consensus(boxes: list[str]) -> str:
    counts = motif_counts_df(boxes)
    return "".join(counts.idxmax(axis=1).tolist())


def save_all(fig: plt.Figure, prefix: Path) -> None:
    fig.savefig(f"{prefix}.pdf", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(f"{prefix}.svg", dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(f"{prefix}.png", dpi=FIGURE_DPI, bbox_inches="tight")


def save_panel(fig: plt.Figure, ax: plt.Axes, prefix: Path, extra_artists: list | None = None, xpad: int = 18, ypad: int = 18) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_tightbbox(renderer)
    for artist in extra_artists or []:
        bbox = Bbox.union([bbox, artist.get_tightbbox(renderer)])
    bbox = Bbox.from_extents(
        bbox.x0 - xpad,
        bbox.y0 - ypad,
        bbox.x1 + xpad,
        bbox.y1 + ypad,
    )
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"{prefix}.pdf", dpi=FIGURE_DPI, bbox_inches=bbox_inches)
    fig.savefig(f"{prefix}.svg", dpi=FIGURE_DPI, bbox_inches=bbox_inches)
    fig.savefig(f"{prefix}.png", dpi=FIGURE_DPI, bbox_inches=bbox_inches)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    x = -0.055 if label == "A" else -0.12
    offset = ScaledTranslation(-10 / ax.figure.dpi, 10 / ax.figure.dpi, ax.figure.dpi_scale_trans)
    ax.text(
        x,
        1.08,
        label,
        transform=ax.transAxes + offset,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        color=GRAY,
    )


def format_logo_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=14, color=GRAY)
    ax.set_ylabel("Information (bits)", color=GRAY)
    ax.set_xlabel("Position", color=GRAY)
    ax.set_xticks(range(BOX_LENGTH))
    ax.set_xticklabels(range(1, BOX_LENGTH + 1), color=GRAY)
    ax.tick_params(axis="y", colors=GRAY)


def style_summary_axis(ax: plt.Axes) -> None:
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(colors=GRAY)
    ax.xaxis.label.set_color(GRAY)
    ax.yaxis.label.set_color(GRAY)
    ax.title.set_color(GRAY)


def plot_schematic(ax: plt.Axes, hits_df: pd.DataFrame) -> None:
    pass_count = int(hits_df["passes_sigma70_pattern"].sum())
    total_count = len(hits_df)
    pass_rate = pass_count / total_count * 100
    passed_hits = hits_df[hits_df["passes_sigma70_pattern"]]
    minus35_consensus = consensus(passed_hits["minus35_seq"].tolist())
    minus10_consensus = consensus(passed_hits["minus10_seq"].tolist())
    spacer_mode = int(passed_hits["spacer_length"].mode().iloc[0])

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.add_patch(Rectangle((18, 3.7), 18, 2.6, facecolor=ORANGE, edgecolor=GRAY, linewidth=0.8))
    ax.add_patch(Rectangle((64, 3.7), 18, 2.6, facecolor=LIGHT_BLUE, edgecolor=GRAY, linewidth=0.8))
    ax.text(27, 5.0, "-35", color="white", fontweight="bold", ha="center", va="center")
    ax.text(73, 5.0, "-10", color="white", fontweight="bold", ha="center", va="center")
    ax.text(27, 2.6, minus35_consensus, color=GRAY, fontweight="bold", ha="center", va="center")
    ax.text(73, 2.6, minus10_consensus, color=GRAY, fontweight="bold", ha="center", va="center")
    ax.annotate("", xy=(64, 7.4), xytext=(36, 7.4), arrowprops={"arrowstyle": "<->", "lw": 1.1, "color": GRAY})
    ax.text(50, 8.1, f"spacer {SPACER_MIN}-{SPACER_MAX} bp; mode = {spacer_mode} bp", ha="center", va="center", color=GRAY)
    ax.text(
        50,
        0.9,
        f"{pass_count}/{total_count} predicted promoters passed the sigma70-like -35/-10 pattern ({pass_rate:.1f}%)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=GRAY,
    )


def plot_summary_figure(hits_df: pd.DataFrame) -> None:
    minus35_threshold = score_threshold(reference_minus35_boxes)
    minus10_threshold = score_threshold(reference_minus10_boxes)
    passed_hits = hits_df[hits_df["passes_sigma70_pattern"]]

    fig = plt.figure(figsize=(7.2, 7.4))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.75, 1.0, 1.1], hspace=0.62, wspace=0.42)

    ax_schematic = fig.add_subplot(grid[0, :])
    plot_schematic(ax_schematic, hits_df)
    add_panel_label(ax_schematic, "A")

    ax_logo35 = fig.add_subplot(grid[1, 0])
    logomaker.Logo(logo_matrix(passed_hits["minus35_seq"].tolist()), ax=ax_logo35)
    format_logo_axis(ax_logo35, "Passed predicted -35 boxes")
    add_panel_label(ax_logo35, "B")

    ax_logo10 = fig.add_subplot(grid[1, 1])
    logomaker.Logo(logo_matrix(passed_hits["minus10_seq"].tolist()), ax=ax_logo10)
    format_logo_axis(ax_logo10, "Passed predicted -10 boxes")
    add_panel_label(ax_logo10, "C")

    ax_spacer = fig.add_subplot(grid[2, 0])
    style_summary_axis(ax_spacer)
    spacer_counts = passed_hits["spacer_length"].value_counts().reindex(range(SPACER_MIN, SPACER_MAX + 1), fill_value=0)
    ax_spacer.bar(spacer_counts.index.astype(str), spacer_counts.values, color=ORANGE, edgecolor=GRAY, linewidth=0.7)
    for index, value in enumerate(spacer_counts.values):
        ax_spacer.text(index, value + 1.0, str(int(value)), ha="center", va="bottom", color=GRAY)
    ax_spacer.set_xlabel("Spacer length (bp)")
    ax_spacer.set_ylabel("Number of sequences")
    ax_spacer.set_title("Best -35/-10 spacer distribution", pad=14)
    add_panel_label(ax_spacer, "D")

    ax_scatter = fig.add_subplot(grid[2, 1])
    style_summary_axis(ax_scatter)
    colors = np.where(hits_df["passes_sigma70_pattern"], RED, LIGHT_BLUE)
    ax_scatter.scatter(hits_df["minus35_score"], hits_df["minus10_score"], c=colors, s=28, alpha=0.82, edgecolor="white", linewidth=0.35)
    ax_scatter.axvline(minus35_threshold, color=GRAY, linestyle="--", linewidth=1.0)
    ax_scatter.axhline(minus10_threshold, color=GRAY, linestyle="--", linewidth=1.0)
    ax_scatter.set_xlabel("-35 PSSM score")
    ax_scatter.set_ylabel("-10 PSSM score")
    ax_scatter.set_title("Motif score thresholding", pad=14)
    ax_scatter.text(minus35_threshold + 0.08, hits_df["minus10_score"].min() + 0.25, "-35 threshold", rotation=90, va="bottom", color=GRAY)
    ax_scatter.text(hits_df["minus35_score"].min() + 0.25, minus10_threshold + 0.15, "-10 threshold", va="bottom", color=GRAY)
    save_all(fig, SUMMARY_PREFIX)
    save_panel(fig, ax_schematic, Path(f"{SUMMARY_PANEL_PREFIX}_A"), xpad=18, ypad=16)
    save_panel(fig, ax_logo35, Path(f"{SUMMARY_PANEL_PREFIX}_B"), xpad=18, ypad=16)
    save_panel(fig, ax_logo10, Path(f"{SUMMARY_PANEL_PREFIX}_C"), xpad=18, ypad=16)
    save_panel(fig, ax_spacer, Path(f"{SUMMARY_PANEL_PREFIX}_D"), xpad=18, ypad=16)
    save_panel(fig, ax_scatter, Path(f"{SUMMARY_PANEL_PREFIX}_E"), xpad=18, ypad=16)
    plt.close(fig)


def plot_logo_comparison() -> None:
    passed_hits = hits_df[hits_df["passes_sigma70_pattern"]]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))

    panels = [
        (axes[0, 0], reference_minus35_boxes, "RegulonDB sigma70 -35"),
        (axes[0, 1], reference_minus10_boxes, "RegulonDB sigma70 -10"),
        (axes[1, 0], passed_hits["minus35_seq"].tolist(), "Predicted passed -35"),
        (axes[1, 1], passed_hits["minus10_seq"].tolist(), "Predicted passed -10"),
    ]
    for ax, boxes, title in panels:
        logomaker.Logo(logo_matrix(boxes), ax=ax)
        format_logo_axis(ax, title)

    plt.tight_layout()
    save_all(fig, LOGO_PREFIX)
    plt.close(fig)


hits_df = pd.read_csv(SCAN_TABLE_PATH, sep="\t")
reference_minus35_boxes, reference_minus10_boxes = load_reference_boxes()
plot_summary_figure(hits_df)
plot_logo_comparison()

print(f"Saved summary figure: {SUMMARY_PREFIX}.pdf/.svg/.png")
print(f"Saved logo comparison: {LOGO_PREFIX}.pdf/.svg/.png")
