"""Conceptual figures for docs/06 (no training; pure matplotlib).

  fig12 - Level-8 'place-then-harvest' architecture schematic
  fig13 - control-validity decision diagram (operationalises Proposition 1 / Corollary 2)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})

BLUE, RED, GREEN, GREY, PURPLE = "#4C72B0", "#C44E52", "#55A868", "#8C8C8C", "#8172B3"


def _box(ax, x, y, w, h, text, fc, fontsize=9.5, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="black", lw=1.1, alpha=0.95))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=tc, wrap=True)


def _arrow(ax, x0, y0, x1, y1, color="black"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=color, shrinkA=2, shrinkB=2))


def schematic():
    fig, ax = plt.subplots(figsize=(14.4, 4.6))
    ax.set_xlim(-1, 109); ax.set_ylim(0, 40); ax.axis("off")
    y, h = 14, 13
    geo = [
        (1.5, 12, "Molecule\n(SMILES /\ngraph)", "#EAEAF2"),
        (15.5, 16.5, "Coarse-grain →\nK qubit-nodes\n+ adjacency A", "#EAEAF2"),
        (35, 16.5, "Encode (FIXED)\nLinear(5→2)\nRY, RZ /qubit", BLUE),
        (54.5, 16.5, "PLACE\nIsingXX(A·θ)\nentangler", RED),
        (73, 16.5, "MEASURE\n1- & 2-qubit\nPauli expectations", PURPLE),
        (90.5, 16.5, "HARVEST → head\nbond-pool\n→ 12 tasks", GREEN),
    ]
    for x, w, t, fc in geo:
        tc = "white" if fc in (RED, BLUE, GREEN, PURPLE) else "black"
        _box(ax, x, y, w, h, t, fc, tc=tc)
    centers = [(g[0] + g[1] / 2) for g in geo]
    rights = [(g[0] + g[1]) for g in geo]
    for i in range(len(geo) - 1):
        _arrow(ax, rights[i], y + h / 2, geo[i + 1][0], y + h / 2)

    # annotation: A is fixed data (the non-absorbable injection) feeding PLACE and HARVEST
    ax.text(15.5 + 8.5, y - 3.2, "A = fixed per-molecule data (Corollary 2b: non-absorbable)",
            ha="center", va="top", fontsize=9, color=RED, style="italic")
    _arrow(ax, 15.5 + 8.5, y, 54.5 + 8, y - 2.4, color=RED)
    _arrow(ax, 15.5 + 8.5, y, 90.5 + 7, y - 2.4, color=RED)
    ax.plot([54.5 + 8, 90.5 + 7], [y - 2.4, y - 2.4], color=RED, lw=0.8, ls=":")

    ax.text(54.5 + 8, y + h + 2.6, "place quantum correlation\nON true bonds", ha="center",
            va="bottom", fontsize=8.5, color=RED)
    ax.text(90.5 + 7, y + h + 2.6, "harvest it back\nALONG the same bonds", ha="center",
            va="bottom", fontsize=8.5, color=GREEN)
    ax.set_title("Level 8: a measurement-based, non-absorbable quantum graph inductive bias",
                 fontsize=13, fontweight="bold", pad=14)
    fig.savefig(os.path.join(OUT, "fig12_schematic.png"))
    plt.close(fig)
    print("wrote fig12_schematic.png", flush=True)


def _diamond(ax, cx, cy, w, h, text, fc="#FFF3CC"):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="black", lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.8)


def decision():
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    lbl = dict(fontsize=9, ha="center", va="center",
               bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    _box(ax, 30, 90, 40, 7, "Structured signal enters the circuit", "#EAEAF2", 10)
    _arrow(ax, 50, 90, 50, 83)
    _diamond(ax, 50, 77, 46, 13, "Does it pass through a\ntrainable layer first?")
    # NO -> condition (b)
    _arrow(ax, 27, 77, 14, 77); ax.text(20.5, 77, "No", color=GREEN, **lbl)
    _box(ax, 1, 62, 26, 12, "Fixed per-molecule data\nupstream of the only\ntrainable layer (cond. b)",
         "#EAEAF2", 8.5)
    _arrow(ax, 14, 62, 14, 58)
    _box(ax, 1, 47, 26, 10, "GENUINE & SCALABLE\nLevel 8", GREEN, 9.5, tc="white")
    # YES -> condition (a)
    _arrow(ax, 50, 70.5, 50, 60); ax.text(50, 65, "Yes", color=GREY, **lbl)
    _diamond(ax, 50, 52, 50, 15, "Same vector reused under\n≥2 inconsistent\npermutations? (cond. a)")
    # cond a yes -> genuine
    _arrow(ax, 75, 52, 84, 52); ax.text(79.5, 52, "Yes", color=GREEN, **lbl)
    _box(ax, 73, 46.5, 26, 11, "GENUINE\nLevels 5–7", GREEN, 9.5, tc="white")
    # cond a no -> vacuous
    _arrow(ax, 50, 44.5, 50, 33); ax.text(50, 39, "No", color=RED, **lbl)
    _box(ax, 33, 22, 34, 11, "ABSORBABLE / VACUOUS\nLevels 1 / 2 / 4\n(residual = 0, bit-exact)",
         RED, 9.2, tc="white")
    ax.text(50, 10, "Proposition 1 / Corollary 2 — apply BEFORE running the experiment",
            ha="center", fontsize=9, style="italic", color="#444")
    ax.set_title("Is your structured-vs-scrambled control valid?", fontsize=13,
                 fontweight="bold", pad=10)
    fig.savefig(os.path.join(OUT, "fig13_control_decision.png"))
    plt.close(fig)
    print("wrote fig13_control_decision.png", flush=True)


if __name__ == "__main__":
    schematic()
    decision()
    print("SCHEMATICS DONE", flush=True)
