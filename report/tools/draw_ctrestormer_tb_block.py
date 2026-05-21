from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"E:\a_ST\CTformer\CTformer-main")
OUT_REPORT = ROOT / "report" / "figures" / "ctrestormer_tb_block.png"
OUT_LATEX = ROOT / "report" / "zjui_latex_thesis" / "images" / "ctrestormer_tb_block.png"

EDGE = "#303030"
FLOW = "#092a82"
SKIP = "#8a8a8a"
BLUE = "#d7e8f7"
ORANGE = "#f4c9ad"
GREEN = "#d9ead3"
YELLOW = "#fff1b8"
GRAY = "#eeeeee"


def block(ax, x, y, w, h, label, color, fs=9.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=color,
        edgecolor=EDGE,
        linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, linespacing=1.1)


def arrow(ax, start, end, color=FLOW, lw=1.6, ms=10):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def elbow_arrow(ax, points, color=SKIP, lw=1.35, ms=9):
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=lw)
    arrow(ax, points[-2], points[-1], color=color, lw=lw, ms=ms)


def main():
    fig, ax = plt.subplots(figsize=(3.1, 4.9))
    ax.set_xlim(0, 3.1)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    ax.text(1.55, 4.68, "Transformer Block (TB)", ha="center", va="center", fontsize=11.5, fontweight="bold")

    x, w, h = 1.00, 1.10, 0.38
    y_positions = {
        "input": 0.18,
        "ln1": 0.72,
        "mdta": 1.26,
        "add1": 1.80,
        "ln2": 2.34,
        "gdfn": 2.88,
        "add2": 3.42,
        "output": 3.96,
    }

    block(ax, x, y_positions["input"], w, h, "Input\nC x H x W", GRAY, fs=7.4)
    block(ax, x, y_positions["ln1"], w, h, "LN2d", BLUE, fs=8.5)
    block(ax, x, y_positions["mdta"], w, h, "MDTA", ORANGE, fs=8.5)
    block(ax, x + 0.32, y_positions["add1"], 0.46, h, "+", YELLOW, fs=12.5)
    block(ax, x, y_positions["ln2"], w, h, "LN2d", BLUE, fs=8.5)
    block(ax, x, y_positions["gdfn"], w, h, "GDFN", GREEN, fs=8.5)
    block(ax, x + 0.32, y_positions["add2"], 0.46, h, "+", YELLOW, fs=12.5)
    block(ax, x, y_positions["output"], w, h, "Output\nC x H x W", GRAY, fs=7.4)

    cx = x + w / 2
    for lower, upper in [
        ("input", "ln1"),
        ("ln1", "mdta"),
        ("mdta", "add1"),
        ("add1", "ln2"),
        ("ln2", "gdfn"),
        ("gdfn", "add2"),
        ("add2", "output"),
    ]:
        y0 = y_positions[lower] + h
        y1 = y_positions[upper]
        arrow(ax, (cx, y0 + 0.04), (cx, y1 - 0.04))

    # Residual shortcuts, drawn as straight orthogonal paths.
    add_x_left = x + 0.32
    add_x_right = x + 0.78
    elbow_arrow(
        ax,
        [
            (x, y_positions["input"] + h / 2),
            (0.52, y_positions["input"] + h / 2),
            (0.52, y_positions["add1"] + h / 2),
            (add_x_left - 0.04, y_positions["add1"] + h / 2),
        ],
    )
    elbow_arrow(
        ax,
        [
            (add_x_right, y_positions["add1"] + h / 2),
            (2.58, y_positions["add1"] + h / 2),
            (2.58, y_positions["add2"] + h / 2),
            (add_x_right + 0.04, y_positions["add2"] + h / 2),
        ],
    )

    fig.tight_layout(pad=0.08)
    for out in (OUT_REPORT, OUT_LATEX):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=360, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_REPORT}")
    print(f"Saved {OUT_LATEX}")


if __name__ == "__main__":
    main()
