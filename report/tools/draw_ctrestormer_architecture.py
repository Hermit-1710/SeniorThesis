from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(r"E:\a_ST\CTformer\CTformer-main")
OUT_REPORT = ROOT / "report" / "figures" / "ctrestormer_architecture.png"
OUT_LATEX = ROOT / "report" / "zjui_latex_thesis" / "images" / "ctrestormer_architecture.png"


BLUE = "#d7e8f7"
ORANGE = "#f4c9ad"
PURPLE = "#d8c6f1"
GREEN = "#d9ead3"
YELLOW = "#fff1b8"
RED = "#b22222"
UP = "#008b2f"
SKIP = "#9a9a9a"
FLOW = "#092a82"
EDGE = "#303030"


def block(ax, x, y, w, h, text, fc, fs=8.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        facecolor=fc,
        edgecolor=EDGE,
        linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#111111",
        linespacing=1.12,
    )


def arrow(ax, start, end, color=FLOW, lw=1.65, ms=11, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def text(ax, x, y, s, color="#333333", fs=7.2):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color)


def main():
    fig, ax = plt.subplots(figsize=(7.4, 4.55))
    ax.set_xlim(0, 7.4)
    ax.set_ylim(0, 4.55)
    ax.axis("off")

    ax.text(3.70, 4.30, "CTRestormer Architecture", ha="center", va="center", fontsize=13, fontweight="bold")

    w, h = 1.03, 0.52
    x_in, x_l, x_c, x_r, x_out = 0.10, 1.46, 3.18, 4.90, 6.25
    y0, y1, y2, y3, y4 = 3.55, 2.82, 2.09, 1.36, 0.30

    block(ax, x_in, y0, w, h, "Input\n1x64x64", GREEN, fs=7.3)
    block(ax, x_l, y0, w, h, "Patch Embed\n3x3 Conv\n24x64x64", YELLOW, fs=6.8)
    block(ax, x_l, y1, w, h, "Encoder 1\nTB x1\n24x64x64", BLUE, fs=7.0)
    block(ax, x_l, y2, w, h, "Encoder 2\nTB x2\n48x32x32", BLUE, fs=7.0)
    block(ax, x_l, y3, w, h, "Encoder 3\nTB x2\n96x16x16", BLUE, fs=7.0)
    block(ax, x_c, y4, w, h, "Latent\nTB x3\n192x8x8", PURPLE, fs=7.0)
    block(ax, x_r, y3, w, h, "Decoder 3\nTB x2\n96x16x16", ORANGE, fs=7.0)
    block(ax, x_r, y2, w, h, "Decoder 2\nTB x2\n48x32x32", ORANGE, fs=7.0)
    block(ax, x_r, y1, w, h, "Decoder 1\nTB x1\n48x64x64", ORANGE, fs=7.0)
    block(ax, x_r, y0, w, h, "Refinement\nTB x1\n48x64x64", YELLOW, fs=6.8)
    block(ax, x_out, y0, w, h, "Output Conv\n+ LDCT\n1x64x64", GREEN, fs=6.8)

    mid = h / 2
    # Forward path.
    arrow(ax, (x_in + w, y0 + mid), (x_l - 0.07, y0 + mid), FLOW)
    arrow(ax, (x_l + w / 2, y0), (x_l + w / 2, y1 + h), RED)
    arrow(ax, (x_l + w / 2, y1), (x_l + w / 2, y2 + h), RED)
    arrow(ax, (x_l + w / 2, y2), (x_l + w / 2, y3 + h), RED)
    arrow(ax, (x_l + w * 0.80, y3), (x_c + w * 0.28, y4 + h), RED)

    arrow(ax, (x_c + w * 0.72, y4 + h), (x_r + w * 0.20, y3), UP)
    arrow(ax, (x_r + w / 2, y3 + h), (x_r + w / 2, y2), UP)
    arrow(ax, (x_r + w / 2, y2 + h), (x_r + w / 2, y1), UP)
    arrow(ax, (x_r + w / 2, y1 + h), (x_r + w / 2, y0), FLOW)
    arrow(ax, (x_r + w, y0 + mid), (x_out - 0.07, y0 + mid), FLOW)

    # Skip connections.
    arrow(ax, (x_l + w, y1 + mid), (x_r - 0.07, y1 + mid), SKIP, lw=1.35, ms=9)
    arrow(ax, (x_l + w, y2 + mid), (x_r - 0.07, y2 + mid), SKIP, lw=1.35, ms=9)
    arrow(ax, (x_l + w, y3 + mid), (x_r - 0.07, y3 + mid), SKIP, lw=1.35, ms=9)

    fig.tight_layout(pad=0.08)
    for out in [OUT_REPORT, OUT_LATEX]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=360, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_REPORT}")
    print(f"Saved {OUT_LATEX}")


if __name__ == "__main__":
    main()
