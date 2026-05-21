from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"E:\a_ST\CTformer\CTformer-main")
FIG_DIR = ROOT / "report" / "figures"
LATEX_IMG = ROOT / "report" / "zjui_latex_thesis" / "images"
RESIDUAL_DIR = FIG_DIR / "residual_noise"
CTRESTORMER_LOG = ROOT / "model_projects" / "CTRestormer" / "runs" / "ctrestormer_v1" / "train.log"

TRUNC_MIN = -160.0
TRUNC_MAX = 240.0
SLICE_ID = "L506_88"


def save_all(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    fig.savefig(LATEX_IMG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_metric_labels(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_metrics_comparison():
    labels = ["Low-dose", "RED-CNN\n9500", "CTformer\n100000", "CTRestormer\n21500"]
    psnr = [29.2489, 32.6656, 32.6688, 32.6738]
    ssim = [0.8759, 0.9067, 0.9067, 0.9096]
    rmse = [14.2416, 9.4867, 9.5197, 9.4842]

    colors = ["#8c8c8c", "#3b75af", "#59a14f", "#c44e52"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), constrained_layout=True)

    bars = axes[0].bar(labels, psnr, color=colors)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR higher is better")
    axes[0].set_ylim(28.8, 33.0)
    add_metric_labels(axes[0], bars, "{:.4f}")

    bars = axes[1].bar(labels, ssim, color=colors)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM higher is better")
    axes[1].set_ylim(0.865, 0.914)
    add_metric_labels(axes[1], bars, "{:.4f}")

    bars = axes[2].bar(labels, rmse, color=colors)
    axes[2].set_ylabel("RMSE")
    axes[2].set_title("RMSE lower is better")
    axes[2].set_ylim(9.2, 14.7)
    add_metric_labels(axes[2], bars, "{:.4f}")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.28)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Selected Final Results on Held-out Patient L506", fontsize=13)
    save_all(fig, "metrics_comparison.png")


def save_ctformer_checkpoint_progress():
    iters = np.array([21500, 54718, 70000, 100000])
    psnr = np.array([31.8459, 32.3852, 32.4017, 32.6688])
    ssim = np.array([0.9011, 0.9026, 0.9012, 0.9067])
    rmse = np.array([10.5260, 9.8366, 9.8020, 9.5197])

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), constrained_layout=True)

    axes[0].plot(iters, psnr, marker="o", linewidth=2, color="#59a14f", label="CTformer")
    axes[0].axhline(32.6656, linestyle="--", linewidth=1.4, color="#3b75af", label="RED-CNN 9500")
    axes[0].set_title("PSNR")
    axes[0].set_ylabel("dB")
    axes[0].legend(fontsize=8)

    axes[1].plot(iters, ssim, marker="o", linewidth=2, color="#59a14f", label="CTformer")
    axes[1].axhline(0.9067, linestyle="--", linewidth=1.4, color="#3b75af", label="RED-CNN 9500")
    axes[1].set_title("SSIM")

    axes[2].plot(iters, rmse, marker="o", linewidth=2, color="#59a14f", label="CTformer")
    axes[2].axhline(9.4867, linestyle="--", linewidth=1.4, color="#3b75af", label="RED-CNN 9500")
    axes[2].set_title("RMSE")
    axes[2].set_ylabel("lower is better")

    for ax in axes:
        ax.set_xlabel("Training iteration")
        ax.set_xticks(iters)
        ax.set_xticklabels(["21.5k", "54.7k", "70k", "100k"])
        ax.grid(True, linestyle="--", alpha=0.28)

    fig.suptitle("Effect of Continued CTformer Training", fontsize=13)
    save_all(fig, "ctformer_checkpoint_progress.png")


def add_image(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=TRUNC_MIN, vmax=TRUNC_MAX)
    ax.set_title(title, fontsize=10, pad=5)
    ax.axis("off")


def save_qualitative_comparison():
    input_img = np.load(RESIDUAL_DIR / f"{SLICE_ID}_input_hu.npy")
    target_img = np.load(RESIDUAL_DIR / f"{SLICE_ID}_target_hu.npy")
    ctformer = np.load(RESIDUAL_DIR / f"{SLICE_ID}_CTformer_prediction.npy")
    redcnn = np.load(RESIDUAL_DIR / f"{SLICE_ID}_REDCNN_prediction.npy")
    ctrestormer = np.load(RESIDUAL_DIR / f"{SLICE_ID}_CTRestormer_prediction.npy")

    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.2), constrained_layout=True)
    panels = [
        (input_img, "(a) Low-dose input"),
        (ctformer, "(b) CTformer 100k"),
        (redcnn, "(c) RED-CNN 9500"),
        (ctrestormer, "(d) CTRestormer 21.5k"),
        (target_img, "(e) Full-dose target"),
    ]
    for ax, (img, title) in zip(axes, panels):
        add_image(ax, img, title)
    save_all(fig, "qualitative_l506_88_ctformer.png")


def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def parse_logged_losses(log_path):
    losses = []
    pattern = re.compile(r"LOSS:\s*([0-9.eE+-]+)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            losses.append(float(match.group(1)))
    return np.asarray(losses, dtype=np.float64)


def save_ctrestormer_loss_curve():
    losses = parse_logged_losses(CTRESTORMER_LOG)
    if losses.size == 0:
        return

    window = 40
    avg = moving_average(losses, window)
    x = np.arange(losses.size)
    avg_x = np.arange(window - 1, losses.size)

    fig, ax = plt.subplots(figsize=(7.2, 3.9), constrained_layout=True)
    ax.plot(x, losses, linewidth=0.7, alpha=0.38, color="#4e79a7", label="logged loss")
    ax.plot(avg_x, avg, linewidth=1.6, color="#c44e52", label=f"moving average ({window})")
    ax.set_title("CTRestormer training loss trend")
    ax.set_xlabel("Logged training record")
    ax.set_ylabel("Hybrid loss")
    ax.set_ylim(0, np.percentile(losses, 99) * 1.15)
    ax.grid(True, linestyle="--", alpha=0.24)
    ax.legend(fontsize=8)
    save_all(fig, "ctrestormer_loss_curve.png")


def main():
    save_metrics_comparison()
    save_ctformer_checkpoint_progress()
    save_qualitative_comparison()
    save_ctrestormer_loss_curve()
    print("Updated experiment figures in", FIG_DIR, "and", LATEX_IMG)


if __name__ == "__main__":
    main()
