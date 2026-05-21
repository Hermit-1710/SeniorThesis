from pathlib import Path
import csv
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"E:\a_ST\CTformer\CTformer-main")
DATA = ROOT / "report" / "figures" / "residual_noise"
OUT = ROOT / "report" / "figures" / "standardized"
LATEX_IMG = ROOT / "report" / "zjui_latex_thesis" / "images"
OUT.mkdir(parents=True, exist_ok=True)
LATEX_IMG.mkdir(parents=True, exist_ok=True)

SLICE_ID = "L506_88"
TRUNC_MIN = -160.0
TRUNC_MAX = 240.0


def load(name):
    return np.load(DATA / name)


def add_image(ax, img, title, cmap="gray", vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, pad=5)
    ax.axis("off")
    return im


def save_prediction_comparison(input_img, target_img, preds):
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.2), constrained_layout=True)
    add_image(axes[0], input_img, "(a) Low-dose input", vmin=TRUNC_MIN, vmax=TRUNC_MAX)
    add_image(axes[1], target_img, "(b) Full-dose target", vmin=TRUNC_MIN, vmax=TRUNC_MAX)
    for ax, (name, pred) in zip(axes[2:], preds.items()):
        add_image(ax, pred, f"({chr(99 + list(preds).index(name))}) {name}", vmin=TRUNC_MIN, vmax=TRUNC_MAX)
    fig.savefig(OUT / f"{SLICE_ID}_prediction_comparison_standard.png", dpi=300)
    plt.close(fig)


def save_removed_residuals(input_img, target_img, preds):
    residuals = {"Input - target": input_img - target_img}
    residuals.update({f"Input - {name}": input_img - pred for name, pred in preds.items()})
    vmax = max(np.percentile(np.abs(v), 99.5) for v in residuals.values())
    vmax = max(vmax, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.3), constrained_layout=True)
    last = None
    labels = ["(a) Reference residual", "(b) Removed by CTformer", "(c) Removed by CTRestormer", "(d) Removed by RED-CNN"]
    for ax, label, (_, residual) in zip(axes, labels, residuals.items()):
        last = add_image(ax, residual, label, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(last, ax=axes, shrink=0.82, location="right")
    cbar.set_label("Residual intensity (HU)", fontsize=9)
    fig.savefig(OUT / f"{SLICE_ID}_removed_residuals_standard.png", dpi=300)
    plt.close(fig)


def save_prediction_errors(target_img, preds):
    errors = {f"{name} - target": pred - target_img for name, pred in preds.items()}
    vmax = max(np.percentile(np.abs(v), 99.5) for v in errors.values())
    vmax = max(vmax, 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.3), constrained_layout=True)
    last = None
    labels = ["(a) CTformer error", "(b) CTRestormer error", "(c) RED-CNN error"]
    for ax, label, (_, error) in zip(axes, labels, errors.items()):
        last = add_image(ax, error, label, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    cbar = fig.colorbar(last, ax=axes, shrink=0.82, location="right")
    cbar.set_label("Error intensity (HU)", fontsize=9)
    fig.savefig(OUT / f"{SLICE_ID}_prediction_errors_standard.png", dpi=300)
    plt.close(fig)


def save_metrics_bar():
    models = ["CTformer", "CTRestormer", "RED-CNN"]
    metrics_path = DATA / f"{SLICE_ID}_residual_metrics.csv"
    by_model = {}
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_model[row["model"]] = row
    residual_corr = [float(by_model[name]["residual_corr_to_input_minus_target"]) for name in models]
    edge_leakage = [float(by_model[name]["edge_leakage_corr"]) for name in models]
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.bar(x - width / 2, residual_corr, width, label="Residual correlation")
    ax.bar(x + width / 2, edge_leakage, width, label="Edge leakage correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Correlation")
    ax.set_title("Residual Similarity and Edge Leakage on L506_88")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    fig.savefig(OUT / f"{SLICE_ID}_residual_metrics_bar_standard.png", dpi=300)
    plt.close(fig)


def main():
    input_img = load(f"{SLICE_ID}_input_hu.npy")
    target_img = load(f"{SLICE_ID}_target_hu.npy")
    preds = {
        "CTformer": load(f"{SLICE_ID}_CTformer_prediction.npy"),
        "CTRestormer": load(f"{SLICE_ID}_CTRestormer_prediction.npy"),
        "RED-CNN": load(f"{SLICE_ID}_REDCNN_prediction.npy"),
    }
    save_prediction_comparison(input_img, target_img, preds)
    save_removed_residuals(input_img, target_img, preds)
    save_prediction_errors(target_img, preds)
    save_metrics_bar()
    for png in OUT.glob(f"{SLICE_ID}_*.png"):
        shutil.copy2(png, LATEX_IMG / png.name)
    print("Saved standardized figures to", OUT)


if __name__ == "__main__":
    main()
