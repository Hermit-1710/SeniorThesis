from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOL_DIR))
from residual_noise_analysis import (  # noqa: E402
    TRUNC_MAX,
    TRUNC_MIN,
    denormalize,
    load_ctformer,
    load_ctrestormer,
    load_redcnn,
    predict_ctformer,
    predict_ctrestormer,
    predict_full_image,
    truncate,
)
sys.path.remove(str(TOOL_DIR))


ROOT = Path(r"E:\a_ST\CTformer\CTformer-main")
RED_ROOT = Path(r"E:\a_ST\RED-CNN")
DATA_DIR = ROOT / "npy_img_3mm_B30"
FIG_DIR = ROOT / "report" / "figures"
LATEX_IMG = ROOT / "report" / "zjui_latex_thesis" / "images"


def save_all(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight")
    fig.savefig(LATEX_IMG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_pair(slice_id):
    input_norm = np.load(DATA_DIR / f"{slice_id}_input.npy").astype(np.float32)
    target_norm = np.load(DATA_DIR / f"{slice_id}_target.npy").astype(np.float32)
    input_hu = truncate(denormalize(input_norm))
    target_hu = truncate(denormalize(target_norm))
    return input_norm, input_hu, target_hu


def add_panel(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=TRUNC_MIN, vmax=TRUNC_MAX)
    ax.set_title(title, fontsize=10, pad=5)
    ax.axis("off")


def save_qualitative(slice_id, models, device):
    input_norm, input_hu, target_hu = load_pair(slice_id)
    ctformer, redcnn, ctrestormer = models

    preds = {
        "CTformer 100k": truncate(denormalize(predict_ctformer(ctformer, input_norm, device))),
        "RED-CNN 9500": truncate(denormalize(predict_full_image(redcnn, input_norm, device))),
        "CTRestormer 21.5k": truncate(denormalize(predict_ctrestormer(ctrestormer, input_norm, device))),
    }

    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.2), constrained_layout=True)
    panels = [
        (input_hu, "(a) Low-dose input"),
        (preds["CTformer 100k"], "(b) CTformer 100k"),
        (preds["RED-CNN 9500"], "(c) RED-CNN 9500"),
        (preds["CTRestormer 21.5k"], "(d) CTRestormer 21.5k"),
        (target_hu, "(e) Full-dose target"),
    ]
    for ax, (img, title) in zip(axes, panels):
        add_panel(ax, img, title)
    save_all(fig, f"qualitative_{slice_id.lower()}_comparison.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctformer = load_ctformer(
        ROOT / "model_projects" / "CTformer" / "code",
        ROOT / "model_projects" / "CTformer" / "runs" / "train_100ep_bs4" / "T2T_vit_100000iter.ckpt",
        device,
    )
    redcnn = load_redcnn(RED_ROOT, RED_ROOT / "save_cuda_new" / "REDCNN_9500iter.ckpt", device)
    ctrestormer = load_ctrestormer(
        ROOT / "model_projects" / "CTRestormer" / "code",
        ROOT / "model_projects" / "CTRestormer" / "runs" / "ctrestormer_v1" / "ctrestormer_21500iter.ckpt",
        device,
    )

    models = (ctformer, redcnn, ctrestormer)
    for slice_id in ["L506_45", "L506_180"]:
        save_qualitative(slice_id, models, device)
        print("saved", slice_id)


if __name__ == "__main__":
    main()
