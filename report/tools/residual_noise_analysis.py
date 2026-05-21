import argparse
import csv
import importlib
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


NORM_MIN = -1024.0
NORM_MAX = 3072.0
TRUNC_MIN = -160.0
TRUNC_MAX = 240.0


def denormalize(arr):
    return arr * (NORM_MAX - NORM_MIN) + NORM_MIN


def truncate(arr):
    return np.clip(arr, TRUNC_MIN, TRUNC_MAX)


def split_arr(arr, patch_size=64, stride=32):
    pad = (16, 16, 16, 16)
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _, _, h, _ = arr.shape
    num = h // stride - 1
    arrs = torch.zeros(num * num, 1, patch_size, patch_size)
    for i in range(num):
        for j in range(num):
            arrs[i * num + j, 0] = arr[0, 0, i * stride:i * stride + patch_size, j * stride:j * stride + patch_size]
    return arrs


def agg_arr(arrs, size, stride=32):
    arr = torch.zeros(size, size)
    num = size // stride
    for i in range(num):
        for j in range(num):
            arr[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride] = arrs[i * num + j, :, 16:48, 16:48]
    return arr.unsqueeze(0).unsqueeze(1)


def corr(a, b):
    av = np.asarray(a, dtype=np.float64).ravel()
    bv = np.asarray(b, dtype=np.float64).ravel()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.sqrt(np.sum(av * av) * np.sum(bv * bv))
    if denom == 0:
        return float("nan")
    return float(np.sum(av * bv) / denom)


def grad_mag(arr):
    gy, gx = np.gradient(arr.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)


def residual_metrics(input_img, target_img, pred_img):
    ref = input_img - target_img
    removed = input_img - pred_img
    err = pred_img - target_img
    edge = grad_mag(target_img)
    removed_edge = grad_mag(removed)
    return {
        "removed_mean": float(np.mean(removed)),
        "removed_std": float(np.std(removed)),
        "removed_mae": float(np.mean(np.abs(removed))),
        "error_mean": float(np.mean(err)),
        "error_std": float(np.std(err)),
        "error_mae": float(np.mean(np.abs(err))),
        "residual_corr_to_input_minus_target": corr(removed, ref),
        "edge_leakage_corr": corr(removed_edge, edge),
    }


def radial_power_spectrum(arr):
    centered = arr - np.mean(arr)
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    h, w = power.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2).astype(np.int32)
    sums = np.bincount(r.ravel(), weights=power.ravel())
    counts = np.bincount(r.ravel())
    return sums / np.maximum(counts, 1)


def load_ctformer(ctformer_code, ckpt_path, device):
    sys.path.insert(0, str(ctformer_code))
    try:
        from CTformer import CTformer
        model = CTformer(
            img_size=64,
            tokens_type="performer",
            embed_dim=64,
            depth=1,
            num_heads=8,
            kernel=8,
            stride=4,
            mlp_ratio=2.0,
            token_dim=64,
        )
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        return model
    finally:
        sys.path.remove(str(ctformer_code))


def load_ctrestormer(ctrestormer_code, ckpt_path, device):
    sys.path.insert(0, str(ctrestormer_code))
    try:
        module = importlib.import_module("ctrestormer")
        model = module.CTRestormer(
            inp_channels=1,
            out_channels=1,
            dim=24,
            num_blocks=(1, 2, 2, 3),
            num_refinement_blocks=1,
            heads=(1, 2, 4, 4),
            ffn_expansion_factor=2.66,
        )
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        return model
    finally:
        sys.path.remove(str(ctrestormer_code))


def load_redcnn(redcnn_code, ckpt_path, device):
    sys.path.insert(0, str(redcnn_code))
    try:
        from networks import RED_CNN
        model = RED_CNN()
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state)
        model.to(device).eval()
        return model
    finally:
        sys.path.remove(str(redcnn_code))


@torch.no_grad()
def predict_ctformer(model, x, device):
    x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
    tiles = split_arr(x_t.cpu(), 64, stride=32).to(device)
    preds = torch.zeros_like(tiles)
    for start in range(0, tiles.shape[0], 64):
        preds[start:start + 64] = model(tiles[start:start + 64])
    pred = agg_arr(preds.cpu(), x.shape[-1], stride=32)
    return pred.squeeze().numpy()


@torch.no_grad()
def predict_ctrestormer(model, x, device):
    x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
    tiles = split_arr(x_t.cpu(), 64, stride=32).to(device)
    preds = torch.zeros_like(tiles)
    for start in range(0, tiles.shape[0], 16):
        preds[start:start + 16] = model(tiles[start:start + 16])
    pred = agg_arr(preds.cpu(), x.shape[-1], stride=32)
    return pred.squeeze().numpy()


@torch.no_grad()
def predict_full_image(model, x, device):
    x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to(device)
    return model(x_t).squeeze().cpu().numpy()


def imshow(ax, img, title, vmin=None, vmax=None, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def save_panel(out_path, input_img, target_img, preds, metrics):
    ref_res = input_img - target_img
    vmax_res = np.percentile(np.abs(ref_res), 99)
    for pred in preds.values():
        vmax_res = max(vmax_res, np.percentile(np.abs(input_img - pred), 99))
        vmax_res = max(vmax_res, np.percentile(np.abs(pred - target_img), 99))
    vmax_res = max(vmax_res, 1.0)

    names = list(preds.keys())
    fig, axes = plt.subplots(3, len(names) + 2, figsize=(4 * (len(names) + 2), 11))
    imshow(axes[0, 0], input_img, "Low-dose input", TRUNC_MIN, TRUNC_MAX)
    imshow(axes[0, 1], target_img, "Full-dose target", TRUNC_MIN, TRUNC_MAX)
    for idx, name in enumerate(names):
        imshow(axes[0, idx + 2], preds[name], f"{name} prediction", TRUNC_MIN, TRUNC_MAX)

    imshow(axes[1, 0], ref_res, "input - target", -vmax_res, vmax_res, cmap="coolwarm")
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Reference residual", fontsize=11)
    for idx, name in enumerate(names):
        imshow(axes[1, idx + 2], input_img - preds[name], f"input - {name}", -vmax_res, vmax_res, cmap="coolwarm")

    axes[2, 0].axis("off")
    axes[2, 0].set_title("Prediction error", fontsize=11)
    axes[2, 1].axis("off")
    for idx, name in enumerate(names):
        imshow(axes[2, idx + 2], preds[name] - target_img, f"{name} - target", -vmax_res, vmax_res, cmap="coolwarm")

    fig.suptitle("Residual noise analysis: removed residuals and prediction errors", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_histogram(out_path, input_img, target_img, preds):
    plt.figure(figsize=(9, 5))
    ref = (input_img - target_img).ravel()
    plt.hist(ref, bins=140, density=True, histtype="step", linewidth=2, label="input - target")
    for name, pred in preds.items():
        plt.hist((input_img - pred).ravel(), bins=140, density=True, histtype="step", linewidth=1.6, label=f"input - {name}")
    plt.xlabel("Residual intensity (HU)")
    plt.ylabel("Density")
    plt.title("Residual value distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_power_spectrum(out_path, input_img, target_img, preds):
    plt.figure(figsize=(9, 5))
    ref_curve = radial_power_spectrum(input_img - target_img)
    plt.semilogy(ref_curve[:180], linewidth=2, label="input - target")
    for name, pred in preds.items():
        curve = radial_power_spectrum(input_img - pred)
        plt.semilogy(curve[:180], linewidth=1.6, label=f"input - {name}")
    plt.xlabel("Radial spatial frequency bin")
    plt.ylabel("Power (log scale)")
    plt.title("Residual noise power spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"E:\a_ST\CTformer\CTformer-main")
    parser.add_argument("--redcnn", default=r"E:\a_ST\RED-CNN")
    parser.add_argument("--slice-id", default="L506_88")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    repo = Path(args.repo)
    out_dir = repo / "report" / "figures" / "residual_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = repo / "npy_img_3mm_B30"
    input_norm = np.load(data_dir / f"{args.slice_id}_input.npy").astype(np.float32)
    target_norm = np.load(data_dir / f"{args.slice_id}_target.npy").astype(np.float32)
    input_img = truncate(denormalize(input_norm))
    target_img = truncate(denormalize(target_norm))

    device = torch.device(args.device)
    preds = {}
    pred_norm = {}

    ctformer = load_ctformer(
        repo / "model_projects" / "CTformer" / "code",
        repo / "model_projects" / "CTformer" / "runs" / "train_100ep_bs4" / "T2T_vit_100000iter.ckpt",
        device,
    )
    pred_norm["CTformer"] = predict_ctformer(ctformer, input_norm, device)
    del ctformer

    ctrestormer = load_ctrestormer(
        repo / "model_projects" / "CTRestormer" / "code",
        repo / "model_projects" / "CTRestormer" / "runs" / "ctrestormer_v1" / "ctrestormer_21500iter.ckpt",
        device,
    )
    pred_norm["CTRestormer"] = predict_ctrestormer(ctrestormer, input_norm, device)
    del ctrestormer

    red_ckpt = Path(args.redcnn) / "save_cuda_new" / "REDCNN_9500iter.ckpt"
    if red_ckpt.exists():
        redcnn = load_redcnn(Path(args.redcnn), red_ckpt, device)
        pred_norm["RED-CNN"] = predict_full_image(redcnn, input_norm, device)
        del redcnn

    for name, arr in pred_norm.items():
        preds[name] = truncate(denormalize(arr))
        np.save(out_dir / f"{args.slice_id}_{name.replace('-', '').replace(' ', '_')}_prediction.npy", preds[name])
        np.save(out_dir / f"{args.slice_id}_{name.replace('-', '').replace(' ', '_')}_removed_residual.npy", input_img - preds[name])

    np.save(out_dir / f"{args.slice_id}_input_hu.npy", input_img)
    np.save(out_dir / f"{args.slice_id}_target_hu.npy", target_img)
    np.save(out_dir / f"{args.slice_id}_input_minus_target.npy", input_img - target_img)

    rows = []
    for name, pred in preds.items():
        row = {"model": name}
        row.update(residual_metrics(input_img, target_img, pred))
        rows.append(row)
    metrics_path = out_dir / f"{args.slice_id}_residual_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_panel(out_dir / f"{args.slice_id}_residual_panel.png", input_img, target_img, preds, rows)
    save_histogram(out_dir / f"{args.slice_id}_residual_histogram.png", input_img, target_img, preds)
    save_power_spectrum(out_dir / f"{args.slice_id}_residual_power_spectrum.png", input_img, target_img, preds)

    print("Saved residual analysis to:", out_dir)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
