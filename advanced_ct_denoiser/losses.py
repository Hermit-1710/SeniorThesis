import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def gaussian(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def create_window(window_size, channel, device, dtype):
    kernel_1d = gaussian(window_size, 1.5, device, dtype).unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    return kernel_2d.expand(channel, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        channel = pred.shape[1]
        window = create_window(self.window_size, channel, pred.device, pred.dtype)
        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
        return 1.0 - ssim_map.mean()


class HybridLoss(nn.Module):
    def __init__(self, charbonnier_weight=0.8, ssim_weight=0.2):
        super().__init__()
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        return self.charbonnier_weight * self.charbonnier(pred, target) + self.ssim_weight * self.ssim(pred, target)
