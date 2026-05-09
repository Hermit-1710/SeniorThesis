import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, dropout=0.0):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels, bias=True)
        self.sg = SimpleGate()
        self.sca = SimpleChannelAttention(dw_channels // 2)
        self.pw2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)

        self.norm2 = LayerNorm2d(channels)
        self.pw3 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.pw4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.pw2(x)
        x = self.dropout1(x)
        y = residual + x * self.beta

        x = self.norm2(y)
        x = self.pw3(x)
        x = self.sg2(x)
        x = self.pw4(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=2, stride=2, bias=True)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class CTNAFNet(nn.Module):
    def __init__(self, img_channel=1, width=32, enc_blocks=(1, 1, 2), middle_blocks=4, dec_blocks=(1, 1, 1), dropout=0.0):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = width
        for num_blocks in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels, dropout=dropout) for _ in range(num_blocks)]))
            self.downs.append(Downsample(channels))
            channels *= 2

        self.middle = nn.Sequential(*[NAFBlock(channels, dropout=dropout) for _ in range(middle_blocks)])

        for num_blocks in dec_blocks:
            self.ups.append(Upsample(channels))
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels, dropout=dropout) for _ in range(num_blocks)]))

    def forward(self, x):
        inp = x
        x = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = x + skip
            x = decoder(x)

        return inp + self.ending(x)
