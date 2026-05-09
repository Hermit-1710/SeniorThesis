import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=2.66, bias=True):
        super().__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=True):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim, expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, inp_channels=1, embed_dim=32, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(inp_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class CTRestormer(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=32,
        num_blocks=(2, 3, 3, 4),
        num_refinement_blocks=2,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias=bias)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim * 4, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=dim * 8, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=dim * 4, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=dim * 2, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        return self.output(out_dec_level1) + inp_img
