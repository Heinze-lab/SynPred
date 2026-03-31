"""
model.py  –  Dual-head U-Net for synaptic partner detection (setup03 / dh_unet).

Shared encoder → two independent decoders:
  • mask_decoder   → pred_syn_indicators  (1-ch, sigmoid)
  • vec_decoder    → pred_partner_vectors (3-ch, linear)

Driven entirely by the parameter JSON:
  fmap_num, fmap_inc_factor, downsample_factors, kernel_size
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _make_norm(num_channels: int, norm_type: str = "group", target_groups: int = 4) -> nn.Module:
    """BatchNorm3d or GroupNorm (up to target_groups, falling back to largest valid divisor)."""
    if norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    g = target_groups
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvBlock(nn.Module):
    """Two conv→Norm→ReLU layers (same-padding so spatial dims are preserved)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, norm_type: str = "group"):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            _make_norm(out_ch, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            _make_norm(out_ch, norm_type),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder (shared between both heads)
# ---------------------------------------------------------------------------

class UNetEncoder(nn.Module):
    """
    Hierarchical encoder.

    Level fmaps:  fmap_num * fmap_inc_factor^i  for i = 0 … n_levels-1
    n_levels = len(downsample_factors) + 1   (last level is the bottleneck)
    """

    def __init__(
        self,
        in_channels: int,
        fmap_num: int,
        fmap_inc_factor: int,
        downsample_factors: List[List[int]],
        kernel_size: int = 3,
        norm_type: str = "group",
    ):
        super().__init__()
        n_levels = len(downsample_factors) + 1
        self.fmaps: List[int] = [
            int(fmap_num * (fmap_inc_factor ** i)) for i in range(n_levels)
        ]

        self.conv_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for i in range(n_levels):
            self.conv_blocks.append(ConvBlock(in_ch, self.fmaps[i], kernel_size, norm_type))
            in_ch = self.fmaps[i]
            if i < len(downsample_factors):
                ds = [int(d) for d in downsample_factors[i]]
                self.pools.append(
                    nn.MaxPool3d(kernel_size=ds, stride=ds)
                )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (bottleneck_features, [skip_0, skip_1, …]) fine→coarse."""
        skips: List[torch.Tensor] = []
        for i, conv in enumerate(self.conv_blocks):
            x = conv(x)
            if i < len(self.pools):
                skips.append(x)
                x = self.pools[i](x)
        return x, skips


# ---------------------------------------------------------------------------
# Decoder (one instance per head)
# ---------------------------------------------------------------------------

class UNetDecoder(nn.Module):
    """
    Hierarchical decoder.  Matches skips from the encoder via center-crop
    before concatenation (safe for both same- and valid-padded encoders).
    """

    def __init__(
        self,
        fmaps: List[int],
        downsample_factors: List[List[int]],
        kernel_size: int = 3,
        norm_type: str = "group",
    ):
        super().__init__()
        n_levels = len(downsample_factors)
        self.upsamples = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()

        # iterate from bottleneck → finest resolution
        for i in range(n_levels):
            level = n_levels - 1 - i       # encoder level we're merging with
            in_ch = fmaps[level + 1]       # channels coming from below
            skip_ch = fmaps[level]         # skip channels from encoder
            out_ch = fmaps[level]
            ds = [int(d) for d in downsample_factors[level]]

            self.upsamples.append(
                nn.ConvTranspose3d(in_ch, in_ch, kernel_size=ds, stride=ds)
            )
            self.conv_blocks.append(
                ConvBlock(in_ch + skip_ch, out_ch, kernel_size, norm_type)
            )

    def forward(
        self, x: torch.Tensor, skips: List[torch.Tensor]
    ) -> torch.Tensor:
        for up, conv, skip in zip(
            self.upsamples, self.conv_blocks, reversed(skips)
        ):
            x = up(x)
            skip = self._center_crop(skip, x)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)
        return x

    @staticmethod
    def _center_crop(
        skip: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Crop skip to the spatial shape of target (handles any size mismatch)."""
        slices = [slice(None), slice(None)]
        for d in range(2, 5):
            diff = skip.shape[d] - target.shape[d]
            start = diff // 2
            slices.append(slice(start, start + target.shape[d]))
        return skip[slices]


# ---------------------------------------------------------------------------
# Full dual-head model
# ---------------------------------------------------------------------------

class DHUNet(nn.Module):
    """
    Dual-Head U-Net: one shared encoder, two independent decoders.

    Outputs:
        pred_mask : (B, 1, Z, Y, X)  – sigmoid probability of postsynaptic site
        pred_vec  : (B, 3, Z, Y, X)  – direction vector (post → pre, linear)
    """

    def __init__(
        self,
        in_channels: int = 1,
        fmap_num: int = 6,
        fmap_inc_factor: int = 6,
        downsample_factors: List[List[int]] | None = None,
        kernel_size: int = 3,
        norm_type: str = "group",
    ):
        super().__init__()
        if downsample_factors is None:
            downsample_factors = [[1, 3, 3], [1, 3, 3], [3, 3, 3]]

        self.encoder = UNetEncoder(
            in_channels, fmap_num, fmap_inc_factor, downsample_factors, kernel_size, norm_type
        )
        fmaps = self.encoder.fmaps

        self.mask_decoder = UNetDecoder(fmaps, downsample_factors, kernel_size, norm_type)
        self.vec_decoder = UNetDecoder(fmaps, downsample_factors, kernel_size, norm_type)

        # Output heads
        self.mask_head = nn.Conv3d(fmaps[0], 1, kernel_size=1)  # raw logits, no sigmoid
        self.vec_head = nn.Conv3d(fmaps[0], 3, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bottleneck, skips = self.encoder(x)
        pred_mask = self.mask_head(self.mask_decoder(bottleneck, skips))
        pred_vec  = self.vec_head(self.vec_decoder(bottleneck, skips))
        return pred_mask, pred_vec   # always (mask, vec)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(params: dict) -> DHUNet:
    """Instantiate DHUNet from the parameter JSON dict."""
    return DHUNet(
        in_channels=1,
        fmap_num=params["fmap_num"],
        fmap_inc_factor=params["fmap_inc_factor"],
        downsample_factors=params["downsample_factors"],
        kernel_size=params.get("kernel_size", 3),
        norm_type=params.get("norm_type", "group"),
    )