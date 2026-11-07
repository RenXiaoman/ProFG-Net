from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DIY_boundary_msag import DIYBoundaryMSAG, DIYMultiScaleGate
from .components import (
    ChannelAttention,
    ComplementaryMultiScaleAttentionGate,
    LearnableGaussianFilterBank3D,
    MultiScaleAttentionGate,
    SpatialAttention,
)
from .diy import LearnableWaveletBank


WFE_VARIANTS = (
    "original_hf",
    "sobel_replacement",
    "laplacian_replacement",
    "gaussian_highpass_replacement",
    "sobel_residual",
    "laplacian_residual",
)


def _make_kernel_3d(values: dict[tuple[int, int, int], float]) -> torch.Tensor:
    kernel = torch.zeros(3, 3, 3, dtype=torch.float32)
    for (z, y, x), value in values.items():
        kernel[z, y, x] = value
    return kernel.view(1, 1, 3, 3, 3)


def _make_kernel_2d(values: dict[tuple[int, int], float]) -> torch.Tensor:
    kernel = torch.zeros(3, 3, dtype=torch.float32)
    for (y, x), value in values.items():
        kernel[y, x] = value
    return kernel.view(1, 1, 3, 3)


class FixedSobel3D(nn.Module):
    """Fixed 3-D Sobel operator returning the per-channel gradient magnitude."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "kd",
            _make_kernel_3d({(0, 1, 1): -1.0, (2, 1, 1): 1.0}),
        )
        self.register_buffer(
            "kh",
            _make_kernel_3d({(1, 0, 1): -1.0, (1, 2, 1): 1.0}),
        )
        self.register_buffer(
            "kw",
            _make_kernel_3d({(1, 1, 0): -1.0, (1, 1, 2): 1.0}),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        gd = F.conv3d(x, self.kd.repeat(c, 1, 1, 1, 1), padding=1, groups=c)
        gh = F.conv3d(x, self.kh.repeat(c, 1, 1, 1, 1), padding=1, groups=c)
        gw = F.conv3d(x, self.kw.repeat(c, 1, 1, 1, 1), padding=1, groups=c)
        return torch.sqrt(gd.square() + gh.square() + gw.square() + 1e-6)


class FixedSobel2D(nn.Module):
    """Fixed 2-D Sobel operator returning the per-channel gradient magnitude."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("kh", _make_kernel_2d({(0, 1): -1.0, (2, 1): 1.0}))
        self.register_buffer("kw", _make_kernel_2d({(1, 0): -1.0, (1, 2): 1.0}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        gh = F.conv2d(x, self.kh.repeat(c, 1, 1, 1), padding=1, groups=c)
        gw = F.conv2d(x, self.kw.repeat(c, 1, 1, 1), padding=1, groups=c)
        return torch.sqrt(gh.square() + gw.square() + 1e-6)


class FixedLaplacian3D(nn.Module):
    """Fixed 6-neighbor 3-D Laplacian operator."""

    def __init__(self) -> None:
        super().__init__()
        values = {
            (1, 1, 1): -6.0,
            (0, 1, 1): 1.0,
            (2, 1, 1): 1.0,
            (1, 0, 1): 1.0,
            (1, 2, 1): 1.0,
            (1, 1, 0): 1.0,
            (1, 1, 2): 1.0,
        }
        self.register_buffer("kernel", _make_kernel_3d(values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        return F.conv3d(x, self.kernel.repeat(c, 1, 1, 1, 1), padding=1, groups=c)


class FixedLaplacian2D(nn.Module):
    """Fixed 4-neighbor 2-D Laplacian operator."""

    def __init__(self) -> None:
        super().__init__()
        values = {
            (1, 1): -4.0,
            (0, 1): 1.0,
            (2, 1): 1.0,
            (1, 0): 1.0,
            (1, 2): 1.0,
        }
        self.register_buffer("kernel", _make_kernel_2d(values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        return F.conv2d(x, self.kernel.repeat(c, 1, 1, 1), padding=1, groups=c)


class WFEVariantWavEnhance(nn.Module):
    """Learnable WFE with a configurable high-frequency enhancement variant."""

    def __init__(
        self,
        wavelet_bank: LearnableWaveletBank,
        in_dim: int,
        variant: str,
    ) -> None:
        super().__init__()
        if variant not in WFE_VARIANTS:
            raise ValueError(f"Unsupported WFE variant: {variant}")

        self.variant = variant
        self.in_dim = in_dim
        object.__setattr__(self, "wavelet_bank", wavelet_bank)

        self.ca = ChannelAttention(in_dim)
        self.sa = SpatialAttention()
        self.gaussian_filter_2d_depth = LearnableGaussianFilterBank3D(
            kernel_size=3,
            num_filters=1,
            num_channels=3 * in_dim,
        )
        self.gaussian_filter_3d = LearnableGaussianFilterBank3D(
            kernel_size=3,
            num_filters=1,
            num_channels=7 * in_dim,
        )
        self.sobel2d = FixedSobel2D()
        self.sobel3d = FixedSobel3D()
        self.laplacian2d = FixedLaplacian2D()
        self.laplacian3d = FixedLaplacian3D()

        if variant in ("sobel_residual", "laplacian_residual"):
            self.alpha = nn.Parameter(torch.tensor(0.5))

        self.last_attended_low: torch.Tensor | None = None
        self.last_raw_high: torch.Tensor | None = None

    def _apply_highpass_variant(self, yh: torch.Tensor, d: int) -> torch.Tensor:
        if self.variant == "original_hf":
            return yh

        if self.variant == "gaussian_highpass_replacement":
            gaussian_filter = self.gaussian_filter_2d_depth if d == 1 else self.gaussian_filter_3d
            return yh - gaussian_filter(yh)

        if d == 1:
            sobel = self.sobel2d(yh)
            laplacian = self.laplacian2d(yh)
        else:
            sobel = self.sobel3d(yh)
            laplacian = self.laplacian3d(yh)

        if self.variant == "sobel_replacement":
            return sobel
        if self.variant == "laplacian_replacement":
            return laplacian
        if self.variant == "sobel_residual":
            return yh + self.alpha * sobel
        if self.variant == "laplacian_residual":
            return yh + self.alpha * laplacian

        raise RuntimeError(f"Unhandled WFE variant: {self.variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            b, c, d, h, w = x.shape
            dwt_out = self.wavelet_bank.dwt(x)

            ll = dwt_out[:, :c, :, :, :]
            yh = dwt_out[:, c:, :, :, :]

            ca_mask = self.ca(ll)
            sa_mask = self.sa(ca_mask * ll)
            att = ca_mask * sa_mask
            self.last_attended_low = (ll * att).detach()

            self.last_raw_high = yh.detach()
            yh = self._apply_highpass_variant(yh, d)

            subbands = 3 if d == 1 else 7
            yh = yh * att.repeat_interleave(subbands, dim=1)

            x_rec = self.wavelet_bank.idwt(ll, yh)
        return x_rec.to(orig_dtype)


class WFEVariantDIYMultiScaleGate(DIYMultiScaleGate):
    """DIYMultiScaleGate with learnable wavelet filters and WFE-variant blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
        wavelet: str = "db1",
        mode: str = "zero",
        wfe_variant: str = "original_hf",
        attention_gate_cls: type[nn.Module] = MultiScaleAttentionGate,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
            wavelet=wavelet,
            mode=mode,
            attention_gate_cls=attention_gate_cls,
        )

        self.wavelet_bank = LearnableWaveletBank(wavelet=wavelet, mode=mode)
        self.enhance = nn.ModuleList(
            [
                WFEVariantWavEnhance(
                    wavelet_bank=self.wavelet_bank,
                    in_dim=feature_size * (2 ** i),
                    variant=wfe_variant,
                )
                for i in range(depth)
            ]
        )


class WFEVariantDIYBoundaryMSAG(DIYBoundaryMSAG):
    """Boundary/MSAG network whose WFE uses a selected high-frequency variant."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
        boundary_channels: int = 1,
        wavelet: str = "db1",
        mode: str = "zero",
        wfe_variant: str = "original_hf",
        attention_gate_cls: type[nn.Module] = MultiScaleAttentionGate,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
            boundary_channels=boundary_channels,
            wavelet=wavelet,
            mode=mode,
            attention_gate_cls=attention_gate_cls,
        )

        new_backbone = WFEVariantDIYMultiScaleGate(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
            wavelet=wavelet,
            mode=mode,
            wfe_variant=wfe_variant,
            attention_gate_cls=attention_gate_cls,
        )
        self.backbone = new_backbone
        self.seg_head = new_backbone.out
        new_backbone.out = nn.Identity()


class WFEVariantDIYBoundaryCMSAG(WFEVariantDIYBoundaryMSAG):
    """CMSAG version of WFEVariantDIYBoundaryMSAG."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
        boundary_channels: int = 1,
        wavelet: str = "db1",
        mode: str = "zero",
        wfe_variant: str = "original_hf",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
            boundary_channels=boundary_channels,
            wavelet=wavelet,
            mode=mode,
            wfe_variant=wfe_variant,
            attention_gate_cls=ComplementaryMultiScaleAttentionGate,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WFEVariantDIYBoundaryCMSAG(
        in_channels=1,
        out_channels=2,
        feature_size=16,
        depth=4,
        boundary_channels=1,
        wfe_variant="sobel_residual",
    ).to(device)
    x = torch.randn(1, 1, 16, 64, 64).to(device)
    seg, boundary = model(x, training=True)
    print(seg.shape, boundary.shape)
