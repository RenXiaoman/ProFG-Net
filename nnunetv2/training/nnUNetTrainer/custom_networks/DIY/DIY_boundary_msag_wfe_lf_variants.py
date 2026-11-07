from __future__ import annotations

import torch
import torch.nn as nn

from .DIY_boundary_msag import DIYBoundaryMSAG, DIYMultiScaleGate
from .components import (
    ChannelAttention,
    ComplementaryMultiScaleAttentionGate,
    LearnableGaussianFilterBank3D,
    MultiScaleAttentionGate,
    SpatialAttention,
)
from .diy import LearnableWaveletBank


WFE_LF_VARIANTS = (
    "ours",
    "no_lf_guidance",
    "direct_lf_mul",
)


class WFELowFreqVariantWavEnhance(nn.Module):
    """Learnable WFE with a configurable low-frequency guidance variant."""

    def __init__(
        self,
        wavelet_bank: LearnableWaveletBank,
        in_dim: int,
        lf_variant: str,
    ) -> None:
        super().__init__()
        if lf_variant not in WFE_LF_VARIANTS:
            raise ValueError(f"Unsupported WFE low-frequency variant: {lf_variant}")

        self.lf_variant = lf_variant
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
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.last_attended_low: torch.Tensor | None = None
        self.last_raw_high: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            b, c, d, h, w = x.shape
            dwt_out = self.wavelet_bank.dwt(x)

            ll = dwt_out[:, :c, :, :, :]
            yh = dwt_out[:, c:, :, :, :]

            gaussian_filter = self.gaussian_filter_2d_depth if d == 1 else self.gaussian_filter_3d
            self.last_raw_high = yh.detach()
            yh_low = gaussian_filter(yh)
            yh_high = yh - yh_low
            yh = yh + self.alpha * yh_high

            subbands = 3 if d == 1 else 7
            if self.lf_variant == "ours":
                ca_mask = self.ca(ll)
                sa_mask = self.sa(ca_mask * ll)
                att = ca_mask * sa_mask
                self.last_attended_low = (ll * att).detach()
                yh = yh * att.repeat_interleave(subbands, dim=1)
            elif self.lf_variant == "direct_lf_mul":
                yh = yh * torch.sigmoid(ll).repeat_interleave(subbands, dim=1)
            elif self.lf_variant == "no_lf_guidance":
                pass
            else:
                raise RuntimeError(f"Unhandled WFE low-frequency variant: {self.lf_variant}")

            x_rec = self.wavelet_bank.idwt(ll, yh)
        return x_rec.to(orig_dtype)


class WFELowFreqVariantDIYMultiScaleGate(DIYMultiScaleGate):
    """DIYMultiScaleGate with learnable wavelet filters and LF-variant WFE."""

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
        lf_variant: str = "ours",
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
                WFELowFreqVariantWavEnhance(
                    wavelet_bank=self.wavelet_bank,
                    in_dim=feature_size * (2 ** i),
                    lf_variant=lf_variant,
                )
                for i in range(depth)
            ]
        )


class WFELowFreqVariantDIYBoundaryMSAG(DIYBoundaryMSAG):
    """Boundary/MSAG network whose WFE uses a selected LF guidance variant."""

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
        lf_variant: str = "ours",
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

        new_backbone = WFELowFreqVariantDIYMultiScaleGate(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
            feature_size=feature_size,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            depth=depth,
            wavelet=wavelet,
            mode=mode,
            lf_variant=lf_variant,
            attention_gate_cls=attention_gate_cls,
        )
        self.backbone = new_backbone
        self.seg_head = new_backbone.out
        new_backbone.out = nn.Identity()


class WFELowFreqVariantDIYBoundaryCMSAG(WFELowFreqVariantDIYBoundaryMSAG):
    """CMSAG version of WFELowFreqVariantDIYBoundaryMSAG."""

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
        lf_variant: str = "ours",
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
            lf_variant=lf_variant,
            attention_gate_cls=ComplementaryMultiScaleAttentionGate,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WFELowFreqVariantDIYBoundaryCMSAG(
        in_channels=1,
        out_channels=2,
        feature_size=16,
        depth=4,
        boundary_channels=1,
        lf_variant="direct_lf_mul",
    ).to(device)
    x = torch.randn(1, 1, 16, 64, 64).to(device)
    seg, boundary = model(x, training=True)
    print(seg.shape, boundary.shape)
