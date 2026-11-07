from __future__ import annotations

import torch.nn as nn

from monai.networks.blocks import UnetOutBlock

from .DIY_boundary_msag import DIYBoundaryMSAG, DIYMultiScaleGate
from .components import ComplementaryMultiScaleAttentionGate, MultiScaleAttentionGate
from .diy import LearnableWavEnhance, LearnableWaveletBank


class LearnableDIYMultiScaleGate(DIYMultiScaleGate):
    """DIYMultiScaleGate with one globally shared learnable wavelet bank."""

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
                LearnableWavEnhance(
                    wavelet_bank=self.wavelet_bank,
                    in_dim=feature_size * (2 ** i),
                )
                for i in range(depth)
            ]
        )


class LearnableDIYBoundaryMSAG(DIYBoundaryMSAG):
    """Boundary/MSAG network whose WFE uses learnable convolution wavelets."""

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

        new_backbone = LearnableDIYMultiScaleGate(
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
        self.backbone = new_backbone
        self.seg_head = new_backbone.out
        new_backbone.out = nn.Identity()


class LearnableDIYBoundaryCMSAG(LearnableDIYBoundaryMSAG):
    """Learnable-wavelet version of DIYBoundaryCMSAG."""

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
            attention_gate_cls=ComplementaryMultiScaleAttentionGate,
        )


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LearnableDIYBoundaryCMSAG(
        in_channels=1,
        out_channels=2,
        feature_size=16,
        depth=4,
        boundary_channels=1,
    ).to(device)
    x = torch.randn(1, 1, 16, 64, 64).to(device)
    seg, boundary = model(x, training=True)
    print(seg.shape, boundary.shape)
