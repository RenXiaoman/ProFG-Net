"""Ablation backbone with none of the three FreGNet modules.

Removed relative to DIYBoundaryCMSAG:
  1. WFE (wav_Enhance / learnable WFE)  -> identity, so every encoder stage is
     only its ResidualUnit.
  2. MSAG / CMSAG attention gate        -> identity, the decoder simply
     concatenates the upsampled feature with the encoder skip.
  3. Boundary auxiliary head            -> removed, only the segmentation head
     remains.

The remaining architecture is stem -> 4 ResidualUnit encoders -> 4 plain
transposed-conv decoders -> segmentation head, i.e. the matched baseline of
Table "Ablation study" (no WFE, no MSAG/CMSAG, no boundary supervision).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers.factories import Norm


class PlainUnetrUpBlock(nn.Module):
    """Decoder block without any attention gate.

    Same shapes as MultiScaleGatedUnetrUpBlock, but the encoder skip is used
    directly instead of being reweighted by MSAG/CMSAG.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        super().__init__()
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )
        block = UnetResBlock if res_block else UnetBasicBlock
        self.conv_block = block(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class BaselineBackbone(nn.Module):
    """No-WFE / no-attention-gate / no-boundary-head segmentation backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth != 4:
            raise ValueError(f"BaselineBackbone currently supports depth=4 only, got {depth}.")
        self.depth = depth

        self.stem = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        def encoder(in_channels: int, out_channels: int) -> ResidualUnit:
            return ResidualUnit(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                kernel_size=3,
                adn_ordering="NDA",
                act=("leakyrelu", {"negative_slope": 0.2}),
                norm=Norm.BATCH,
                dropout=dropout_rate,
                dropout_dim=1,
            )

        # WFE removed: each encoder stage is only its ResidualUnit.
        self.enc1 = encoder(feature_size, 2 * feature_size)
        self.enc2 = encoder(2 * feature_size, 4 * feature_size)
        self.enc3 = encoder(4 * feature_size, 8 * feature_size)
        self.enc4 = encoder(8 * feature_size, 16 * feature_size)

        # Attention gate removed: plain transposed conv + concat + conv block.
        self.decoder5 = PlainUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = PlainUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = PlainUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = PlainUnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        up4 = self.decoder5(x5, x4)
        up3 = self.decoder4(up4, x3)
        up2 = self.decoder3(up3, x2)
        up1 = self.decoder2(up2, x1)
        return self.out(up1)


if __name__ == "__main__":
    from thop import profile

    model = BaselineBackbone(
        in_channels=1,
        out_channels=2,
        spatial_dims=3,
        feature_size=16,
        norm_name="instance",
        dropout_rate=0.0,
        depth=4,
    ).eval()

    dummy = torch.randn(1, 1, 16, 256, 256)
    with torch.no_grad():
        output = model(dummy)
        macs, thop_params = profile(model, inputs=(dummy,), verbose=False)

    params = sum(parameter.numel() for parameter in model.parameters())
    print(f"input : {tuple(dummy.shape)}")
    print(f"output: {tuple(output.shape)}")
    print(f"Params      : {params:,} ({params / 1e6:.4f} M)")
    print(f"thop params : {int(thop_params):,}")
    print(f"MACs (thop) : {macs / 1e9:.4f} G")
