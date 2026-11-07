from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryCMSAGAuxTrainer import (
    DIYBoundaryCMSAGAuxTrainer,
)
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag import (
    DIYBoundaryCMSAG,
)
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.components import (
    AdaptiveComplementaryMultiScaleDepthwiseBlock,
    ComplementaryMultiScaleAttentionGate,
)


def replace_cmsag_w_x_with_adaptive_block(model: nn.Module) -> int:
    """Swap W_x of every CMSAG gate for the adaptive depthwise block.

    The adaptive block reuses the same submodule names as
    ComplementaryMultiScaleDepthwiseBlock (projection, branches, fusion), so a
    checkpoint trained with DIYBoundaryCMSAGAuxTrainer loads into this model
    with only the per-branch alphas missing.

    Scanning by module type keeps this valid for every encoder depth: depth 3
    uses decoder_depth3_*, depth 4 and 5 use decoder5..decoder2. Returns the
    number of gates that were replaced.
    """
    gates = [module for module in model.modules() if isinstance(module, ComplementaryMultiScaleAttentionGate)]
    for gate in gates:
        original = gate.W_x
        adaptive = AdaptiveComplementaryMultiScaleDepthwiseBlock(
            spatial_dims=3,
            in_channels=original.projection[0].in_channels,
            out_channels=original.projection[0].out_channels,
            dropout=0.0,
        )
        adaptive.load_state_dict(original.state_dict(), strict=False)
        gate.W_x = adaptive
    return len(gates)


class DIYBoundaryAdaptiveCMSAGAuxTrainer(DIYBoundaryCMSAGAuxTrainer):
    """WFE + CMSAG with an adaptive W_x + boundary auxiliary supervision.

    Identical to DIYBoundaryCMSAGAuxTrainer except that the W_x of every CMSAG
    gate is AdaptiveComplementaryMultiScaleDepthwiseBlock, whose per-branch
    residual scalars alpha are initialized to 0. At initialization the network
    is therefore numerically equivalent to the non-adaptive CMSAG model.

    Training hyperparameters are inherited unchanged:
      boundary_weight = 0.2
      boundary_radius = 1
      num_epochs = 120
      initial_lr = 1e-3
      enable_deep_supervision = False

    Subclasses only need to override the ``wavelet`` class attribute to build a
    wavelet-ablation variant that keeps the adaptive W_x.
    """

    wavelet = "db1"

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = DIYBoundaryCMSAG(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            spatial_dims=3,
            feature_size=16,
            norm_name="instance",
            dropout_rate=0.0,
            depth=4,
            boundary_channels=max(1, num_output_channels - 1),
            wavelet=cls.wavelet,
            mode="zero",
        )
        replaced = replace_cmsag_w_x_with_adaptive_block(model)
        if replaced != 4:
            raise RuntimeError(f"Expected 4 CMSAG gates at depth 4, replaced {replaced}")
        return model
