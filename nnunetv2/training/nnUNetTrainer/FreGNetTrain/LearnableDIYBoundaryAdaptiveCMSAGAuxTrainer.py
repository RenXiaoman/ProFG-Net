from __future__ import annotations

from torch import nn

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryAdaptiveCMSAGAuxTrainer import (
    DIYBoundaryAdaptiveCMSAGAuxTrainer,
    replace_cmsag_w_x_with_adaptive_block,
)
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag_learnable import (
    LearnableDIYBoundaryCMSAG,
)


class LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer(DIYBoundaryAdaptiveCMSAGAuxTrainer):
    """AdaptiveCMSAG trainer with LWTformer-style learnable wavelet filters.

    Same adaptive W_x + boundary auxiliary supervision as
    ``DIYBoundaryAdaptiveCMSAGAuxTrainer``, except the ptwt-based wavelet
    decomposition/reconstruction inside every WFE block is replaced by a
    globally shared learnable separable convolution wavelet bank.
    """

    wavelet = "db1"
    boundary_weight = 0.2
    boundary_radius = 1
    num_epochs = 300

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = LearnableDIYBoundaryCMSAG(
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
