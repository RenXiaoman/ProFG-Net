from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryAdaptiveCMSAGAuxTrainer import (
    DIYBoundaryAdaptiveCMSAGAuxTrainer,
    replace_cmsag_w_x_with_adaptive_block,
)
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag import (
    DIYBoundaryCMSAG,
)


class DIYBoundaryAdaptiveCMSAGAuxTrainerDepth3(DIYBoundaryAdaptiveCMSAGAuxTrainer):
    """Depth-3 ablation of WFE + adaptive CMSAG + boundary auxiliary supervision.

    Same as DIYBoundaryCMSAGAuxTrainerDepth3 except that the W_x of the three
    CMSAG gates (decoder_depth3_3/2/1) is AdaptiveComplementaryMultiScaleDepthwiseBlock.
    Training hyperparameters follow the non-adaptive depth-3 ablation
    (num_epochs = 80, initial_lr = 1e-3).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 80
        self.initial_lr = 1e-3

    @staticmethod
    def build_network_architecture(
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
            depth=3,
            boundary_channels=max(1, num_output_channels - 1),
        )
        replaced = replace_cmsag_w_x_with_adaptive_block(model)
        if replaced != 3:
            raise RuntimeError(f"Expected 3 CMSAG gates at depth 3, replaced {replaced}")
        return model
