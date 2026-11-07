from __future__ import annotations

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryAdaptiveCMSAGAuxTrainer import (
    replace_cmsag_w_x_with_adaptive_block,
)
from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer import (
    LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer,
)
from nnunetv2.training.nnUNetTrainer.custom_networks.DIY.DIY_boundary_msag_wfe_variants import (
    WFEVariantDIYBoundaryCMSAG,
)


class LearnableWFEVariantAuxTrainer(LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer):
    """Base trainer for the six learnable-WFE high-frequency ablation variants."""

    wfe_variant = None

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
        self.num_epochs = 100
        self.initial_lr = 1e-3

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        model = WFEVariantDIYBoundaryCMSAG(
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
            wfe_variant=cls.wfe_variant,
        )
        replaced = replace_cmsag_w_x_with_adaptive_block(model)
        if replaced != 4:
            raise RuntimeError(f"Expected 4 CMSAG gates at depth 4, replaced {replaced}")
        return model
