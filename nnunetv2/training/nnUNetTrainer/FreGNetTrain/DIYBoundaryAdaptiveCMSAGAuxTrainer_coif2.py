from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.DIYBoundaryAdaptiveCMSAGAuxTrainer import (
    DIYBoundaryAdaptiveCMSAGAuxTrainer,
)


class DIYBoundaryAdaptiveCMSAGAuxTrainer_coif2(DIYBoundaryAdaptiveCMSAGAuxTrainer):
    """Adaptive CMSAG with the coif2 wavelet (Table 5 wavelet ablation)."""

    wavelet = "coif2"

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
        self.num_epochs = 70
        self.initial_lr = 1e-3
