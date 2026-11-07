from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer import (
    LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer,
)


class LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer_2(LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer):
    """Learnable Adaptive CMSAG boundary-radius sweep, radius 2, 70 epochs."""

    boundary_radius = 2

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

