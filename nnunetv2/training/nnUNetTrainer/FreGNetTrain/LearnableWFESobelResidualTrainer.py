from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableWFEVariantAuxTrainer import (
    LearnableWFEVariantAuxTrainer,
)


class LearnableWFESobelResidualTrainer(LearnableWFEVariantAuxTrainer):
    """Residual enhancement with a learnable alpha on the 3-D Sobel response."""

    wfe_variant = "sobel_residual"

