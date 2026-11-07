from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableWFEVariantAuxTrainer import (
    LearnableWFEVariantAuxTrainer,
)


class LearnableWFESobelReplacementTrainer(LearnableWFEVariantAuxTrainer):
    """Replace the WFE high-frequency branch with a fixed 3-D Sobel response."""

    wfe_variant = "sobel_replacement"

