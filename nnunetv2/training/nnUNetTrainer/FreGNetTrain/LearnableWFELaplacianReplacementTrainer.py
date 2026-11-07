from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableWFEVariantAuxTrainer import (
    LearnableWFEVariantAuxTrainer,
)


class LearnableWFELaplacianReplacementTrainer(LearnableWFEVariantAuxTrainer):
    """Replace the WFE high-frequency branch with a fixed 3-D Laplacian response."""

    wfe_variant = "laplacian_replacement"

