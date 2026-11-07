from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableWFEVariantAuxTrainer import (
    LearnableWFEVariantAuxTrainer,
)


class LearnableWFELaplacianResidualTrainer(LearnableWFEVariantAuxTrainer):
    """Residual enhancement with a learnable alpha on the 3-D Laplacian response."""

    wfe_variant = "laplacian_residual"

