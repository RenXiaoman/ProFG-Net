from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableWFEVariantAuxTrainer import (
    LearnableWFEVariantAuxTrainer,
)


class LearnableWFEGaussianHighPassReplacementTrainer(LearnableWFEVariantAuxTrainer):
    """Use only the Gaussian high-pass residual as the high-frequency branch."""

    wfe_variant = "gaussian_highpass_replacement"

