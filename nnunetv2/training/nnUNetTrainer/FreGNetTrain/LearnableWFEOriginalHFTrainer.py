from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableWFEVariantAuxTrainer import (
    LearnableWFEVariantAuxTrainer,
)


class LearnableWFEOriginalHFTrainer(LearnableWFEVariantAuxTrainer):
    """Original high-frequency branch: identity enhancement."""

    wfe_variant = "original_hf"

