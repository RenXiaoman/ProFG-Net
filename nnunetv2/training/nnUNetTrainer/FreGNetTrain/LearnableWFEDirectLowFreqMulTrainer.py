from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableWFELowFreqVariantAuxTrainer import (
    LearnableWFELowFreqVariantAuxTrainer,
)


class LearnableWFEDirectLowFreqMulTrainer(LearnableWFELowFreqVariantAuxTrainer):
    """WFE with direct low-frequency multiplication, without channel/spatial attention."""

    lf_variant = "direct_lf_mul"

