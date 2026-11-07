from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableWFELowFreqVariantAuxTrainer import (
    LearnableWFELowFreqVariantAuxTrainer,
)


class LearnableWFENoLowFreqGuidanceTrainer(LearnableWFELowFreqVariantAuxTrainer):
    """WFE without low-frequency attention guidance to high-frequency subbands."""

    lf_variant = "no_lf_guidance"

