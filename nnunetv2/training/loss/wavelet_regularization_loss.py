from typing import Optional

import torch
from torch import nn


class WaveletRegularizationLoss(nn.Module):
    """Soft structural constraints for the learnable wavelet filter bank.

    This is an auxiliary loss, not a segmentation loss. It takes the raw
    1-D filter vectors from ``LearnableConvDWT`` and ``LearnableConvIDWT``
    and encourages two properties:

    1. High-pass filters are zero-mean (no DC leakage):
       ``(sum(g))^2 / (||g||^2 + eps)``
    2. Low-pass and high-pass filters are orthogonal:
       ``(h^T g)^2 / ((||h||^2 + eps) * (||g||^2 + eps))``

    Both terms are normalized, so they are invariant to an overall gain of
    the filters. The first term therefore prevents the model from cheating
    by shrinking the high-pass filter towards zero. The second term is the
    squared cosine similarity between ``h`` and ``g``.

    By default both analysis and synthesis filter pairs are constrained.
    """

    def __init__(
        self,
        weight_dc: float = 1.0,
        weight_orth: float = 1.0,
        eps: float = 1e-8,
        apply_to_synthesis: bool = True,
    ):
        super().__init__()
        self.weight_dc = weight_dc
        self.weight_orth = weight_orth
        self.eps = eps
        self.apply_to_synthesis = apply_to_synthesis

    def _as_vector(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1).float()
        if x.numel() == 0:
            raise ValueError("Wavelet filter tensors must not be empty.")
        return x

    def _zero_mean_loss(self, g: torch.Tensor) -> torch.Tensor:
        g = self._as_vector(g)
        numerator = g.sum().square()
        denominator = g.square().sum() + self.eps
        return numerator / denominator

    def _orthogonality_loss(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = self._as_vector(h)
        g = self._as_vector(g)
        numerator = (h * g).sum().square()
        denominator = (h.square().sum() + self.eps) * (g.square().sum() + self.eps)
        return numerator / denominator

    def forward(
        self,
        dec_lo: torch.Tensor,
        dec_hi: torch.Tensor,
        rec_lo: Optional[torch.Tensor] = None,
        rec_hi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the scalar wavelet-regularization loss.

        Args:
            dec_lo: Analysis low-pass filter, shape ``[1, K]`` or ``[K]``.
            dec_hi: Analysis high-pass filter, shape ``[1, K]`` or ``[K]``.
            rec_lo: Synthesis low-pass filter. Required when
                ``apply_to_synthesis`` is ``True``.
            rec_hi: Synthesis high-pass filter. Required when
                ``apply_to_synthesis`` is ``True``.
        """
        if self.apply_to_synthesis and (rec_lo is None or rec_hi is None):
            raise ValueError(
                "apply_to_synthesis=True requires rec_lo and rec_hi."
            )

        orth = self._orthogonality_loss(dec_lo, dec_hi)
        dc = self._zero_mean_loss(dec_hi)

        if self.apply_to_synthesis:
            orth = orth + self._orthogonality_loss(rec_lo, rec_hi)
            dc = dc + self._zero_mean_loss(rec_hi)

        return self.weight_orth * orth + self.weight_dc * dc


if __name__ == "__main__":
    # db1-initialized filters should have zero orthogonality and DC leakage.
    loss_fn = WaveletRegularizationLoss()
    dec_lo = torch.tensor([[0.70710678, 0.70710678]])
    dec_hi = torch.tensor([[-0.70710678, 0.70710678]])
    rec_lo = torch.tensor([[0.70710678, 0.70710678]])
    rec_hi = torch.tensor([[0.70710678, -0.70710678]])

    initial = loss_fn(dec_lo, dec_hi, rec_lo, rec_hi)
    print(f"db1-initialized loss: {initial.item():.8f}")

    # Trained filters that drift away from the wavelet property should be
    # penalized by the same loss.
    trained_lo = torch.tensor([[0.50614077, 0.55333656]])
    trained_hi = torch.tensor([[-1.09529567, 0.31578246]])
    trained_rec_lo = torch.tensor([[0.87311327, 0.37176257]])
    trained_rec_hi = torch.tensor([[0.75666529, -0.84625649]])
    trained = loss_fn(trained_lo, trained_hi, trained_rec_lo, trained_rec_hi)
    print(f"trained-filter loss:   {trained.item():.8f}")
