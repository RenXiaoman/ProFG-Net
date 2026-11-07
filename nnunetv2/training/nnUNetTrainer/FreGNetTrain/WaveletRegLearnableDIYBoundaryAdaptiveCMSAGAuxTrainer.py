from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.loss.wavelet_regularization_loss import WaveletRegularizationLoss
from nnunetv2.training.nnUNetTrainer.FreGNetTrain.LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer import (
    LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer,
)
from nnunetv2.utilities.helpers import dummy_context


class WaveletRegLearnableDIYBoundaryAdaptiveCMSAGAuxTrainer(
    LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer
):
    """Learnable-AdaptiveCMSAG plus the wavelet-regularization loss.

    This is the "测试模型": the current newest model
    (LearnableDIYBoundaryAdaptiveCMSAGAuxTrainer) with the same architecture,
    data pipeline, boundary auxiliary task, and training hyperparameters. The
    only difference is that every training step adds a soft wavelet constraint
    to the segmentation loss.

    The added loss keeps the four learnable wavelet filter vectors:
      - high-pass filters zero-mean (no DC leakage)
      - low-pass and high-pass filters orthogonal
    It does not add a perfect-reconstruction term.
    """

    # Small defaults; these are intended to be tuned.
    wavelet_reg_dc_weight = 0.01
    wavelet_reg_orth_weight = 0.01
    wavelet_reg_eps = 1e-6

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.wavelet_reg_loss = WaveletRegularizationLoss(
            weight_dc=self.wavelet_reg_dc_weight,
            weight_orth=self.wavelet_reg_orth_weight,
            eps=self.wavelet_reg_eps,
            apply_to_synthesis=True,
        )
        self.num_epochs = 400

    def _get_plain_network(self) -> nn.Module:
        network = self.network
        if hasattr(network, "module"):
            network = network.module
        if hasattr(network, "_orig_mod"):
            network = network._orig_mod
        return network

    def _wavelet_regularization_loss(self) -> torch.Tensor:
        bank = self._get_plain_network().backbone.wavelet_bank
        return self.wavelet_reg_loss(
            dec_lo=bank.dwt.dec_lo,
            dec_hi=bank.dwt.dec_hi,
            rec_lo=bank.idwt.rec_lo,
            rec_hi=bank.idwt.rec_hi,
        )

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target_for_boundary = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            target_for_boundary = target

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            seg_logits, boundary_logits = self.network(data, training=True)
            boundary_target = self.make_3d_boundary(target_for_boundary, boundary_logits.shape[1])
            seg_loss = self.loss(seg_logits, target)
            boundary_loss = self.boundary_dice_loss(boundary_logits, boundary_target)
            boundary_loss = boundary_loss + F.binary_cross_entropy_with_logits(boundary_logits, boundary_target)
            total_loss = seg_loss + self.boundary_weight * boundary_loss

        reg_loss = self._wavelet_regularization_loss()
        total_loss = total_loss + reg_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": total_loss.detach().cpu().numpy()}


if __name__ == "__main__":
    # Smoke test: build the network like nnU-Net would, run a small forward
    # pass, and verify that the db1-initialized filter bank has zero wavelet
    # regularization while a drifted filter bank is penalized.
    model = WaveletRegLearnableDIYBoundaryAdaptiveCMSAGAuxTrainer.build_network_architecture(
        None, None, num_input_channels=1, num_output_channels=2, enable_deep_supervision=False
    )
    x = torch.randn(1, 1, 16, 32, 32)
    with torch.no_grad():
        seg, boundary = model(x, training=True)
    print(f"seg shape: {tuple(seg.shape)}, boundary shape: {tuple(boundary.shape)}")

    reg_loss = WaveletRegularizationLoss()
    bank = model.backbone.wavelet_bank
    initial = reg_loss(
        bank.dwt.dec_lo,
        bank.dwt.dec_hi,
        bank.idwt.rec_lo,
        bank.idwt.rec_hi,
    )
    print(f"db1-initialized regularization: {initial.item():.8f}")

    bank.dwt.dec_lo.data.copy_(torch.tensor([[0.50614077, 0.55333656]]))
    bank.dwt.dec_hi.data.copy_(torch.tensor([[-1.09529567, 0.31578246]]))
    bank.idwt.rec_lo.data.copy_(torch.tensor([[0.87311327, 0.37176257]]))
    bank.idwt.rec_hi.data.copy_(torch.tensor([[0.75666529, -0.84625649]]))
    drifted = reg_loss(
        bank.dwt.dec_lo,
        bank.dwt.dec_hi,
        bank.idwt.rec_lo,
        bank.idwt.rec_hi,
    )
    print(f"drifted-filter regularization: {drifted.item():.8f}")
