import torch
import torch.nn as nn
import torch.nn.functional as F


# Base Components & DALW
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x):
        return F.gelu(x + self.conv(x))


class DirectionAwareLearnableWavelet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.P_HL = self._make_pu_net(channels)
        self.P_LH = self._make_pu_net(channels)
        self.P_HH = self._make_pu_net(channels)
        self.U_HL = self._make_pu_net(channels)
        self.U_LH = self._make_pu_net(channels)
        self.U_HH = self._make_pu_net(channels)

    def _make_pu_net(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, x):
        c = x[:, :, 0::2, 0::2]
        x_HL = x[:, :, 0::2, 1::2]
        x_LH = x[:, :, 1::2, 0::2]
        x_HH = x[:, :, 1::2, 1::2]

        d_HL = x_HL - self.P_HL(c)
        d_LH = x_LH - self.P_LH(c)
        d_HH = x_HH - self.P_HH(c)
        s_LL = c + self.U_HL(d_HL) + self.U_LH(d_LH) + self.U_HH(d_HH)
        return s_LL, d_HL, d_LH, d_HH



# ==========================================
# Dynamics Core: Frequency-Coupled Liquid Cell
# ==========================================
class FrequencyCoupledLiquidCell(nn.Module):
    def __init__(self, dim, tau_base=0.5):
        super().__init__()
        self.tau_base = tau_base
        self.W_f = nn.Conv2d(dim, dim, 3, padding=1)
        self.W_in = nn.Conv2d(dim, dim, 1)
        self.W_tau = nn.Sequential(
            nn.Conv2d(dim + 1, dim, 3, padding=1),
            nn.Softplus()
        )
        self.drive_gate = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.Sigmoid())

    def forward(self, u_base, h_t, H_energy, dt):
        u_t = u_base * self.drive_gate(h_t)
        tau_input = torch.cat([h_t + self.W_in(u_t), H_energy], dim=1)
        tau = self.tau_base + self.W_tau(tau_input)

        # Optional: Store tau for analysis
        if self.training:
            self.last_tau_mean = tau.detach().mean().item()

        f_val = torch.tanh(self.W_f(h_t + self.W_in(u_t)))
        dh = (f_val - h_t) / (tau + 1e-6)
        return h_t + dt * dh


class SemanticLiquidEvolution(nn.Module):
    def __init__(self, dim, time_steps=6):
        super().__init__()
        self.time_steps = time_steps
        self.dt = 1.0 / time_steps
        self.cell = FrequencyCoupledLiquidCell(dim, tau_base=0.5)

    def forward(self, state_LL, drive_LL, H_energy):
        h = state_LL
        for t in range(self.time_steps):
            h = self.cell(drive_LL, h, H_energy, self.dt)
        return h


# ==========================================
# Directional Boundary Branch
# ==========================================
class DirectionalBoundaryBranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=3),
            nn.GroupNorm(3, dim * 3),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim, 1)
        )

    def forward(self, HL_pre, HL_post, LH_pre, LH_post, HH_pre, HH_post):
        diff_HL = torch.abs(HL_pre - HL_post)
        diff_LH = torch.abs(LH_pre - LH_post)
        diff_HH = torch.abs(HH_pre - HH_post)
        return self.boundary_conv(torch.cat([diff_HL, diff_LH, diff_HH], dim=1))


# ==========================================
# Auxiliary: Cross-Temporal Band Normalization
# ==========================================
def normalize_band(x):
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


# ==========================================
# Ultimate Network: Hierarchical FD-LDN
# ==========================================
class DFS_Net(nn.Module):
    def __init__(self, pre_ch=3, post_ch=3, base_dim=32, num_classes=1):
        super().__init__()

        # 1. Strong Stem
        self.stem_pre = nn.Sequential(
            nn.Conv2d(pre_ch, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
            ResBlock(base_dim)
        )
        self.stem_post = nn.Sequential(
            nn.Conv2d(post_ch, base_dim, 3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
            ResBlock(base_dim)
        )

        # 2. Stage 1 (DALW-1): 64 -> 32
        self.wavelet1 = DirectionAwareLearnableWavelet(base_dim)
        self.boundary_branch1 = DirectionalBoundaryBranch(base_dim)  # Output: 32

        # Deep Semantic Encoder: Maintain LL main track with detail injection
        self.semantic_enc1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 3, padding=1),
            ResBlock(base_dim * 2)
        )
        self.detail_proj1 = nn.Sequential(
            nn.Conv2d(base_dim * 3, base_dim * 2, 1),
            nn.GroupNorm(8, base_dim * 2),
            nn.GELU()
        )
        self.detail_gate1 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 1),
            nn.Sigmoid()
        )

        # 3. Stage 2 (DALW-2): 32 -> 16
        self.wavelet2 = DirectionAwareLearnableWavelet(base_dim * 2)
        self.boundary_branch2 = DirectionalBoundaryBranch(base_dim * 2)  # Output: 64

        # 4. Semantic Evolution (LNN) & State Readout
        self.semantic_evolution = SemanticLiquidEvolution(base_dim * 2, time_steps=6)
        self.state_readout = nn.Sequential(
            nn.Conv2d(base_dim * 6, base_dim * 2, 1),  # LL2_p, LL2_s, evolved (64*3=192) -> 64
            nn.GroupNorm(8, base_dim * 2),
            nn.GELU(),
            ResBlock(base_dim * 2)
        )

        # 5. Semantic Gating for Boundaries
        self.boundary_gate2 = nn.Sequential(nn.Conv2d(base_dim * 2, base_dim * 2, 1), nn.Sigmoid())
        self.boundary_gate1 = nn.Sequential(nn.Conv2d(base_dim * 2, base_dim, 1), nn.Sigmoid())

        # 6. Progressive Decoder
        self.fusion_lvl2 = nn.Sequential(
            nn.Conv2d(base_dim * 4, base_dim * 2, 3, padding=1),  # 64 + 64
            ResBlock(base_dim * 2)
        )
        self.up_lvl2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.fusion_lvl1 = nn.Sequential(
            nn.Conv2d(base_dim * 3, base_dim, 3, padding=1),  # up2(64) + gated_b1(32)
            ResBlock(base_dim)
        )
        self.up_lvl1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1),  # up1(32) + skip_diff(32)
            nn.GELU(),
            nn.Conv2d(base_dim, num_classes, 1)
        )

    def forward(self, img_pre, img_post):
        # --- Pre-processing ---
        feat_pre = self.stem_pre(img_pre)
        feat_post = self.stem_post(img_post)
        skip_diff_0 = torch.abs(feat_pre - feat_post)

        # --- Stage 1: DALW-1 ---
        LL1_p, HL1_p, LH1_p, HH1_p = self.wavelet1(feat_pre)
        LL1_s, HL1_s, LH1_s, HH1_s = self.wavelet1(feat_post)

        boundary_detail_1 = self.boundary_branch1(HL1_p, HL1_s, LH1_p, LH1_s, HH1_p, HH1_s)

        # Residual Detail Injection into Semantic Main Branch
        detail1_pre = self.detail_proj1(torch.cat([HL1_p, LH1_p, HH1_p], dim=1))
        detail1_post = self.detail_proj1(torch.cat([HL1_s, LH1_s, HH1_s], dim=1))
        f1_pre = self.semantic_enc1(LL1_p) + self.detail_gate1(LL1_p) * detail1_pre
        f1_post = self.semantic_enc1(LL1_s) + self.detail_gate1(LL1_s) * detail1_post

        # --- Stage 2: DALW-2 ---
        LL2_p, HL2_p, LH2_p, HH2_p = self.wavelet2(f1_pre)
        LL2_s, HL2_s, LH2_s, HH2_s = self.wavelet2(f1_post)

        boundary_detail_2 = self.boundary_branch2(HL2_p, HL2_s, LH2_p, LH2_s, HH2_p, HH2_s)

        # --- Cross-Temporal High-Freq Prior ---
        d_hl = torch.abs(normalize_band(HL2_p) - normalize_band(HL2_s))
        d_lh = torch.abs(normalize_band(LH2_p) - normalize_band(LH2_s))
        d_hh = torch.abs(normalize_band(HH2_p) - normalize_band(HH2_s))
        H_energy_spatial = torch.sqrt(d_hl.pow(2) + d_lh.pow(2) + d_hh.pow(2) + 1e-6).mean(dim=1, keepdim=True)

        # --- LNN Evolution & State Readout ---
        evolved_semantic_2 = self.semantic_evolution(state_LL=LL2_p, drive_LL=LL2_s, H_energy=H_energy_spatial)
        semantic_change_2 = self.state_readout(torch.cat([LL2_p, LL2_s, evolved_semantic_2], dim=1))

        # --- Semantic Gating for Boundaries ---
        boundary_detail_2 = boundary_detail_2 * self.boundary_gate2(semantic_change_2)

        # --- Progressive Decoder ---
        fused_lvl2 = self.fusion_lvl2(torch.cat([semantic_change_2, boundary_detail_2], dim=1))
        up2 = self.up_lvl2(fused_lvl2)

        boundary_detail_1 = boundary_detail_1 * self.boundary_gate1(up2)  # Gated by deep semantics

        fused_lvl1 = self.fusion_lvl1(torch.cat([up2, boundary_detail_1], dim=1))
        up1 = self.up_lvl1(fused_lvl1)

        logits = self.final_conv(torch.cat([up1, skip_diff_0], dim=1))

        return logits, None

