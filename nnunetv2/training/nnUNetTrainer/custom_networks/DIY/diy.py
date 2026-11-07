from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Norm

import pywt
import ptwt

try:
    from .components import SpatialAttention, ChannelAttention, LearnableGaussianFilterBank3D
except ImportError:
    from components import SpatialAttention, ChannelAttention, LearnableGaussianFilterBank3D


class AttentionGate(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 16,
        gating_channels: int = 16,
        inter_channels: int | None = None,
        norm_name: tuple | str = Norm.BATCH,
    ):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(in_channels // 2, 1)

        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=gating_channels,
                out_channels=inter_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True,
            ),
            Norm[norm_name, spatial_dims](inter_channels),
        )
        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=inter_channels,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True,
            ),
            Norm[norm_name, spatial_dims](inter_channels),
        )
        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=inter_channels,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                conv_only=True,
            ),
            Norm[norm_name, spatial_dims](1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # [B, 128, 2, 32, 32]
        att = self.relu(self.W_g(g) + self.W_x(x))  # [B, 64, 2, 32, 32] + [B, 64, 2, 32, 32]
        att = self.psi(att)  # [B, 1, 2, 32, 32]
        return x * att





class ConvDWT(torch.nn.Module):
    def __init__(self, wavelet='db1', level=1, mode='zero'):
        super(ConvDWT, self).__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        
        
    def forward(self, x):
        b, c, d, h, w = x.shape
        
        if d == 1:
            coeffs = ptwt.wavedec2(x.squeeze(2), wavelet=self.wavelet, level=self.level, mode=self.mode)
            Yl = coeffs[0].unsqueeze(2)
            Yh = torch.stack(coeffs[1], dim=1)
            Yh = Yh.reshape(b, -1, 1, Yh.shape[-2], Yh.shape[-1])
            return torch.cat((Yl, Yh), dim=1)  # [B, 4*C, 1, H/2, W/2]

        coeffs = ptwt.wavedec3(x, wavelet=self.wavelet, level=self.level, mode=self.mode)
        Yl  = coeffs[0]  # Yl (B, C, D/2, H/2, W/2) for low-frequency LL
        Yh = coeffs[1:]  # List Yh for high-frequency:aad, ada,add,daa,dad,dda,ddd
        
        band_order = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")
        Yh = torch.stack([Yh[0][band] for band in band_order], dim=1)  # [B, 7, C, D/2, H/2, W/2]
        Yh = Yh.reshape(Yh.shape[0], -1, Yh.shape[3], Yh.shape[4], Yh.shape[5])  # [B, 7*C, D/2, H/2, W/2]

        return torch.cat((Yl, Yh), dim=1)  # [B, 8*C, D/2, H/2, W/2]    


class ConvIDWT(nn.Module):
    def __init__(self, wavelet='db1', mode='zero'):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.band_order_3d = ("aad", "ada", "add", "daa", "dad", "dda", "ddd")

    def forward(self, low_freqs, high_freqs=None):
        if high_freqs is None:
            b, channels, d, h, w = low_freqs.shape
            if d == 1 and channels % 4 == 0:
                c = channels // 4
            elif channels % 8 == 0:
                c = channels // 8
            else:
                raise ValueError(f"Expected channels to be divisible by 4 or 8, got {channels}.")
            high_freqs = low_freqs[:, c:, :, :, :]
            low_freqs = low_freqs[:, :c, :, :, :]

        b, c, d, h, w = low_freqs.shape
        if high_freqs.shape[1] == 3 * c:
            expected_hf_channels = 3 * c
            if high_freqs.shape != (b, expected_hf_channels, d, h, w):
                raise ValueError(
                    f"Expected high_freqs shape {(b, expected_hf_channels, d, h, w)}, got {tuple(high_freqs.shape)}."
                )
            high_freqs = high_freqs.reshape(b, 3, c, h, w)
            detail = tuple(high_freqs[:, idx] for idx in range(3))
            return ptwt.waverec2((low_freqs.squeeze(2), detail), wavelet=self.wavelet).unsqueeze(2)

        expected_hf_channels = 7 * c
        if high_freqs.shape != (b, expected_hf_channels, d, h, w):
            raise ValueError(
                f"Expected high_freqs shape {(b, expected_hf_channels, d, h, w)}, got {tuple(high_freqs.shape)}."
            )

        high_freqs = high_freqs.reshape(b, 7, c, d, h, w)
        detail = {band: high_freqs[:, idx] for idx, band in enumerate(self.band_order_3d)}
        return ptwt.waverec3((low_freqs, detail), wavelet=self.wavelet)

class wav_Enhance(nn.Module): # Low-frequency Guided Feature Purification
    def __init__(self, wavelet='db1', mode='zero', in_dim=16):
        super().__init__()
        self.in_dim = in_dim
        
        
        self.dwt = ConvDWT(wavelet=wavelet, mode=mode)
        self.idwt = ConvIDWT(wavelet=wavelet, mode=mode)
        
        self.ca = ChannelAttention(in_dim)
        self.sa = SpatialAttention()
        self.gaussian_filter_2d_depth = LearnableGaussianFilterBank3D(
            kernel_size=3,
            num_filters=1,
            num_channels=3 * in_dim,
        )
        self.gaussian_filter_3d = LearnableGaussianFilterBank3D(
            kernel_size=3,
            num_filters=1,
            num_channels=7 * in_dim,
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # Populated during forward for optional feature visualization only.
        self.last_attended_low: torch.Tensor | None = None
        self.last_raw_high: torch.Tensor | None = None
        
    def forward(self, x):
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            B, C, D, H, W = x.shape
            dwt_out = self.dwt(x)  # [B, 8*C, D/2, H/2, W/2] or [B, 4*C, 1, H/2, W/2]
            
            LL = dwt_out[:, :C, :, :, :]
            Yh = dwt_out[:, C:, :, :, :]

            ca_mask = self.ca(LL)
            sa_mask = self.sa(ca_mask * LL)
            att = ca_mask * sa_mask
            self.last_attended_low = (LL * att).detach()
            
            
            ############ 高斯差分高通增强滤波 ############
            gaussian_filter = self.gaussian_filter_2d_depth if D == 1 else self.gaussian_filter_3d
            self.last_raw_high = Yh.detach()
            Yh_low = gaussian_filter(Yh)
            Yh_high = Yh - Yh_low
            Yh = Yh + self.alpha * Yh_high
            
            subbands = 3 if D == 1 else 7
            Yh = Yh * att.repeat_interleave(subbands, dim=1)

            
            x_rec = self.idwt(LL, Yh)
        return x_rec.to(orig_dtype)
        
class DIY(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
        depth: int = 4,
        wavelet: str = "db1",
        mode: str = "zero",
    ) -> None:
        super().__init__()
        if depth != 4:
            raise ValueError(f"DIY currently expects depth=4, got {depth}.")
        self.depth = depth
        
        self.stem = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.enhance = nn.ModuleList(
            [wav_Enhance(wavelet=wavelet, mode=mode, in_dim=feature_size * (2 ** i)) for i in range(depth)]
        )
        
        self.enc1 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        
        self.enc2 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=4 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        
        self.enc3 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=8 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        
        self.enc4 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=16 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.out = UnetOutBlock(spatial_dims=spatial_dims, 
                                in_channels=feature_size, 
                                out_channels=out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.enc1(self.enhance[0](x1))
        x3 = self.enc2(self.enhance[1](x2))
        x4 = self.enc3(self.enhance[2](x3))
        x5 = self.enc4(self.enhance[3](x4))

        up4 = self.decoder5(x5, x4)
        up3 = self.decoder4(up4, x3)
        up2 = self.decoder3(up3, x2)
        up1 = self.decoder2(up2, x1)
        out = self.out(up1)
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = DIY(in_channels=1, out_channels=2).to(device)
    input = torch.randn(1, 1, 16, 64, 64).to(device)
    output = model(input)
    
    print(output.shape)


###############################################################################
# Learnable convolution-based wavelet transform
#
# These classes replace the ptwt calls used by the original ConvDWT/ConvIDWT
# with the same separable filter-bank construction as LWTformer.  The forward
# transform is exactly equivalent to ptwt.wavedec2/3 when the filters are
# initialized from a PyWavelets orthogonal wavelet and the inverse transform is
# exactly equivalent to ptwt.waverec2/3 when the synthesis filters are
# initialized from the corresponding reconstruction filters.
###############################################################################


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Outer product of two 1-D tensors."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    return a[:, None] * b[None, :]


def _outer3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Outer product of three 1-D tensors."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    c = c.reshape(-1)
    return a[:, None, None] * b[None, :, None] * c[None, None, :]


def _wavelet_filter_tensors(wavelet: str, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    wavelet_obj = pywt.Wavelet(wavelet)
    if kind == "analysis":
        lo = torch.tensor(wavelet_obj.dec_lo, dtype=torch.float32).reshape(1, -1)
        hi = torch.tensor(wavelet_obj.dec_hi, dtype=torch.float32).reshape(1, -1)
    elif kind == "synthesis":
        lo = torch.tensor(wavelet_obj.rec_lo, dtype=torch.float32).reshape(1, -1)
        hi = torch.tensor(wavelet_obj.rec_hi, dtype=torch.float32).reshape(1, -1)
    else:
        raise ValueError(f"Unknown wavelet filter kind: {kind}")
    return lo, hi


def _build_2d_wavelet_filters(lo: torch.Tensor, hi: torch.Tensor, flip: bool) -> torch.Tensor:
    lo = lo.reshape(-1)
    hi = hi.reshape(-1)
    if flip:
        lo = lo.flip(0)
        hi = hi.flip(0)

    # Order matches ptwt.wavedec2 details: [LL, horizontal, vertical, diagonal].
    filters = [
        _outer(lo, lo),
        _outer(hi, lo),
        _outer(lo, hi),
        _outer(hi, hi),
    ]
    return torch.stack(filters, dim=0)  # [4, k, k]


def _build_3d_wavelet_filters(lo: torch.Tensor, hi: torch.Tensor, flip: bool) -> torch.Tensor:
    lo = lo.reshape(-1)
    hi = hi.reshape(-1)
    if flip:
        lo = lo.flip(0)
        hi = hi.flip(0)

    # ptwt.wavedec3 returns the details in this exact order:
    #   aad, ada, add, daa, dad, dda, ddd
    # The first filter is the low-pass LLL component.
    bands = [
        ("a", "a", "a"),
        ("a", "a", "d"),
        ("a", "d", "a"),
        ("a", "d", "d"),
        ("d", "a", "a"),
        ("d", "a", "d"),
        ("d", "d", "a"),
        ("d", "d", "d"),
    ]
    filters = []
    for dz, dy, dx in bands:
        fz = lo if dz == "a" else hi
        fy = lo if dy == "a" else hi
        fx = lo if dx == "a" else hi
        filters.append(_outer3(fz, fy, fx))
    return torch.stack(filters, dim=0)  # [8, k, k, k]


def _get_pad_pair(length: int, filt_len: int) -> tuple[int, int]:
    """Return (left, right) padding matching PyWavelets zero padding."""
    pad_left = (2 * filt_len - 3) // 2
    pad_right = pad_left
    if length % 2 != 0:
        pad_right += 1
    return pad_left, pad_right


def _pad_mode(mode: str) -> str:
    if mode in ("zero", "constant"):
        return "constant"
    return mode


class LearnableConvDWT(nn.Module):
    """Learnable separable 2-D/3-D discrete wavelet transform.

    The input format and returned coefficient layout are identical to the
    original ptwt-based :class:`ConvDWT`, so the rest of ``wav_Enhance`` can be
    reused without changes.
    """

    def __init__(self, wavelet: str = "db1", level: int = 1, mode: str = "zero"):
        super().__init__()
        if level != 1:
            raise NotImplementedError("LearnableConvDWT currently only supports level=1.")
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

        dec_lo, dec_hi = _wavelet_filter_tensors(wavelet, "analysis")
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
        self.filt_len = int(self.dec_lo.numel())

    def _pad_2d(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_top, pad_bottom = _get_pad_pair(h, self.filt_len)
        pad_left, pad_right = _get_pad_pair(w, self.filt_len)
        pad_mode = _pad_mode(self.mode)
        if pad_mode == "constant":
            return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode=pad_mode, value=0.0)
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], mode=pad_mode)

    def _pad_3d(self, x: torch.Tensor) -> torch.Tensor:
        d, h, w = x.shape[-3:]
        pad_front, pad_back = _get_pad_pair(d, self.filt_len)
        pad_top, pad_bottom = _get_pad_pair(h, self.filt_len)
        pad_left, pad_right = _get_pad_pair(w, self.filt_len)
        pad_mode = _pad_mode(self.mode)
        if pad_mode == "constant":
            return F.pad(
                x,
                [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back],
                mode=pad_mode,
                value=0.0,
            )
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back], mode=pad_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        lo = self.dec_lo
        hi = self.dec_hi

        if d == 1:
            x2 = x.squeeze(2)
            x2 = self._pad_2d(x2)
            filters = _build_2d_wavelet_filters(lo, hi, flip=True)  # [4, k, k]
            kernel = filters.repeat(c, 1, 1).unsqueeze(1)            # [4*c, 1, k, k]
            out = F.conv2d(x2, kernel, stride=2, groups=c)           # [B, 4*c, H/2, W/2]
            out = out.reshape(b, c, 4, out.shape[-2], out.shape[-1])

            ll = out[:, :, 0:1]
            horizontal = out[:, :, 1:2]
            vertical = out[:, :, 2:3]
            diagonal = out[:, :, 3:4]

            yl = ll
            yh = torch.cat((horizontal, vertical, diagonal), dim=1)  # [B, 3*c, 1, H/2, W/2]
            return torch.cat((yl, yh), dim=1)                        # [B, 4*c, 1, H/2, W/2]

        x3 = self._pad_3d(x)
        filters = _build_3d_wavelet_filters(lo, hi, flip=True)  # [8, k, k, k]
        kernel = filters.repeat(c, 1, 1, 1).unsqueeze(1)         # [8*c, 1, k, k, k]
        out = F.conv3d(x3, kernel, stride=2, groups=c)           # [B, 8*c, D/2, H/2, W/2]
        out = out.reshape(b, c, 8, out.shape[-3], out.shape[-2], out.shape[-1])

        yl = out[:, :, 0]
        details = [out[:, :, i] for i in range(1, 8)]
        yh = torch.stack(details, dim=1)                        # [B, 7, c, D/2, H/2, W/2]
        yh = yh.reshape(b, 7 * c, yh.shape[-3], yh.shape[-2], yh.shape[-1])
        return torch.cat((yl, yh), dim=1)                       # [B, 8*c, D/2, H/2, W/2]


class LearnableConvIDWT(nn.Module):
    """Learnable separable inverse 2-D/3-D discrete wavelet transform."""

    def __init__(self, wavelet: str = "db1", mode: str = "zero"):
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode

        rec_lo, rec_hi = _wavelet_filter_tensors(wavelet, "synthesis")
        self.rec_lo = nn.Parameter(rec_lo, requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi, requires_grad=True)
        self.filt_len = int(self.rec_lo.numel())

    def forward(self, low_freqs: torch.Tensor, high_freqs: torch.Tensor | None = None) -> torch.Tensor:
        if high_freqs is None:
            b, channels, d, h, w = low_freqs.shape
            if d == 1 and channels % 4 == 0:
                c = channels // 4
            elif channels % 8 == 0:
                c = channels // 8
            else:
                raise ValueError(f"Expected channels to be divisible by 4 or 8, got {channels}.")
            high_freqs = low_freqs[:, c:, :, :, :]
            low_freqs = low_freqs[:, :c, :, :, :]

        b, c, d, h, w = low_freqs.shape
        lo = self.rec_lo
        hi = self.rec_hi

        if high_freqs.shape[1] == 3 * c:
            if d != 1:
                raise ValueError("2-D inverse wavelet transform expects a singleton depth dimension.")
            if high_freqs.shape != (b, 3 * c, 1, h, w):
                raise ValueError(
                    f"Expected high_freqs shape {(b, 3 * c, 1, h, w)}, got {tuple(high_freqs.shape)}."
                )
            low = low_freqs.squeeze(2)                       # [B, c, h, w]
            high = high_freqs.squeeze(2)                     # [B, 3*c, h, w]
            high_bands = torch.chunk(high, 3, dim=1)          # [B, c, h, w] each
            coeffs = torch.stack((low, *high_bands), dim=2)   # [B, c, 4, h, w]
            coeffs = coeffs.reshape(b, 4 * c, h, w)           # group-major: [LL,H,V,D] per channel
            filters = _build_2d_wavelet_filters(lo, hi, flip=False)
            kernel = filters.repeat(c, 1, 1).unsqueeze(1)                  # [4*c, 1, k, k]
            rec = F.conv_transpose2d(coeffs, kernel, stride=2, groups=c)
            crop = (2 * self.filt_len - 3) // 2
            if crop > 0:
                rec = rec[..., crop:-crop, crop:-crop]
            return rec.unsqueeze(2)

        if high_freqs.shape[1] != 7 * c:
            raise ValueError(
                f"Expected high_freqs to have 3*c or 7*c channels, got {high_freqs.shape[1]}."
            )
        if high_freqs.shape != (b, 7 * c, d, h, w):
            raise ValueError(
                f"Expected high_freqs shape {(b, 7 * c, d, h, w)}, got {tuple(high_freqs.shape)}."
            )

        high_bands = torch.chunk(high_freqs, 7, dim=1)      # 7 x [B, c, d, h, w]
        coeffs = torch.stack((low_freqs, *high_bands), dim=2)  # [B, c, 8, d, h, w]
        coeffs = coeffs.reshape(b, 8 * c, d, h, w)          # group-major: [LLL + 7 details] per channel
        filters = _build_3d_wavelet_filters(lo, hi, flip=False)
        kernel = filters.repeat(c, 1, 1, 1).unsqueeze(1)    # [8*c, 1, k, k, k]
        rec = F.conv_transpose3d(coeffs, kernel, stride=2, groups=c)
        crop = (2 * self.filt_len - 3) // 2
        if crop > 0:
            rec = rec[..., crop:-crop, crop:-crop, crop:-crop]
        return rec


class LearnableWaveletBank(nn.Module):
    """Container for the learnable analysis/synthesis wavelet filters.

    ``LearnableDIYMultiScaleGate`` owns a single instance of this module and
    passes it to every ``LearnableWavEnhance`` block, mirroring LWTformer's
    globally shared learnable wavelet filters.
    """

    def __init__(self, wavelet: str = "db1", mode: str = "zero"):
        super().__init__()
        self.dwt = LearnableConvDWT(wavelet=wavelet, level=1, mode=mode)
        self.idwt = LearnableConvIDWT(wavelet=wavelet, mode=mode)


class LearnableWavEnhance(nn.Module):
    """Drop-in learnable counterpart of :class:`wav_Enhance`."""

    def __init__(self, wavelet_bank: LearnableWaveletBank, in_dim: int):
        super().__init__()
        self.in_dim = in_dim

        # Keep the shared bank out of this block's child-module registry.  The
        # top-level backbone owns it once, which avoids duplicated state_dict
        # keys while still letting every enhancement block use the same filters.
        object.__setattr__(self, "wavelet_bank", wavelet_bank)

        self.ca = ChannelAttention(in_dim)
        self.sa = SpatialAttention()
        self.gaussian_filter_2d_depth = LearnableGaussianFilterBank3D(
            kernel_size=3,
            num_filters=1,
            num_channels=3 * in_dim,
        )
        self.gaussian_filter_3d = LearnableGaussianFilterBank3D(
            kernel_size=3,
            num_filters=1,
            num_channels=7 * in_dim,
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.last_attended_low: torch.Tensor | None = None
        self.last_raw_high: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            b, c, d, h, w = x.shape
            dwt_out = self.wavelet_bank.dwt(x)

            ll = dwt_out[:, :c, :, :, :]
            yh = dwt_out[:, c:, :, :, :]

            ca_mask = self.ca(ll)
            sa_mask = self.sa(ca_mask * ll)
            att = ca_mask * sa_mask
            self.last_attended_low = (ll * att).detach()

            gaussian_filter = self.gaussian_filter_2d_depth if d == 1 else self.gaussian_filter_3d
            self.last_raw_high = yh.detach()
            yh_low = gaussian_filter(yh)
            yh_high = yh - yh_low
            yh = yh + self.alpha * yh_high

            subbands = 3 if d == 1 else 7
            yh = yh * att.repeat_interleave(subbands, dim=1)

            x_rec = self.wavelet_bank.idwt(ll, yh)
        return x_rec.to(orig_dtype)
