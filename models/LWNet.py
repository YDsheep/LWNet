import torch
import torch.nn as nn
import numbers
from models.CMMamba.Cross_Model_MambaFU import CrossMamba_, SinMamba_
from models.CMMamba.Cross_Model_MambaFU import PatchEmbed
from models.VMamba.classification.models import build_vssm_model
from models.VMamba.classification.config import get_config
from options import opt
from models.wtconv2d import DWT, IWT
import torch.nn.functional as F
import os
import math
from einops import rearrange


class LTCBlockFused(nn.Module):
    def __init__(self, channels, hidden_size=None, time_steps=2, dt=0.2):
        super().__init__()
        self.channels = channels
        self.hidden_size = hidden_size or channels
        self.time_steps = time_steps
        self.dt = dt

        self.W_in = DWConv(channels, self.hidden_size)
        self.W_rec = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.bias = nn.Parameter(torch.zeros(1, self.hidden_size, 1, 1))
        self.tau_net = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )

        self.A_conv = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3,padding=1)  # A(x)

        self.f_gate = nn.Sequential(
            DWConv(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Conv2d(self.hidden_size, self.hidden_size, 3, padding=1),
            nn.Sigmoid()
        )

        self.in_proj = nn.Conv2d(self.channels, self.hidden_size, 1) \
            if self.channels != self.hidden_size else nn.Identity()

        self.output_proj = nn.Conv2d(self.hidden_size, self.channels, 1) \
            if self.hidden_size != self.channels else nn.Identity()

    def forward(self, x):
        x = self.in_proj(x)
        B, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_size, H, W, device=x.device)

        for _ in range(self.time_steps):

            # 驱动项 A
            A = self.A_conv(x)

            f = self.f_gate(torch.cat([x, h], dim=1))
            tau = self.tau_net(x)

            # 更新动力学
            dh = (-(1 / tau + f) * h + f * A)
            h = h + self.dt * dh

        return self.output_proj(h)

class LTCBlockFused2(nn.Module):
    def __init__(self, channels, hidden_size=None, time_steps=2, dt=0.2):
        super().__init__()
        self.channels = channels
        self.hidden_size = hidden_size or channels
        self.time_steps = time_steps
        self.dt = dt

        self.W_in = DWConv(channels, self.hidden_size)
        self.W_rec = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.bias = nn.Parameter(torch.zeros(1, self.hidden_size, 1, 1))
        self.tau_net = nn.Sequential(
            nn.Conv2d(channels, self.hidden_size, kernel_size=1),
            nn.Sigmoid()
        )

        self.A_conv = nn.Conv2d(self.channels, self.hidden_size, kernel_size=1)  # A(x)

        self.f_gate = nn.Sequential(
            DWConv(self.hidden_size + channels, self.hidden_size),
            nn.ReLU(),
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.output_proj = nn.Conv2d(self.hidden_size, self.channels, 1) \
            if self.hidden_size != self.channels else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_size, H, W, device=x.device)

        for _ in range(self.time_steps):

            # 驱动项 A
            A = self.A_conv(x)

            f = self.f_gate(torch.cat([x, h], dim=1))
            tau = self.tau_net(x)

            # 更新动力学
            dh = (-(1 / tau + f) * h + f * A)
            h = h + self.dt * dh

        return self.output_proj(h)


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class DWConv(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=1):
        super(DWConv, self).__init__()
        self.out_planes = out_planes
        self.dwconv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, padding=dilation,
                                groups=in_planes, dilation=dilation)
        self.pconv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.dwconv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pconv(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)  ##返回所有元素的方差
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP2D(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super(MLP2D, self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.permute(0, 2, 1).view(B, -1, H, W)  # [B, C, H, W]
        return x

class AP_MP(nn.Module):
    def __init__(self, stride=2):
        super(AP_MP, self).__init__()
        self.sz = stride
        self.gapLayer = nn.AvgPool2d(kernel_size=self.sz, stride=self.sz)
        self.gmpLayer = nn.MaxPool2d(kernel_size=self.sz, stride=self.sz)

    def forward(self, x):
        apimg = self.gapLayer(x)
        mpimg = self.gmpLayer(x)
        byimg = torch.norm(abs(apimg - mpimg), p=2, dim=1, keepdim=True)
        return byimg


class DynamicDilationConv(nn.Module):
    """动态扩张率卷积 (根据输入内容自适应调整感受野)"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = DWConv(in_ch, out_ch, dilation=3)
        self.conv5 = DWConv(in_ch, out_ch, dilation=5)
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 2, 1),  # 生成两个分支的权重
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.att(x)  # [B,2,1,1]
        return w[:, 0:1] * self.conv3(x) + w[:, 1:2] * self.conv5(x)


class ChannelAttention(nn.Module):
    """通道注意力 (增强重要特征通道)"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedGCM(nn.Module):
    """改进版GCM模块"""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.reduce = nn.Conv2d(in_channel, out_channel, 1)

        # 多分支动态卷积
        self.branch1 = nn.Sequential(
            DynamicDilationConv(out_channel, out_channel),
            ChannelAttention(out_channel)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            ChannelAttention(out_channel)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            DWConv(2 * out_channel, out_channel)
        )

        # 残差连接
        self.res = nn.Identity() if in_channel == out_channel else \
            nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        residual = self.res(x)
        x = self.reduce(x)

        # 并行分支
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        # 动态拼接
        fused = self.fusion(torch.cat([b1, b2], dim=1))
        return fused + residual


class Decoder(nn.Module):
    def __init__(self, dim, scale_factor=2):
        super(Decoder, self).__init__()
        self.down = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.dwconv1 = DWConv(dim // 2, dim // 2, dilation=1)
        self.dwconv2 = DWConv(dim // 2, dim // 2, dilation=2)
        self.dwconv3 = DWConv(dim // 2, dim // 2, dilation=3)
        self.dwconv4 = DWConv(dim // 2, dim // 2, dilation=4)
        self.dwconv5 = DWConv(dim // 2, dim // 2, dilation=5)

        self.ap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=-1)

        self.mlp = MLP2D(in_channels=dim//2, out_channels=dim//2)
        self.bn = nn.BatchNorm2d(dim//2,  eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.dwconv_out1 = DWConv(dim // 2, dim // 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.dwconv_out2 = DWConv(dim // 2, dim//2)
        self.out = nn.Conv2d(dim // 2, 1, kernel_size=1)

    def forward(self, x):
        x = self.down(x)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)
        x4 = self.dwconv4(x)
        x5 = self.dwconv5(x)

        xa = x1 + x2 + x3 + x4 + x5
        xa = self.bn(xa)
        xa = self.relu(xa)
        xa = self.dwconv_out1(xa)
        xa = xa + x
        x_ap = self.ap(xa)
        x_ap = self.mlp(x_ap)
        x_ap = self.softmax(x_ap)
        out = xa * x_ap
        out = self.up(out)
        out = self.dwconv_out2(out)
        out = self.out(out)
        return out


class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y_ = y.squeeze(-1).transpose(-1, -2)

        # Two different branches of ECA module
        y = self.conv(y_)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SelectiveLNN(nn.Module):
    def __init__(self, in_channels, hidden_size=None, time_steps=3, threshold=0.5, fast_mode=False, dt=0.2):
        super().__init__()
        self.threshold = threshold
        self.fast_mode = fast_mode  # 是否使用快速 Euler 近似

        # 生成空间掩码（用于判断深度区域重要性）
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # 下采样特征（降通道）
        self.inner = DWConv(in_channels, in_channels)
        self.out = DWConv(in_channels, in_channels)

        # 轻量 LNN（支持 fast 模式）
        self.light_lnn = LTCBlockFused2(  # 可替换为显式求解器版本
            channels=in_channels,
            hidden_size=hidden_size or in_channels,
            time_steps=time_steps,
            dt=dt
        )

    def forward(self, rgb_feat, depth_feat):
        # Step 1: 生成选择掩码
        fusion_mask = self.fusion_gate(torch.cat([rgb_feat, depth_feat], dim=1))  # [B, 1, H, W]

        # Step 2: 下采样
        depth_down = self.inner(depth_feat)  # [B, C//2, H, W]

        # Step 3: 选择性计算
        with torch.no_grad():
            binary_mask = (fusion_mask > self.threshold).float()

        if self.fast_mode:
            # 使用融合门控 + 显式 LNN + 掩码增强特征
            processed = self.light_lnn(depth_down * binary_mask)
        else:
            # 若非 fast 模式，对整张图处理（可替换为 odeint 版本）
            processed = self.light_lnn(depth_down)

        # Step 4: 重组未处理部分 + 已处理部分
        preserve = depth_down * (1 - binary_mask)
        fused = preserve + processed  # [B, C//2, H, W]

        # Step 5: 通道还原
        out = self.out(fused)  # [B, C, H, W]
        return out


class Mwtblock3(nn.Module):
    def __init__(self, dim, size):
        super(Mwtblock3, self).__init__()

        self.sq = dim
        self.dim = dim//2
        self.Rwt = DWT()

        self.iwt = IWT()
        self.Dwt = DWT()

        self.sig = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.dw1 = DWConv(dim, dim//2)
        self.dw2 = DWConv(dim, dim//2)
        self.apmp = AP_MP()

        self.rmamba = SinMamba_(dim // 2)
        self.dmamba = SinMamba_(dim // 2)
        self.mlpr = MLP2D(in_channels=dim // 2,hidden_channels=dim*2, out_channels=dim//2)
        self.mlpd = MLP2D(in_channels=dim // 2, hidden_channels=dim * 2, out_channels=dim // 2)

        self.to_token1 = PatchEmbed(in_chans=dim//2, embed_dim=dim//2, patch_size=1, stride=1)
        self.to_token2 = PatchEmbed(in_chans=dim // 2, embed_dim=dim // 2, patch_size=1, stride=1)

        self.eca1 = ECA(dim//2)
        self.eca2 = ECA(dim)
        self.eca3 = ECA(dim // 2)

        self.edwc = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1),
            LayerNorm(dim // 2, 'WithBias'),
            nn.GELU()
        )

        self.edwc2 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1),
            LayerNorm(dim // 2, 'WithBias'),
            nn.GELU()
        )

        self.norm1 = LayerNorm(dim//2, 'WithBias')
        self.norm2 = LayerNorm(dim//2, 'WithBias')

        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            LayerNorm(dim, 'WithBias'),
            nn.GELU()
        )


        self.out1 = nn.Sequential(
            nn.Conv2d((dim//2*2)+1, dim//2, kernel_size=1),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, r, d):
        r1, r2 = torch.split(r, self.dim, 1)
        d1, d2 = torch.split(d, self.dim, 1)
        resr2 = r2
        resd2 = d2

        rd = torch.cat([r1, d1], 1)
        rd1 = self.dw1(rd)
        rd2 = self.dw2(rd)

        rd1 = rd1 * r1
        rd2 = rd2 * d1

        r1 = self.to_token1(r1)
        mr = self.rmamba(r1)
        B, HW, C = mr.shape  # B:1, HW:16384,C:32
        H = W = int(math.sqrt(HW))
        mr = mr.transpose(1, 2).view(B, C, H, W)  # Reshape: (1,16384,32)-->(1, 32, 128,128)
        d1 = self.to_token2(d1)
        md = self.dmamba(d1)
        B, HW, C = md.shape  # B:1, HW:16384,C:32
        H = W = int(math.sqrt(HW))
        md = md.transpose(1, 2).view(B, C, H, W)  # Reshape: (1,16384,32)-->(1, 32, 128,128)

        mr = self.mlpr(mr)
        mr = mr * rd1
        md = self.mlpd(md)
        md = md * rd2

        n, c, h, w = r2.shape
        r2 = self.norm1(r2)
        r2 = data_transform(r2)
        r2wt = self.Rwt(r2)
        wrl, wrh = r2wt[:n, ...], r2wt[n:, ...]

        n, c, h, w = d2.shape
        d2 = self.norm2(d2)
        d2 = data_transform(d2)
        d2wt = self.Dwt(d2)

        wdl, wdh = d2wt[:n, ...], d2wt[n:, ...]
        f_h = torch.cat([wrh, wdh], dim=1)
        e = self.eca1(f_h)
        f_h = f_h + e
        f_h = self.edwc(f_h)
        f_h = f_h + wrh * wdh

        f_l = torch.cat([wrl, wdl], dim=1)
        el = self.eca3(f_l)
        f_l = f_l + el
        f_l = self.edwc2(f_l)
        f_l = f_l + wrl * wdl

        iwt = torch.cat([f_l, f_h], dim=0)
        iwt = self.iwt(iwt)
        iwt = inverse_data_transform(iwt)
        iwt = iwt + resr2 + resd2

        frd = mr + md

        res = frd
        frd = self.apmp(frd)
        frd = self.upsample2(frd)
        frd = frd / math.sqrt(self.sq)
        frd = torch.cat([mr, md, frd], 1)
        frd = self.out1(frd) + res

        fuse = torch.cat([frd, iwt], dim=1)
        ee = self.eca2(fuse)
        fuse = fuse + ee
        out = self.out(fuse)
        out = out + fuse

        return out


class DeepLiteFusion(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = channels // reduction

        # 融合通道 attention 引导（轻量注意力）
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 结构增强 DWConv 分支（局部建模）
        self.local_refine = DWConv(channels, channels)

    def forward(self, rgb_feat, depth_feat):
        fused = torch.cat([rgb_feat, depth_feat], dim=1)              # [B, 2C, H, W]
        gate = self.gate(fused)                                       # [B, C, H, W]

        # 融合输出 = RGB增强 + Depth增强
        fused_feat = rgb_feat * gate + depth_feat * (1 - gate)        # [B, C, H, W]
        fused_feat = self.local_refine(fused_feat)                    # 空间细化

        return fused_feat

# WLNet
class WLNet(nn.Module):
    def __init__(self):
        super(WLNet, self).__init__()
        
        # todo 1 Backbone model

        self.vmamba_config = get_config(opt)
        self.encoder = build_vssm_model(self.vmamba_config)
        if opt.pre:
              if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.encoder.load_state_dict(checkpoint['model'], strict=False)
                print("=> loaded checkpoint")

        # todo 3 Cross-Modal Mamba module
        # in_chans是输入特征的维度，embed_dim是想要的输出token的维度
        self.rawD_to_token1 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.rgb_to_token1 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.deep_fusion1 = CrossMamba_(96)
        self.fusion1 = Mwtblock3(96, 8)

        self.rawD_to_token2 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.rgb_to_token2 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.deep_fusion2 = CrossMamba_(96)
        self.fusion2 = Mwtblock3(96, 8)

        self.rlnn3 = LTCBlockFused(192, 96, 3,dt=0.3)
        self.dlnn3 = SelectiveLNN(192,48,3,fast_mode=True,dt=0.3)
        self.fusion3 = Mwtblock3(192, 4)

        self.rlnn4 = LTCBlockFused(384, 192, 4,dt=0.3)
        self.dlnn4 = SelectiveLNN(384,96,4,fast_mode=True,dt=0.3)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.fusion4 = Mwtblock3(384, 2)

        self.fusion5 = DeepLiteFusion(768, 2)

        # todo 4 解码器
        # Decoder 1
        self.rfb0_1 = ImprovedGCM(96, 64)
        self.rfb1_1 = ImprovedGCM(96, 64)
        self.rfb2_1 = ImprovedGCM(192, 64)
        self.rfb3_1 = ImprovedGCM(384, 64)
        self.rfb4_1 = ImprovedGCM(768, 64)

        self.dwc1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dwc2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dwc3 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.dwc4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dwc5 = nn.Conv2d(320, 320, kernel_size=3, padding=1)

        self.salhead1 = Decoder(64, 32)
        self.salhead2 = Decoder(128, 16)
        self.salhead3 = Decoder(192, 8)
        self.salhead4 = Decoder(256, 4)
        self.salhead5 = Decoder(320, 4)

    def forward(self, x_rgb, x_depth):

        x_depth = x_depth.repeat(1, 3, 1, 1)
        x1_rgb, x2_rgb, x3_rgb, x4_rgb, x5_rgb = self.encoder(x_rgb)
        x1_depth, x2_depth, x3_depth, x4_depth, x5_depth = self.encoder(x_depth)
        x1_rgb = x1_rgb.permute(0, 3, 1, 2)
        x2_rgb = x2_rgb.permute(0, 3, 1, 2)
        x3_rgb = x3_rgb.permute(0, 3, 1, 2)
        x4_rgb = x4_rgb.permute(0, 3, 1, 2)
        x5_rgb = x5_rgb.permute(0, 3, 1, 2)

        x1_depth = x1_depth.permute(0, 3, 1, 2)
        x2_depth = x2_depth.permute(0, 3, 1, 2)
        x3_depth = x3_depth.permute(0, 3, 1, 2)
        x4_depth = x4_depth.permute(0, 3, 1, 2)
        x5_depth = x5_depth.permute(0, 3, 1, 2)

        # layer1 merge
        x1_depth_token = self.rawD_to_token1(x1_depth)
        x1_rgb_token = self.rgb_to_token1(x1_rgb)
        r_1, d_1 = self.deep_fusion1(x1_rgb_token, x1_depth_token)  # (b,96,88,88)
        x1_fusion = self.fusion1(r_1, d_1)

        # layer2 merge
        x2_depth_token = self.rawD_to_token2(x2_depth)
        x2_rgb_token = self.rgb_to_token2(x2_rgb)
        r_2, d_2 = self.deep_fusion2(x2_rgb_token, x2_depth_token)  # # (b,96,88,88)
        x2_fusion = self.fusion2(r_2, d_2)

        # layer3 merge
        r_3 = self.rlnn3(x3_rgb)
        d_3 = self.dlnn3(r_3,x3_depth)
        x3_fusion = self.fusion3(r_3, d_3)

        # layer4 merge
        r_4 = self.rlnn4(x4_rgb)
        d_4 = self.dlnn4(r_4,x4_depth)
        x4_fusion = self.fusion4(r_4, d_4)

        # layer5 merge
        x5_fusion = self.fusion5(x5_rgb, x5_depth)


        x0_1 = self.rfb0_1(x1_fusion)
        x1_1 = self.rfb1_1(x2_fusion)
        x2_1 = self.rfb2_1(x3_fusion)  # (b,192,44,44) ->(b,64,44,44)
        x3_1 = self.rfb3_1(x4_fusion)  # (b,384,22,22)->(b,6422,22)
        x4_1 = self.rfb4_1(x5_fusion)  # (b,768,11,11)->(b,64,11,11)

        x1 = self.dwc1(x4_1) + x4_1
        x1_pred = self.salhead1(x1)
        x2 = torch.cat([x3_1, self.up2(x1)], dim=1)


        x2 = self.dwc2(x2) + x2
        x2_pred = self.salhead2(x2)
        x3 = torch.cat([x2_1, self.up2(x2)], dim=1)

        x3 = self.dwc3(x3) + x3
        x3_pred = self.salhead3(x3)
        x4 = torch.cat([x1_1, self.up2(x3)], dim=1)

        x4 = self.dwc4(x4) + x4
        x4_pred = self.salhead4(x4)
        x5 = torch.cat([x0_1, x4], dim=1)

        x5 = self.dwc5(x5) + x5
        x5_pred = self.salhead5(x5)

        return x5_pred, x4_pred, x3_pred, x2_pred,x1_pred


    
    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
