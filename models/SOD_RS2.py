import torch
import torch.nn as nn
from models.CMMamba.Cross_Model_MambaFU import SinMamba_
from models.CMMamba.Cross_Model_MambaFU import PatchEmbed
from models.VMamba.classification.models import build_vssm_model
from models.VMamba.classification.config import get_config
from options import opt
from WTconv import WTSplit
from WTconv import DWT as DWT2
import torch.nn.functional as F
import os
import math


class LTCBlockFused(nn.Module):
    def __init__(self, channels, hidden_size=None, time_steps=2, dt=0.2):
        super().__init__()
        self.channels = channels
        self.hidden_size = hidden_size or channels
        self.time_steps = time_steps
        self.dt = dt

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

            # 门控 f(x,h)
            # x_mean = F.adaptive_avg_pool2d(x, 1)
            # h_mean = F.adaptive_avg_pool2d(h, 1)
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
        w = self.att(x)  # [NSI,2,1,1]
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
            DWConv(2 * out_channel, out_channel),
            # nn.BatchNorm2d(out_channel),
            # nn.ReLU()
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
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [NSI, HW, C]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = x.permute(0, 2, 1).view(B, -1, H, W)  # [NSI, C, H, W]
        return x

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

class ChannelAttention2(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        """
        通道注意力模块
        Args:
            in_channels: 输入通道数
            reduction_ratio: 缩减比例，用于中间层通道数
        """
        super(ChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享权重的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_weights = torch.sigmoid(avg_out + max_out)
        return channel_weights


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        """
        空间注意力模块
        Args:
            kernel_size: 卷积核大小，应为奇数
        """
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # 沿通道维度计算均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = torch.sigmoid(self.conv(combined))
        return spatial_weights


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, spatial_kernel=7):
        """
        CBAM注意力模块
        Args:
            in_channels: 输入通道数
            reduction_ratio: 通道注意力缩减比例
            spatial_kernel: 空间注意力卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention2(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        # 通道注意力
        channel_weights = self.channel_att(x)
        x_channel_att = x * channel_weights

        # 空间注意力
        spatial_weights = self.spatial_att(x_channel_att)
        x_att = x_channel_att * spatial_weights

        # 残差连接
        return x + x_att  # 残差连接增强信息流

class WaveletEdgeFusionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 分支1：方向卷积
        self.dir_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1,3), padding=(0,1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU()
        )

        # 分支2：注意力融合
        self.attn = CBAM(in_channels)

        # 分支3：残差
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(in_channels * 3, in_channels, 1)

    def forward(self, HL, LH, HH):
        edge_raw = HL + LH + HH

        d_feat = self.dir_conv(edge_raw)
        a_feat = self.attn(edge_raw)
        r_feat = self.res_conv(edge_raw) + edge_raw

        fused = torch.cat([d_feat, a_feat, r_feat], dim=1)
        return self.out_conv(fused)


class WEdge(nn.Module):
    def __init__(self, in_ch, out_ch=32, scale_factor=None):
        super(WEdge, self).__init__()

        # 特征变换
        self.wt = DWT2()
        self.split = WTSplit()
        self.scale = scale_factor
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.fusion = WaveletEdgeFusionBlock(in_ch)
        self.out = nn.Conv2d(in_ch, out_ch, 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x):
        wave = self.wt(x)
        _, hl, lh, hh = self.split(wave)
        hl = self.up2(hl)
        lh = self.up2(lh)
        hh = self.up2(hh)
        fuse = self.fusion(hl, lh, hh)
        out = self.out(fuse)
        if self.scale is not None:
            out = self.up(out)
        return out

class RegionBoundaryGraphReasoning(nn.Module):
    def __init__(self, in_channels, edge_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or in_channels // 2

        # 区域和边缘的编码器
        self.region_proj = nn.Conv2d(in_channels, inter_channels, 1)
        self.bound_proj = nn.Conv2d(edge_channels, inter_channels, 1)

        # 融合后的节点特征变换
        self.graph_conv = nn.Sequential(
            nn.Conv1d(inter_channels, inter_channels, 1, bias=False),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True)
        )

        # 输出映射
        self.out_proj = nn.Conv2d(inter_channels, in_channels, 1)

    def forward(self, region_feat, edge_feat):
        """
        region_feat: B x C x H x W（显著区域特征图）
        edge_feat:   B x C x H x W（边缘特征图）
        """
        B, C, H, W = region_feat.shape
        N = H * W

        # 特征映射
        r = self.region_proj(region_feat)  # B x C' x H x W
        e = self.bound_proj(edge_feat)     # B x C' x H x W

        # flatten 为节点序列
        r = r.view(B, -1, N)  # B x C' x N
        e = e.view(B, -1, N)  # B x C' x N

        # 计算相似度（邻接矩阵）
        affinity = torch.bmm(r.transpose(1, 2), e) / (r.size(1) ** 0.5)  # B x N x N
        affinity = F.softmax(affinity, dim=-1)

        # 图卷积更新区域节点（邻接乘以边缘引导）
        out = torch.bmm(r, affinity)  # B x C' x N
        out = self.graph_conv(out)    # Graph conv over nodes
        out = out.view(B, -1, H, W)

        # 投影回原通道并残差连接
        out = self.out_proj(out)
        return region_feat + out  # 可选 gating 融合

class DecoderBase2(nn.Module):
    def __init__(self, dim, scale_factor=2):
        super(DecoderBase2, self).__init__()
        self.down = nn.Conv2d(dim, dim // 2, kernel_size=1)

        self.dwconv1 = DWConv(dim // 2, dim // 2, dilation=1)
        self.dwconv2 = DWConv(dim // 2, dim // 2, dilation=2)
        self.dwconv3 = DWConv(dim // 2, dim // 2, dilation=3)
        self.dwconv4 = DWConv(dim // 2, dim // 2, dilation=4)
        self.dwconv5 = DWConv(dim // 2, dim // 2, dilation=5)

        self.eca = ECA((dim // 2)+32)

        self.ap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=-1)
        self.mlp = MLP2D(in_channels=dim // 2, out_channels=dim // 2)
        self.bn = nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.dwconv_out1 = DWConv(dim // 2, dim // 2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        self.dwconv_out2 = DWConv((dim // 2)+32, dim // 2)
        self.out = nn.Conv2d(dim // 2, 1, kernel_size=1)

    def forward(self, x, edge):
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
        out = torch.cat([out, edge], dim=1)
        e = self.eca(out)
        out = out + e
        out = self.dwconv_out2(out)
        out = self.out(out)
        return out

# WLNet
class WLNet(nn.Module):
    def __init__(self):
        super(WLNet, self).__init__()

        self.vmamba_config = get_config(opt)
        self.encoder = build_vssm_model(self.vmamba_config)
        if opt.pre:
              if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.encoder.load_state_dict(checkpoint['model'], strict=False)
                print("=> loaded checkpoint")

        self.edge = WEdge(96)
        self.rgb_to_token1 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.deep_fusion1 = SinMamba_(96)

        self.rgb_to_token2 = PatchEmbed(in_chans=96, embed_dim=96, patch_size=1, stride=1)
        self.deep_fusion2 = SinMamba_(96)

        self.rlnn3 = LTCBlockFused(192, 96, 3, dt=0.3)

        self.rlnn4 = LTCBlockFused(384, 192, 4,dt=0.3)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)


        # Decoder 1
        self.rfb0_1 = ImprovedGCM(96, 64)
        self.rfb1_1 = ImprovedGCM(96, 64)
        self.rfb2_1 = ImprovedGCM(192, 64)
        self.rfb3_1 = ImprovedGCM(384, 64)
        self.rfb4_1 = ImprovedGCM(768, 64)

        self.graph1 = RegionBoundaryGraphReasoning(64, 32)
        self.graph2 = RegionBoundaryGraphReasoning(64, 32)
        self.graph3 = RegionBoundaryGraphReasoning(64, 32)
        self.graph4 = RegionBoundaryGraphReasoning(64, 32)

        self.dwc1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dwc2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dwc3 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.dwc4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dwc5 = nn.Conv2d(320, 320, kernel_size=3, padding=1)

        self.salhead1 = Decoder(64, 32)
        self.salhead2 = Decoder(128, 16)
        self.salhead3 = Decoder(192, 8)
        self.salhead4 = Decoder(256, 4)
        self.salhead5 = DecoderBase2(320, 4)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.edge_out = nn.Conv2d(32, 1, 1)

    def forward(self, x_rgb):

        x1_rgb, x2_rgb, x3_rgb, x4_rgb, x5_rgb = self.encoder(x_rgb)
        x1_rgb = x1_rgb.permute(0, 3, 1, 2)
        x2_rgb = x2_rgb.permute(0, 3, 1, 2)
        x3_rgb = x3_rgb.permute(0, 3, 1, 2)
        x4_rgb = x4_rgb.permute(0, 3, 1, 2)
        x5_rgb = x5_rgb.permute(0, 3, 1, 2)
        # layer1 merge

        edge1 = self.edge(x2_rgb)
        edge2 = self.down(edge1)
        edge3 = self.down(edge2)
        edge4 = self.down(edge3)

        x1_rgb_token = self.rgb_to_token1(x1_rgb)
        r_1 = self.deep_fusion1(x1_rgb_token)  # (b,96,88,88)
        B, HW, C = r_1.shape
        H = W = int(math.sqrt(HW))
        r_1 = r_1.transpose(1, 2).view(B, C, H, W)  # Reshape: (1,16384,32)-->(1, 32, 128,128)

        # layer2 merge

        x2_rgb_token = self.rgb_to_token2(x2_rgb)
        r_2 = self.deep_fusion2(x2_rgb_token)  # # (b,96,88,88)
        B, HW, C = r_2.shape
        H = W = int(math.sqrt(HW))
        r_2 = r_2.transpose(1, 2).view(B, C, H, W)  # Reshape: (1,16384,32)-->(1, 32, 128,128)

        # layer3 merge

        r_3 = self.rlnn3(x3_rgb)

        # layer4 merge

        r_4 = self.rlnn4(x4_rgb)

        # layer5 merge
        x5_fusion = x5_rgb  # (b,768,11,11)


        # produce initial saliency map by decoder1
        x0_1 = self.rfb0_1(r_1)
        x1_1 = self.rfb1_1(r_2)
        x2_1 = self.rfb2_1(r_3)  # (b,192,44,44) ->(b,64,44,44), x2_1==x3_fusion
        x3_1 = self.rfb3_1(r_4)  # (b,384,22,22)->(b,64,22,22)
        x4_1 = self.rfb4_1(x5_fusion)  # (b,768,11,11)->(b,64,11,11)

        for _ in range(2):
            x1_1 = self.graph1(x1_1, edge1)
        for _ in range(2):
            x2_1 = self.graph2(x2_1, edge2)
        for _ in range(1):
            x3_1 = self.graph3(x3_1, edge3)
        for _ in range(1):
            x4_1 = self.graph4(x4_1, edge4)


        x1 = self.dwc1(x4_1)
        x1_pred = self.salhead1(x1)
        x2 = torch.cat([x3_1, self.up2(x1)], dim=1)


        x2 = self.dwc2(x2)
        x2_pred = self.salhead2(x2)
        x3 = torch.cat([x2_1, self.up2(x2)], dim=1)


        x3 = self.dwc3(x3)
        x3_pred = self.salhead3(x3)
        x4 = torch.cat([x1_1, self.up2(x3)], dim=1)


        x4 = self.dwc4(x4)
        x4_pred = self.salhead4(x4)
        x5 = torch.cat([x0_1, x4], dim=1)


        edge = self.up4(edge1)
        x5 = self.dwc5(x5)
        x5_pred = self.salhead5(x5, edge)
        edge = self.edge_out(edge)

        return x5_pred, x4_pred, x3_pred, x2_pred, x1_pred, edge


    
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
