import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn as nn
import math

# from torch.utils.tensorboard import SummaryWriter

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch


# 具体流程可以参考图1，通道注意力机制
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #

        return x


class Att(nn.Module):

    def __init__(self, channels, shape, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# SE注意力机制
class se_block(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 自定义注意力模块
class DualContextAttention1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualContextAttention1, self).__init__()
        self.dual1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.dual2_l = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dual2_r = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        feature = x
        y = self.dual1(x)
        y = x * y
        y_l = self.dual2_l(y)
        y_r = self.dual2_r(y)
        y = y_l + y_r
        y = self.sig(y)
        result = feature * y

        return result


# 自定义注意力模块
class DualContextAttention2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualContextAttention2, self).__init__()
        self.dual1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.dual2_l = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dual2_r = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        feature = x
        y = self.dual1(x)
        y = x * y
        y_l = self.dual2_l(y)
        y_r = self.dual2_r(y)
        y = y_l + y_r
        y = self.sig(y)
        result = feature * y

        return result


# 自定义注意力机制
# class DualContextAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=16):
#         super(DualContextAttention, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.reduction = reduction
#
#         # 自注意力模块
#         self.query_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#         # 交叉注意力模块
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1)
#
#         # 双重注意力权重
#         self.att1 = nn.Conv2d(out_channels // 2, 1, kernel_size=1, bias=False)
#         self.att2 = nn.Conv2d(out_channels, 1, kernel_size=1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # 自注意力模块
#         proj_query = self.query_conv(x)
#         proj_key = self.key_conv(x)
#         proj_value = self.value_conv(x)
#         n, c, h, w = proj_query.size()
#         proj_query = proj_query.view(n, c // 2, h * w).permute(0, 2, 1)
#         proj_key = proj_key.view(n, c // 2, h * w)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.sigmoid(self.att1(energy)).view(n, 1, h, w)
#         proj_value = proj_value.view(n, self.out_channels, h * w)
#         out = torch.bmm(proj_value, attention.view(n, h * w, 1))
#         out = out.view(n, self.out_channels // 2, h, w)
#
#         # 交叉注意力模块
#         x_pool = self.pool(x)
#         x_pool = self.conv2(self.relu(self.conv1(x_pool)))
#         x_conv = self.conv2(self.relu(self.conv1(x)))
#         attention = self.sigmoid(self.att2(x_pool * x_conv))
#         out = attention * out + (1 - attention) * x
#
#         return out

# 自定义注意力机制
class DualContextAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualContextAttention, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.query_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.dual1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.dual2_l = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.dual2_r = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size, _, h, w = x.size()
        # print(x.shape)
        # Q, K, V的计算
        proj_query = self.query_conv(x).view(batch_size, self.out_channels // 2, -1).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, self.out_channels // 2, -1)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, self.out_channels, -1)
        feature = torch.bmm(proj_value, attention.permute(0, 2, 1))
        feature = feature.view(batch_size, self.out_channels, h, w)

        # print(feature.shape)

        # Dual Context Attention
        y = self.dual1(feature)
        y = feature * y
        y_l = self.dual2_l(y)
        y_r = self.dual2_r(y)
        y = y_l + y_r
        y = self.sig(y)
        result = feature * y

        return result


class My_AttentionFusion(nn.Module):
    def __init__(self, channels=64, reduction=4):
        super(My_AttentionFusion, self).__init__()

        # Global Attention Branch
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Local Attention Branch
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Channel Attention Branch
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Fusion Branch
        self.conv = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Global Attention
        x_global = self.global_att(x) * x

        # Local Attention
        x_local = self.local_att(x) * x

        # Channel Attention
        x_channel = self.channel_att(x) * x

        # Concatenate the attention branches
        out = torch.cat((x_global, x_local, x_channel), dim=1)

        # Fusion Branch
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


if __name__ == "__main__":
    data = torch.rand(3, 64, 64, 64)
    model = DualContextAttention2(64, 64)
    result = model(data)
    print(result.shape)

    # 创建一个输入张量，假设输入张量的shape是[batch_size, in_channels, height, width]
    # x = torch.randn(1, 64, 32, 32)
    #
    # # 初始化DualContextAttention
    # dual_context_attention = DualContextAttention(64, 128)
    #
    # # 将DualContextAttention添加到Tensorboard中，方便可视化
    # writer = SummaryWriter("./logs/attention")
    # writer.add_graph(dual_context_attention, x)
    #
    # # 关闭Tensorboard写入器
    # writer.close()
