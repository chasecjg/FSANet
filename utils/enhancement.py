import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


# from tensorboardX import writer, SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=.20)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class EnhancementNet(nn.Module):
    def __init__(self, num_layers=3):
        super(EnhancementNet, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.blocks = self.StackBlock(ConvBlock, num_layers, in_channels=64, out_channels=64)

    def StackBlock(self, block, layer_num, in_channels, out_channels):
        layers = []
        for _ in range(layer_num):
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        input_x = x
        x1 = self.input(x)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        out = self.blocks(x1)
        out = self.output(out)
        return out


class EnhancementNet_v1(nn.Module):
    def __init__(self):
        super().__init__()

        # self.input = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )

        self.branch0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )
        #
        # self.output = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        feature = x
        # feature = F.interpolate(feature, scale_factor=0.5, mode="bilinear", align_corners=False)
        # x = self.input(x)  # [b, 64, 512, 512]
        x10 = self.branch0(x)
        x11 = x10 + feature
        x20 = self.branch1(x11)
        x21 = x20 + feature
        x30 = self.branch2(x21)
        x31 = x30 + feature
        # output = self.output(x31)
        return x31


# 只做gamma校正
class EnhancementNet_v2(nn.Module):
    def __init__(self):
        super().__init__()

        # self.input = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )

        self.branch0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.fc1 = nn.Linear(640 * 640 * 3, 640)
        self.fc2 = nn.Linear(640, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        feature = x
        x10 = self.branch0(x)
        x11 = x10 + feature
        x20 = self.branch1(x11)
        x21 = x20 + feature
        x30 = self.branch2(x21)
        x31 = x30 + feature
        out = x31.view(-1, 640 * 640 * 3)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        result = self.sig(out)

        gamma = result[:, 0:1] * 1.0 + 0.5

        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min)
        n = x.shape[0]
        # print(x)
        output = torch.empty_like(x)
        for i in range(n):
            xi = x[i].clone()
            x_gamma = torch.pow(xi, gamma[i])
            output[i] = x_gamma
        output = output * (x_max - x_min) + x_min

        return output


if __name__ == '__main__':
    # 创建一个输入张量，假设输入张量的shape是[batch_size, in_channels, height, width]
    # x = torch.randn(1, 3, 32, 32)
    #
    # # 初始化DualContextAttention
    # dual_context_attention = EnhancementNet_v1()
    #
    # # 将DualContextAttention添加到Tensorboard中，方便可视化
    # writer = SummaryWriter("./logs/attention")
    # writer.add_graph(dual_context_attention, x)
    #
    # # 关闭Tensorboard写入器
    # writer.close()
    model = EnhancementNet_v2()
    model.train()
    data = torch.rand(3, 3, 640, 640)
    result = model(data)
    print(result.shape)
