from models.ResNet import ResNet50
from utils.Attention import se_block
from utils.tensor_ops import cus_sample


from lib.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class HCF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HCF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch1_1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch1_2 = nn.Sequential(
            BasicConv2d(out_channel * 2, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
        )
        self.branch1_3 = nn.Sequential(
            BasicConv2d(out_channel * 2, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )
        self.branch5 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )
        self.conv_cat = BasicConv2d(2 * out_channel, out_channel, 3, padding=1)

    def forward(self, x):
        x1 = self.branch1_1(x)
        print("@"*100)
        print(x1.shape)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x12 = torch.cat((x1, x2), dim=1)
        x12 = self.branch1_2(x12)
        x13 = torch.cat((x12, x3), dim=1)
        x13 = self.branch1_3(x13)
        x14 = torch.cat((x13, x4), dim=1)
        x14 = self.conv_cat(x14)
        out = self.relu(x5 + x14)

        return out



# 8倍下采样
class Down_8(nn.Module):
    def __init__(self, in_chan, out_chan1, out_chan2, out_chan3, kernal_size=3, stride=2, pad=1):
        super(Down_8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan1, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.conv2 = nn.Conv2d(in_channels=out_chan1, out_channels=out_chan2, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.conv3 = nn.Conv2d(in_channels=out_chan2, out_channels=out_chan3, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.bn = nn.BatchNorm2d(out_chan3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.bn(x)
        return out


# 4倍下采样
class Down_4(nn.Module):
    def __init__(self, in_chan, out_chan1, out_chan2, kernal_size=3, stride=2, pad=1):
        super(Down_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan1, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.conv2 = nn.Conv2d(in_channels=out_chan1, out_channels=out_chan2, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.bn = nn.BatchNorm2d(out_chan2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.bn(x)
        return out


# 2倍下采样
class Down_2(nn.Module):
    def __init__(self, in_chan, out_chan, kernal_size=3, stride=2, pad=1):
        super(Down_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv1(x)
        out = self.bn(x)
        return out


# n倍上采样
class Up_n(nn.Module):
    def __init__(self, in_chan, out_chan, kernal_size=1, stride=1, pad=0, n=2):
        super(Up_n, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernal_size, stride=stride,
                               padding=pad)
        self.bn = nn.BatchNorm2d(out_chan)
        self.upsample = nn.Upsample(scale_factor=n, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.upsample(x)
        return out



class HighPassFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.radius = nn.Parameter(torch.tensor([100.0], dtype=torch.float), requires_grad=True)
        # self.radius = 0.5

    def forward(self, x):
        B, C, H, W = x.shape

        # generate low pass filter
        grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1).float().cuda()
        center = torch.Tensor([(H - 1) / 2, (W - 1) / 2]).cuda()
        dist = torch.sqrt(torch.sum((grid - center) ** 2, dim=-1))
        lpf = torch.where(dist <= self.radius, torch.ones((B, C, H, W)).cuda(), torch.zeros((B, C, H, W)).cuda())

        # generate high pass filter
        hpf = 1 - lpf

        # move data to Fourier domain
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        x_fft = torch.roll(x_fft, (H // 2, W // 2), dims=(2, 3))
        # apply high pass filter
        hf = x_fft * hpf

        # move data back to image domain
        img_h = torch.fft.ifftn(hf, dim=(-2, -1)).abs()

        return img_h




class SEA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(SEA, self).__init__()
        out_channels = int(channels // r)

        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # channel_att
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // r, kernel_size=1),
            nn.BatchNorm2d(channels // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # local_att
        xl = self.local_att(x)
        # global_att
        xg = self.global_att(x)
        # channel_att
        xc = self.channel_att(x)
        # weighted sum of local and global attention features
        xlg = xl + xg
        # apply channel attention
        xla = xc * xlg
        # sigmoid activation
        wei = self.sig(xla)

        return wei

class SFF(nn.Module):
    def __init__(self, channels=64):
        super(SFF, self).__init__()

        self.sea = SEA(channels)
        self.upsample = cus_sample
        # feature fusion with gated mechanism
        self.conv_xy = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_xy = nn.BatchNorm2d(channels * 2)
        self.conv_gate = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_gate = nn.BatchNorm2d(channels * 2)
        self.sigmoid = nn.Sigmoid()

        self.conv_out = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        y = self.upsample(y, scale_factor=2)
        xy = torch.cat((x, y), dim=1)

        # feature fusion with gated mechanism
        xy_conv = self.conv_xy(xy)
        xy_bn = self.bn_xy(xy_conv)
        xy_relu = self.relu(xy_bn)

        gate_conv = self.conv_gate(xy)
        gate_bn = self.bn_gate(gate_conv)
        gate_sigmoid = self.sigmoid(gate_bn)

        feat = xy_relu * gate_sigmoid
        feat = self.conv_out(feat)

        feat_weight = self.sea(feat)

        feat_weighted_x = x * feat_weight
        feat_weighted_y = y * (1 - feat_weight)

        feat_sum = feat_weighted_x + feat_weighted_y


        return feat_sum


class FSANet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64):
        super(FSANet, self).__init__()
        self.sff = SFF()
        self.frequency = HighPassFilter()

        # self.feture_pre = pre_feature_v3()
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = 'G:\CJG\PVT-HRNet\pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.relu = nn.ReLU(inplace=True)

        self.resnet_f = ResNet50('rgbf')

        self.se1 = se_block(64)
        self.se2 = se_block(128)
        self.se3 = se_block(320)
        self.se4 = se_block(512)

        self.conv_1 = nn.Conv2d(256, 64, 1, 1)
        self.conv_2 = nn.Conv2d(512, 128, 1, 1)
        self.conv_3 = nn.Conv2d(1024, 320, 1, 1)
        self.conv_4 = nn.Conv2d(2048, 512, 1, 1)

        # self.conv_1 = RFB_v2(256, 64)
        # self.conv_2 = RFB_v2(512, 128)
        # self.conv_3 = RFB_v2(1024, 320)
        # self.conv_4 = RFB_v2(2048, 512)

        self.rfb1_after = HCF(64, 64)
        self.rfb2_after = HCF(512, 64)
        self.rfb3_after = HCF(2816, 64)
        self.rfb4_after = HCF(4864, 64)

        self.upsample4_2 = Up_n(512, 320, 1, 1, 0, 2)
        self.upsample4_4 = Up_n(512, 128, 1, 1, 0, 4)
        self.upsample3_2 = Up_n(320, 128, 1, 1, 0, 2)
        self.upsample41_2 = Up_n(2048, 1024, 1, 1, 0, 2)

        self.down1_2 = Down_2(64, 128, 3, 2, 1)
        self.down1_4 = Down_4(64, 128, 320, 3, 2, 1)
        self.down1_8 = Down_8(64, 128, 320, 512, 3, 2, 1)
        self.down2_2 = Down_2(128, 320, 3, 2, 1)
        self.down2_4 = Down_4(128, 128, 512, 3, 2, 1)
        self.down3_2 = Down_2(320, 512, 3, 2, 1)

        self.down21_2 = Down_2(512, 512, 3, 2, 1)
        self.down21_4 = Down_4(512, 512, 512, 3, 2, 1)

        self.down31_2 = Down_2(1280, 1280, 3, 2, 1)

        self.down32_2 = Down_2(2816, 1024, 3, 2, 1)

        self.conv2_1 = nn.Conv2d(64, 1, 3, 1, 1)
        if self.training:
            self.initialize_weights()

        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up4 = nn.Conv2d(64, 64, 3, 1, 1)

    # initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        # all_params = {}
        # for k, v in self.resnet.state_dict().items():
        #     if k in pretrained_dict.keys():
        #         v = pretrained_dict[k]
        #         all_params[k] = v
        #     elif '_1' in k:
        #         name = k.split('_1')[0] + k.split('_1')[1]
        #         v = pretrained_dict[name]
        #         all_params[k] = v
        #     elif '_2' in k:
        #         name = k.split('_2')[0] + k.split('_2')[1]
        #         v = pretrained_dict[name]
        #         all_params[k] = v
        # assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        # self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_f.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_f.state_dict().keys())
        self.resnet_f.load_state_dict(all_params)

    def forward(self, x):
        layer = self.backbone(x)
        frequency = self.frequency(x)
        frequency = torch.mean(frequency, dim=1, keepdim=True)
        # print(frequency.shape)
        # feature = self.backbone_f(frequency)

        x1 = layer[0]  # bs, 64, 256, 256
        x2 = layer[1]  # bs, 128, 128, 128
        x3 = layer[2]  # bs, 320, 64, 64
        x4 = layer[3]  # bs, 512, 32, 32

        x1_f = self.resnet_f.conv1(frequency)
        x_f = self.resnet_f.bn1(x1_f)
        x_f = self.resnet_f.relu(x_f)
        x_f = self.resnet_f.maxpool(x_f)

        # print(x_f.shape)
        y1 = self.resnet_f.layer1(x_f)
        # print(y1.shape)
        y2 = self.resnet_f.layer2(y1)
        y3 = self.resnet_f.layer3_1(y2)
        y4 = self.resnet_f.layer4_1(y3)

        x1 = x1 + self.se1(self.conv_1(y1))
        x2 = x2 + self.se2(self.conv_2(y2))
        x3 = x3 + self.se3(self.conv_3(y3))
        x4 = x4 + self.se4(self.conv_4(y4))

        x1_down2 = self.down1_2(x1)
        x1_down4 = self.down1_4(x1)
        x1_down8 = self.down1_8(x1)

        x2_down2 = self.down2_2(x2)
        x2_down4 = self.down2_4(x2)

        x3_down2 = self.down3_2(x3)
        x3_up2 = self.upsample3_2(x3)

        x4_up2 = self.upsample4_2(x4)
        x4_up4 = self.upsample4_4(x4)

        x21 = torch.cat((x1_down2, x2, x3_up2, x4_up4), dim=1)  # [1, 512, 128, 128]
        x31 = torch.cat((x1_down4, x2_down2, x3, x4_up2), dim=1)  # [1, 1280, 64, 64]
        x41 = torch.cat((x1_down8, x2_down4, x3_down2, x4), dim=1)  # [1, 2048, 32, 32]

        x21_down2 = self.down21_2(x21)
        x21_down4 = self.down21_4(x21)

        x31_down2 = self.down31_2(x31)

        x41_up2 = self.upsample41_2(x41)

        x32 = torch.cat((x21_down2, x31, x41_up2), dim=1)  # [1, 2816, 64, 64]
        x42 = torch.cat((x21_down4, x31_down2, x41), dim=1)  # [1, 3840, 32, 32]

        x32_down2 = self.down32_2(x32)
        x43 = torch.cat((x32_down2, x42), dim=1)  # [1, 4864, 32, 32]

        # x1 = self.sa1(x1) * x1
        # x1 = self.ca1(x1) * x1

        x1_rfb = self.rfb1_after(x1)
        x21_rfb = self.rfb2_after(x21)
        x32_rfb = self.rfb3_after(x32)
        x43_rfb = self.rfb4_after(x43)

        out43 = self.sff(x32_rfb, x43_rfb)
        out432 = self.sff(x21_rfb, out43)
        out4321 = self.sff(x1_rfb, out432)

        out4321 = F.interpolate(out4321, scale_factor=2, mode='bilinear')
        out432 = F.interpolate(out432, scale_factor=2, mode='bilinear')
        out43 = F.interpolate(out43, scale_factor=2, mode='bilinear')
        x43_rfb = F.interpolate(x43_rfb, scale_factor=2, mode='bilinear')

        out4321 = self.conv_up1(out4321)
        out432 = self.conv_up1(out432)
        out43 = self.conv_up1(out43)
        x43_rfb = self.conv_up1(x43_rfb)

        p1 = self.conv2_1(out4321)
        p2 = self.conv2_1(out432)
        p3 = self.conv2_1(out43)
        p4 = self.conv2_1(x43_rfb)

        P1 = F.interpolate(p1, scale_factor=2, mode='bilinear')
        P2 = F.interpolate(p2, scale_factor=4, mode='bilinear')
        P3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        P4 = F.interpolate(p4, scale_factor=16, mode='bilinear')

        return P1, P2, P3, P4, frequency


if __name__ == '__main__':
    model = FSANet().cuda()

    data = torch.randn(3, 3, 256, 256).cuda()
    out = model(data)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)

