import torch
from tensorboardX import writer, SummaryWriter
from torch import nn
from torch.autograd import Variable
import os
import argparse
from datetime import datetime

import torch.nn.functional as F

from lib.FSANet import FSANet

from utils.AdaX import AdaXW
from utils.dataloader import get_loader
from utils.utils import AvgMeter, clip_gradient, poly_lr, adjust_lr

P1_writer = SummaryWriter("./logs/FSANet/P1")
P2_writer = SummaryWriter("./logs/FSANet/P2")
P3_writer = SummaryWriter("./logs/FSANet/P3")
P4_writer = SummaryWriter("./logs/FSANet/P4")



def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]  # 多尺度训练，此处默认用1,0.75, 1, 1.25
    loss_P1_record = AvgMeter()
    loss_P2_record = AvgMeter()
    loss_P3_record = AvgMeter()
    loss_P4_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            # ---- forward ----
            P1, P2, P3, P4, P5 = model(images)

            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)



            loss = 0.7*loss_P1 + 0.1*loss_P2 + 0.1*loss_P3 + 0.1*loss_P4

            # ---- backward ----
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P1_record.update(loss_P1.data, opt.batchsize)
                loss_P2_record.update(loss_P2.data, opt.batchsize)
                loss_P3_record.update(loss_P3.data, opt.batchsize)
                loss_P4_record.update(loss_P4.data, opt.batchsize)

        # ---- train visualization ----
        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-1: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P1_record.show()))

        if i == total_step:
            P1_writer.add_scalar("Average_loss_P1", loss_P1_record.show(), epoch)
            P2_writer.add_scalar("Average_loss_P2", loss_P2_record.show(), epoch)
            P3_writer.add_scalar("Average_loss_P3", loss_P3_record.show(), epoch)
            P4_writer.add_scalar("Average_loss_P4", loss_P4_record.show(), epoch)

    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 5 == 0:
        state_dict = {
            "opech": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss
        }
        torch.save(state_dict, save_path + 'FSANet-%d.pth' % (epoch + 1))
        print('[Saving Snapshot:]', save_path + 'FSANet-%d.pth' % (epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=2e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=736, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.2, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=20, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='FSANet')
    parser.add_argument('--pre_model', type=str,
                        default="None")
    opt = parser.parse_args()

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model = FSANet()
    # model.train()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    params = model.parameters()
    optimizer = AdaXW(params, opt.lr, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if os.path.exists(opt.pre_model):
        checkpoint = torch.load(opt.pre_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['opech']
        print('加载 epoch {} 成功！'.format(start_epoch + 1))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    # 训练

    for epoch in range(opt.epoch):
        epoch = epoch + start_epoch + 1
        # adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)
        print("第{}epoch的学习率是：{}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))
