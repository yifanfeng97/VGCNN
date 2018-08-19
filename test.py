# -*- coding: utf-8 -*-
import torch
import config
import models
import numpy as np
import os.path as osp
from utils import meter, cal_map
from torch import nn
from models import VGCNN as Net
from torch.utils.data import DataLoader
from datasets import *

def append(raw, data, flaten=False):
    data = np.array(data)
    if flaten:
        data = data.reshape(-1, 1)
    if raw is None:
        raw = np.array(data)
    else:
        raw = np.vstack((raw, data))
    return raw


def validate(val_loader, net):
    """
    validation for one epoch on the val set
    """
    batch_time = meter.TimeMeter(True)
    data_time = meter.TimeMeter(True)
    prec = meter.ClassErrorMeter(topk=[1], accuracy=True)
    ft_all, lbl_all=None, None

    # testing mode
    net.eval()

    for i, (views, labels) in enumerate(val_loader):
        batch_time.reset()
        # bz x 12 x 3 x 224 x 224
        views = views.to(device=config.device)
        labels = labels.to(device=config.device)

        preds, fts = net(views, get_ft=True)  # bz x C x H x W

        prec.add(preds.detach(), labels.detach())
        ft_all = append(ft_all, fts.detach())
        lbl_all = append(lbl_all, labels.detach(), flaten=True)

        if i % config.print_freq == 0:
            print(f'[{i}/{len(val_loader)}]\t'
                  f'Batch Time {batch_time.value():.3f}\t'
                  f'Epoch Time {data_time.value():.3f}\t'
                  f'Prec@1 {prec.value(1):.3f}\t')

    mAP = cal_map(ft_all, lbl_all)

    print(f'mean class accuracy : {prec.value(1)} ')
    print(f'Retrieval mAP : {mAP} ')
    return prec.value(1), mAP


def main():
    print('Training Process\nInitializing...\n')
    config.init_env()

    val_dataset = view_data(config.view_net.data_root,
                                     status=STATUS_TEST,
                                     base_model_name=config.base_model_name)

    val_loader = DataLoader(val_dataset, batch_size=config.view_net.test.batch_sz,
                            num_workers=config.num_workers,shuffle=True)

    # create model
    net = Net(pretrained=True)
    net = net.to(device=config.device)
    net = nn.DataParallel(net)

    print(f'loading pretrained model from {config.view_net.ckpt_file}')
    checkpoint = torch.load(config.view_net.ckpt_file)
    net.module.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        validate(val_loader, net)

    print('test Finished!')


if __name__ == '__main__':
    main()

