# import torch.nn as nn
# # from modeling.backbone.resnet import ResNet101
# import argparse
# from utils.copy_state_dict import copy_state_dict
# from modeling.deeplab import DeepLab
# from auto_deeplab import AutoDeeplab
# from thop import profile
# from torchsummary import summary
# import warnings
# from new_model import get_default_net
# warnings.filterwarnings('ignore')
#
#
# def total_params(model, log=True):
#     params = sum(p.numel() / 1000.0 for p in model.parameters())
#     if log:
#         print(">>> total params: {:.2f}K".format(params))
#     return params
#
#
# def each_param(model):
#     for p in model.parameters():
#         print(">>> {:.5f}K".format(p.numel()))
#
#
# def obtain_default_search_args():
#     parser = argparse.ArgumentParser(
#         description="PyTorch DeeplabV3Plus Training")
#     parser.add_argument('--backbone', type=str, default='resnet',
#                         choices=['resnet', 'xception', 'drn', 'mobilenet'],
#                         help='backbone name (default: resnet)')
#     parser.add_argument('--out-stride', type=int, default=16,
#                         help='network output stride (default: 8)')
#     parser.add_argument('--dataset', type=str, default='cityscapes',
#                         choices=['pascal', 'coco', 'cityscapes', 'kd'],
#                         help='dataset name (default: pascal)')
#     parser.add_argument('--autodeeplab', type=str, default='search',
#                         choices=['search', 'train'])
#     parser.add_argument('--use-sbd', action='store_true', default=False,
#                         help='whether to use SBD dataset (default: True)')
#     parser.add_argument('--load-parallel', type=int, default=0)
#     parser.add_argument('--clean-module', type=int, default=0)
#     parser.add_argument('--workers', type=int, default=0,
#                         metavar='N', help='dataloader threads')
#     parser.add_argument('--base_size', type=int, default=321,
#                         help='base image size')
#     parser.add_argument('--crop_size', type=int, default=321,
#                         help='crop image size')
#     parser.add_argument('--resize', type=int, default=512,
#                         help='resize image size')
#     parser.add_argument('--sync-bn', type=bool, default=None,
#                         help='whether to use sync bn (default: auto)')
#     parser.add_argument('--freeze-bn', type=bool, default=False,
#                         help='whether to freeze bn parameters (default: False)')
#     parser.add_argument('--loss-type', type=str, default='ce',
#                         choices=['ce', 'focal'],
#                         help='loss func type (default: ce)')
#     # training hyper params
#     parser.add_argument('--epochs', type=int, default=None, metavar='N',
#                         help='number of epochs to train (default: auto)')
#     parser.add_argument('--start_epoch', type=int, default=0,
#                         metavar='N', help='start epochs (default:0)')
#     parser.add_argument('--filter_multiplier', type=int, default=8)
#     parser.add_argument('--block_multiplier', type=int, default=5)
#     parser.add_argument('--step', type=int, default=5)
#     parser.add_argument('--alpha_epoch', type=int, default=20,
#                         metavar='N', help='epoch to start training alphas')
#     parser.add_argument('--batch-size', type=int, default=2,
#                         metavar='N', help='input batch size for \
#                                 training (default: auto)')
#     parser.add_argument('--test-batch-size', type=int, default=None,
#                         metavar='N', help='input batch size for \
#                                 testing (default: auto)')
#     parser.add_argument('--use_balanced_weights', action='store_true', default=False,
#                         help='whether to use balanced weights (default: False)')
#     # optimizer params
#     parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
#                         help='learning rate (default: auto)')
#     parser.add_argument('--min_lr', type=float, default=0.001)
#     parser.add_argument('--arch-lr', type=float, default=3e-3, metavar='LR',
#                         help='learning rate for alpha and beta in architect searching process')
#
#     parser.add_argument('--lr-scheduler', type=str, default='cos',
#                         choices=['poly', 'step', 'cos'],
#                         help='lr scheduler mode')
#     parser.add_argument('--momentum', type=float, default=0.9,
#                         metavar='M', help='momentum (default: 0.9)')
#     parser.add_argument('--weight-decay', type=float, default=3e-4,
#                         metavar='M', help='w-decay (default: 5e-4)')
#     parser.add_argument('--arch-weight-decay', type=float, default=1e-3,
#                         metavar='M', help='w-decay (default: 5e-4)')
#
#     parser.add_argument('--nesterov', action='store_true', default=False,
#                         help='whether use nesterov (default: False)')
#     # cuda, seed and logging
#     parser.add_argument('--no-cuda', action='store_true',
#                         default=False, help='disables CUDA training')
#     parser.add_argument('--gpu-ids', type=str, default='0',
#                         help='use which gpu to train, must be a \
#                         comma-separated list of integers only (default=0)')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     # checking point
#     parser.add_argument('--resume', type=str, default=None,
#                         help='put the path to resuming file if needed')
#     parser.add_argument('--checkname', type=str, default=None,
#                         help='set the checkpoint name')
#     # finetuning pre-trained models
#     parser.add_argument('--ft', action='store_true', default=False,
#                         help='finetuning on a different dataset')
#     # evaluation option
#     parser.add_argument('--eval-interval', type=int, default=1,
#                         help='evaluuation interval (default: 1)')
#     parser.add_argument('--no-val', action='store_true', default=False,
#                         help='skip validation during training')
#     return parser.parse_args()
#
#
# def obtain_default_train_args():
#     parser = argparse.ArgumentParser(
#         description="PyTorch DeeplabV3Plus Training")
#     parser.add_argument('--backbone', type=str, default='resnet',
#                         choices=['resnet', 'xception', 'drn', 'mobilenet'],
#                         help='backbone name (default: resnet)')
#     parser.add_argument('--out-stride', type=int, default=16,
#                         help='network output stride (default: 8)')
#     parser.add_argument('--dataset', type=str, default='pascal',
#                         choices=['pascal', 'coco', 'cityscapes'],
#                         help='dataset name (default: pascal)')
#     parser.add_argument('--use-sbd', action='store_true', default=True,
#                         help='whether to use SBD dataset (default: True)')
#     parser.add_argument('--workers', type=int, default=4,
#                         metavar='N', help='dataloader threads')
#     parser.add_argument('--base-size', type=int, default=513,
#                         help='base image size')
#     parser.add_argument('--crop-size', type=int, default=513,
#                         help='crop image size')
#     parser.add_argument('--sync-bn', type=bool, default=None,
#                         help='whether to use sync bn (default: auto)')
#     parser.add_argument('--freeze-bn', type=bool, default=False,
#                         help='whether to freeze bn parameters (default: False)')
#     parser.add_argument('--loss-type', type=str, default='ce',
#                         choices=['ce', 'focal'],
#                         help='loss func type (default: ce)')
#     # training hyper params
#     parser.add_argument('--epochs', type=int, default=None, metavar='N',
#                         help='number of epochs to train (default: auto)')
#     parser.add_argument('--start_epoch', type=int, default=0,
#                         metavar='N', help='start epochs (default:0)')
#     parser.add_argument('--batch-size', type=int, default=None,
#                         metavar='N', help='input batch size for \
#                                 training (default: auto)')
#     parser.add_argument('--test-batch-size', type=int, default=None,
#                         metavar='N', help='input batch size for \
#                                 testing (default: auto)')
#     parser.add_argument('--use-balanced-weights', action='store_true', default=False,
#                         help='whether to use balanced weights (default: False)')
#     # optimizer params
#     parser.add_argument('--lr', type=float, default=None, metavar='LR',
#                         help='learning rate (default: auto)')
#     parser.add_argument('--lr-scheduler', type=str, default='poly',
#                         choices=['poly', 'step', 'cos'],
#                         help='lr scheduler mode: (default: poly)')
#     parser.add_argument('--momentum', type=float, default=0.9,
#                         metavar='M', help='momentum (default: 0.9)')
#     parser.add_argument('--weight-decay', type=float, default=5e-4,
#                         metavar='M', help='w-decay (default: 5e-4)')
#     parser.add_argument('--nesterov', action='store_true', default=False,
#                         help='whether use nesterov (default: False)')
#     # cuda, seed and logging
#     parser.add_argument('--no-cuda', action='store_true',
#                         default=False, help='disables CUDA training')
#     parser.add_argument('--gpu-ids', type=str, default='0',
#                         help='use which gpu to train, must be a \
#                         comma-separated list of integers only (default=0)')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     # checking point
#     parser.add_argument('--resume', type=str, default=None,
#                         help='put the path to resuming file if needed')
#     parser.add_argument('--checkname', type=str, default=None,
#                         help='set the checkpoint name')
#     # finetuning pre-trained models
#     parser.add_argument('--ft', action='store_true', default=False,
#                         help='finetuning on a different dataset')
#     # evaluation option
#     parser.add_argument('--eval-interval', type=int, default=1,
#                         help='evaluation interval (default: 1)')
#     parser.add_argument('--no-val', action='store_true', default=False,
#                         help='skip validation during training')
#     parser.add_argument('--filter_multiplier', type=int,
#                         default=32, help='F in paper')
#     parser.add_argument('--steps', type=int, default=5, help='B in paper')
#     parser.add_argument('--down_sample_level', type=int,
#                         default=8, help='s in paper')
#     return parser.parse_args()
#
#
# if __name__ == "__main__":
#     import torch
#     import time
#     args = obtain_default_search_args()
#     criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
#     model = AutoDeeplab(num_classes=19, num_layers=12, criterion=criterion, filter_multiplier=args.filter_multiplier,
#                         block_multiplier=args.block_multiplier, step=args.step)
#     model = nn.DataParallel(model).cuda()
#     # torch.save(model.state_dict(), './checkpoint.pts.tar')
#     checkpoint = torch.load('./checkpoint.pts.tar')
#     st = time.time()
#     # copy_state_dict(model.state_dict(), checkpoint)
#     model.load_state_dict(checkpoint)
#     et = time.time()
#     print(et-st)




import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.datasets.cityscapes import CityscapesSegmentation


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.resize = 513
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='retrain')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
