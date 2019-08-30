import torch.nn as nn
from modeling.backbone.resnet import ResNet101
# from lib.config_utils.basic_args import obtain_default_train_args
import argparse
from modeling.deeplab import DeepLab
from thop import profile


def total_params(model):
    params = sum(p.numel() / 1000.0 for p in model.parameters())
    print(">>> total params: {:.5f}K".format(params))
    return params


def each_param(model):
    for p in model.parameters():
        print(">>> {:.5f}K".format(p.numel()))


def obtain_default_train_args():
    parser = argparse.ArgumentParser(
        description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    return parser.parse_args()


if __name__ == "__main__":
    import torch
    # model = ResNet101(BatchNorm=nn.BatchNorm2d,
    #                   pretrained=True, output_stride=8)
    # input = torch.rand(1, 3, 128, 128)
    # output, low_level_feat = model(input)
    # print(output.size())
    # print(low_level_feat.size())
    args = obtain_default_train_args()
    model = DeepLab(num_classes=19,
                    backbone='xception',
                    output_stride=8,
                    sync_bn=False,
                    freeze_bn=False, args=args, separate=True)
    # input = torch.randn(1, 3, 1025, 2049)

    # params, flops = profile(model, inputs=(input,))
    # print(params)
    # print(flops)

    # a = total_params(model.backbone) - total_params(model.backbone.entry_flow) - total_params(model.backbone.exit_flow)
    # print(a)

    total_params(nn.Sequential(model))
    input = torch.rand(2, 3, 1025, 2049)

    params, flops = profile(model, inputs=(input,))
    print(params)
    print(flops)