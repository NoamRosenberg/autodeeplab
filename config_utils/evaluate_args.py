import argparse


def obtain_evaluate_args():
    parser = argparse.ArgumentParser(description='---------------------evaluate args---------------------')
    parser.add_argument('--train', action='store_true', default=False, help='training mode')
    parser.add_argument('--exp', type=str, default='bnlr7e-3', help='name of experiment')
    parser.add_argument('--gpu', type=int, default=0, help='test time gpu device id')
    parser.add_argument('--backbone', type=str, default='resnet101', help='resnet101')
    parser.add_argument('--dataset', type=str, default='cityscapes', help='pascal or cityscapes')
    parser.add_argument('--groups', type=int, default=None, help='num of groups for group normalization')
    parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--base_lr', type=float, default=0.00025, help='base learning rate')
    parser.add_argument('--last_mult', type=float, default=1.0, help='learning rate multiplier for last layers')
    parser.add_argument('--scratch', action='store_true', default=False, help='train from scratch')
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze batch normalization parameters')
    parser.add_argument('--weight_std', action='store_true', default=False, help='weight standardization')
    parser.add_argument('--beta', action='store_true', default=False, help='resnet101 beta')
    parser.add_argument('--crop_size', type=int, default=513, help='image crop size')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    return parser
