import argparse

def obtain_decode_args():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes', 'kd'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--autodeeplab', type=str, default='train',
                        choices=['search', 'train'])
    parser.add_argument('--load-parallel', type=int, default=0)
    parser.add_argument('--clean-module', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=320,
                        help='crop image size')
    parser.add_argument('--resize', type=int, default=512,
                        help='resize image size')
    parser.add_argument('--filter_multiplier', type=int, default=8)
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')

    return parser.parse_args()