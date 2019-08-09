import argparse
import os
import numpy as np
from tqdm import tqdm
import sys
import torch
from collections import OrderedDict
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from auto_deeplab import AutoDeeplab
from architect import Architect
from decoding_formulas import Decoder


class Loader(object):
    def __init__(self, args):
        self.args = args
        if self.args.dataset == 'cityscapes':
            self.nclass = 19

        self.model = AutoDeeplab(num_classes=self.nclass, num_layers=12, filter_multiplier=self.args.filter_multiplier,
                                 block_multiplier=args.block_multiplier, step=args.step)
        # Using cuda
        if args.cuda:
            if (torch.cuda.device_count() > 1 or args.load_parallel):
                self.model = torch.nn.DataParallel(self.model.cuda())
                patch_replication_callback(self.model)
            self.model = self.model.cuda()
            print('cuda finished')

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            # if the weights are wrapped in module object we have to clean it
            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

            else:
                if (torch.cuda.device_count() > 1 or args.load_parallel):
                    self.model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint['state_dict'])

        self.decoder = Decoder(self.model.alphas,
                               self.model.bottom_betas,
                               self.model.betas8,
                               self.model.betas16,
                               self.model.top_betas,
                               args.block_multiplier,
                               args.step)

    def retreive_alphas_betas(self):
        return self.model.alphas, self.model.bottom_betas, self.model.betas8, self.model.betas16, self.model.top_betas

    def decode_architecture(self):
        paths, paths_space = self.decoder.viterbi_decode()
        return paths, paths_space

    def decode_cell(self):
        genotype = self.decoder.genotype_decode()
        return genotype


def get_new_network_cell() :
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


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    load_model = Loader(args)
    result_paths, _ = load_model.decode_architecture()
    network_path = result_paths.numpy()
    genotype = load_model.decode_cell()

    print ('architecture search results:',network_path)
    print ('new cell structure:', genotype)

    dir_name = os.path.dirname(args.resume)
    network_path_filename = os.path.join(dir_name,'network_path')
    genotype_filename = os.path.join(dir_name, 'genotype')
    np.save(network_path_filename, network_path)
    np.save(genotype_filename, genotype)

    print('saved to :', dir_name)

if __name__ == '__main__' :
    get_new_network_cell()
