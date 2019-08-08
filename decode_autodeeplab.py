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
from decode import Decoder

class Loader(object):

    def __init__(self, args):
        self.args = args
        if self.args.dataset == 'cityscapes':
            self.nclass = 19

        self.model = AutoDeeplab(num_classes=self.nclass, num_layers=12, criterion=self.criterion,
                            filter_multiplier=self.args.filter_multiplier)
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

    def retreive_alphas_betas(self):

        return self.model.alphas, self.model.bottom_betas, self.model.betas8, self.model.betas16, self.model.top_betas

    def decode(self):

        decoder = Decoder(self.model.bottom_betas, self.model.betas8, self.model.betas16, self.model.top_betas)
        paths, path_space = decoder.viterbi_decode()

class trainNew(object):

    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args,**kwargs)


        weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define network
        model = AutoDeeplab(num_classes=self.nclass, num_layers=12, criterion=self.criterion,
                            filter_multiplier=self.args.filter_multiplier)
        optimizer = torch.optim.SGD(
            model.weight_parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        self.model, self.optimizer = model, optimizer
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader), min_lr=args.min_lr)
        # TODO: Figure out if len(self.train_loader) should be devided by two ? in other module as well
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

            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

def main () :
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
    parser.add_argument('--workers', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=int, default=320,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=320,
                        help='crop image size')
    parser.add_argument('--resize', type=int, default=512,
                        help='resize image size')
    parser.add_argument('--filter_multiplier', type=int, default=8)

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
    loader = Loader(args)
    loader.decode()


    model = AutoDeeplab (7, 12, None)
    x = torch.tensor (torch.ones (4, 3, 224, 224))
    resultdfs = model.decode_dfs ()
    resultviterbi = model.decode_viterbi()[0]


    print (resultviterbi)
    print (model.genotype())

if __name__ == '__main__' :
    main ()
