import os
import platform
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable

import dataloaders
from utils.utils import AverageMeter
from utils.loss import build_criterion
import retrain_model.new_model as new_model
from utils.optimizer_distributed import Optimizer
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args


def main():
    args = obtain_retrain_autodeeplab_args()
    warnings.filterwarnings('ignore')
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(args.port),
                            world_size=torch.cuda.device_count(), rank=args.local_rank)
    assert torch.cuda.is_available()
    assert not platform.platform().startswith('Win'), ValueError('Now distributed can not support system {:}'.format(platform.platform()))
    torch.backends.cudnn.benchmark = True
    model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(args.backbone, args.dataset, args.exp)
    if args.dataset == 'pascal':
        raise NotImplementedError
    elif args.dataset == 'cityscapes':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone == 'autodeeplab':
        if args.net_arch is not None and args.cell_arch is not None:
            net_arch, cell_arch = np.load(args.net_arch), np.load(args.cell_arch)
        else:
            network_arch, cell_arch = new_model.get_default_arch()
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu)) * args.crop_size[0] * args.crop_size[1] // 16)
    criterion = build_criterion(args)

    max_iteration = len(dataset_loader) * args.epochs

    optimizer = Optimizer(model, args, max_iteration=max_iteration)
    if args.resume:
        if os.path.isfile(args.resume):
            if dist.get_rank() == 0:
                print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if dist.get_rank() == 0:
                print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))
    else:
        start_epoch = 0
    model = nn.parallel.DistributedDataParallel(model.train().cuda(), device_ids=[args.local_rank, ], output_device=args.local_rank)

    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        for i, sample in enumerate(dataset_loader):
            cur_iter = epoch * len(dataset_loader) + i
            inputs = Variable(sample['image'].cuda())
            target = Variable(sample['label'].cuda())
            outputs = model(inputs)
            loss = criterion(outputs, target)
            losses.update(loss.item(), args.batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if dist.get_rank() == 0:
                print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                    epoch + 1, i + 1, len(dataset_loader), Optimizer.get_lr(optimizer), loss=losses))

        if dist.get_rank() == 0:
            if epoch < args.epochs - 50:
                if epoch % 50 == 0:
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                    }, model_fname % (epoch + 1))
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                }, model_fname % (epoch + 1))

            print('reset local total loss!')
if __name__ == "__main__":
    main()
