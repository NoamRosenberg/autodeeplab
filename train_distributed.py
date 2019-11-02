import os
import sys
import PIL
import time
import random
import logging
import datetime
import os.path as osp

import torch
import torch.backends
import torch.nn as nn
import torch.backends.cudnn
import torch.distributed as dist

from utils.loss import OhemCELoss
from utils.utils import prepare_seed
from utils.utils import time_for_file
from evaluate_distributed import MscEval
from dataloaders import make_data_loader
from utils.logger import Logger, setup_logger
from utils.optimizer_distributed import Optimizer
from config_utils.retrain_config import config_factory
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args


def main():
    args = obtain_retrain_autodeeplab_args()
    torch.cuda.set_device(args.local_rank)
    cfg = config_factory['resnet_cityscapes']
    if not os.path.exists(cfg.respth):
        os.makedirs(cfg.respth)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(cfg.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    setup_logger(cfg.respth)
    logger = logging.getLogger()
    rand_seed = random.randint(0, args.manualSeed)
    prepare_seed(rand_seed)
    if args.local_rank == 0:
        log_string = 'seed-{}-time-{}'.format(rand_seed, time_for_file())
        train_logger = Logger(args, log_string)
        train_logger.log('Arguments : -------------------------------')
        for name, value in args._get_kwargs():
            train_logger.log('{:16} : {:}'.format(name, value))
        train_logger.log("Python  version : {}".format(sys.version.replace('\n', ' ')))
        train_logger.log("Pillow  version : {}".format(PIL.__version__))
        train_logger.log("PyTorch version : {}".format(torch.__version__))
        train_logger.log("cuDNN   version : {}".format(torch.backends.cudnn.version()))
        train_logger.log("random_seed : {}".format(rand_seed))
        if args.checkname is None:
            args.checkname = 'deeplab-' + str(args.backbone)
    # dataset
    kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
    train_loader, args.num_classes, sampler = make_data_loader(args=args, **kwargs)
    # model
    model = Retrain_Autodeeplab(args)
    model.train()
    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank, ], output_device=args.local_rank,
                                                find_unused_parameters=True).cuda()
    n_min = cfg.ims_per_gpu * cfg.crop_size[0] * cfg.crop_size[1] // 16
    criterion = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min).cuda()
    max_iteration = int(cfg.max_epoch * len(train_loader))
    #     max_iteration = int(1500000 * 4 // cfg.gpus)
    it = 0
    # optimizer
    optimizer = Optimizer(model, cfg.lr_start, cfg.momentum, cfg.weight_decay, cfg.warmup_steps,
                          cfg.warmup_start_lr, max_iteration, cfg.lr_power)
    if dist.get_rank() == 0:
        print('======optimizer launch successfully , max_iteration {:}!======='.format(max_iteration))

    # train loop
    loss_avg = []
    start_time = glob_start_time = time.time()
    # for it in range(cfg.max_iter):
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if checkpoint['iter'] is not None:
            args.train_mode = 'iter'
            start_iter = checkpoint['iter']
            n_epoch = checkpoint['epoch']
        elif checkpoint['epoch'] is not None:
            args.train_mode = 'epoch'
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'], checkpoint['iter'])

    else:
        if args.train_mode == 'iter':
            start_iter = 0
            n_epoch = 0
        elif args.train_mode == 'epoch':
            start_epoch = 0

    if args.train_mode is 'iter':

        diter = iter(train_loader)
        for it in range(start_iter, cfg.max_iter):
            try:
                sample = next(diter)
            except StopIteration:
                n_epoch += 1
                sampler.set_epoch(n_epoch)
                diter = iter(train_loader)
                sample = next(diter)

            im, lb = sample['image'].cuda(), sample['label'].cuda()
            lb = torch.squeeze(lb, 1)

            optimizer.zero_grad()
            logits = model(im)
            loss = criterion(logits, lb)
            loss.backward()
            optimizer.step()

            loss_avg.append(loss.item())
            # print training log message

            if it % cfg.msg_iter == 0 and not it == 0 and dist.get_rank() == 0:
                loss_avg = sum(loss_avg) / len(loss_avg)
                lr = optimizer.lr
                ed = time.time()
                t_intv, glob_t_intv = ed - start_time, ed - glob_start_time
                eta = int((max_iteration - it) * (glob_t_intv / it))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join(['iter: {it}/{max_iteration}', 'lr: {lr:4f}', 'loss: {loss:.4f}', 'eta: {eta}', 'time: {time:.4f}',
                                 ]).format(it=it, max_iteration=max_iteration, lr=lr, loss=loss_avg, time=t_intv, eta=eta)
                # TODO : now the logger.info will error if iter > 350000, so use print haha
                if max_iteration > 350000:
                    logger.info(msg)
                else:
                    print(msg)
                loss_avg = []
            it += 1

            if (cfg.msg_iter is not None) and (it % cfg.msg_iter == 0) and (it != 0):
                if args.verbose:
                    logger.info('evaluating the model of iter:{}'.format(it))
                    model.eval()
                    evaluator = MscEval(cfg, args)
                    mIOU, loss = evaluator(model, criteria=criterion, multi_scale=False)
                    logger.info('mIOU is: {}, loss_eval is {}'.format(mIOU, loss))

                model.cpu()
                save_name = 'iter_{}_naive_model.pth'.format(it)
                save_pth = osp.join(cfg.respth, save_name)
                state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

                checkpoint = {'state_dict': state, 'epoch': n_epoch, 'iter': it, 'optimizer': optimizer.optim.state_dict()}
                if dist.get_rank() == 0:
                    torch.save(state, save_pth)
                logger.info('model of iter {} saved to: {}'.format(it, save_pth))
                model.cuda()
                model.train()

    elif args.train_mode is 'epoch':
        for epoch in range(start_epoch, cfg.max_epoch):
            for i, sample in enumerate(train_loader):
                im = sample['image'].cuda()
                lb = sample['label'].cuda()
                lb = torch.squeeze(lb, 1)

                optimizer.zero_grad()
                logits = model(im)
                loss = criterion(logits, lb)
                loss.backward()
                optimizer.step()

                loss_avg.append(loss.item())
                # print training log message

            if i % cfg.msg_iter == 0 and not (i == 0 and epoch == 0) and dist.get_rank() == 0:
                loss_avg = sum(loss_avg) / len(loss_avg)
                lr = optimizer.lr
                ed = time.time()
                t_intv, glob_t_intv = ed - start_time, ed - glob_start_time
                eta = int((max_iteration - it) * (glob_t_intv / it))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join(['iter: {it}/{max_iteration}', 'lr: {lr:4f}', 'loss: {loss:.4f}', 'eta: {eta}', 'time: {time:.4f}',
                                 ]).format(it=it, max_iteration=max_iteration, lr=lr, loss=loss_avg, time=t_intv, eta=eta)
                logger.info(msg)
                loss_avg = []

            # save model and optimizer each epoch
            if args.verbose:
                logger.info('evaluating the model of iter:{}'.format(it))
                model.eval()
                evaluator = MscEval(cfg, args)
                mIOU, loss = evaluator(model, criteria=criterion, multi_scale=False)
                logger.info('mIOU is: {}, loss_eval is {}'.format(mIOU, loss))

            model.cpu()
            save_name = 'iter_{}_naive_model.pth'.format(it)
            save_pth = osp.join(cfg.respth, save_name)
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

            checkpoint = {'state_dict': state, 'epoch': n_epoch, 'iter': it, 'optimizer': optimizer.state_dict()}
            if dist.get_rank() == 0:
                torch.save(state, save_pth)
            logger.info('model of iter {} saved to: {}'.format(it, save_pth))
            model.cuda()
            model.train()

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
