import os
import numpy as np
from tqdm import tqdm
from mypath import Path
import torch.optim as optim
import torch.distributed as dist
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import build_criterion
from utils.calculate_weights import calculate_weigths_labels
from utils.step_lr_scheduler import Iter_LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args


class Trainer(object):

    def __init__(self, args):

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:{}'.format(args.port),
            world_size=torch.cuda.device_count(),
            rank=args.local_rank
        )
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir, use_dist=True)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.sampler, self.val_loader, self.test_loader, self.nclass = make_data_loader(args,
                                                                                                           **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        # Define Optimizer
        optimizer = optim.SGD(model.parameters(), momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.args.weight = weight
        if self.args.criterion == 'Ohem':
            args.thresh = 0.7
            args.n_min = (self.args.batch_size / len(args.gpu_ids)) * args.crop_size[0] * args.crop_size[1] // 16
        self.criterion = build_criterion(args)
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = Iter_LR_Scheduler(args.lr_scheduler, args.lr, args.max_iteration, len(self.train_loader))

        # Using cuda
        if args.dist:
            self.model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank, ],
                                                             output_device=args.local_rank).cuda()
        elif args.cuda:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
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

    def training(self, max_iteration):
        train_loss = 0.0
        self.model.train()
        epoch = 0
        i = 0
        iter_sample = iter(self.train_loader)

        for iteration in range(max_iteration):
            try:
                sample = next(iter_sample)
                i += 1
            except StopIteration:
                epoch += 1
                self.sampler.set_epoch(epoch)
                diter = iter(self.train_loader)
                sample = next(diter)
                i = 0
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, iteration)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            if dist.get_rank() == 0 and iteration % 10000 == 0:
                self.writer.add_scalar('train/total_loss_iter', loss.item(), iteration)
                print('Train loss: %.3f \n' % (train_loss / (iteration + 1)))

            if iteration % 1000 == 0:
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, iteration)
            # Show 10 * 3 inference results each epoch

        if dist.get_rank() == 0:
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.3f' % train_loss)

        if (iteration < 490000 and iteration % 10000 == 0) or (iteration >= 490000 and iteration % 100):
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    args = obtain_retrain_autodeeplab_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)

    trainer.writer.close()


if __name__ == '__main__':
    main()
