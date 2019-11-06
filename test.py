from torch.utils.data.dataloader import DataLoader
from dataloaders.datasets.cityscapes import CityscapesSegmentation
from config_utils.search_args import obtain_search_args
from utils.loss import SegmentationLosses
import torch
import numpy as np
from auto_deeplab import AutoDeeplab

model = AutoDeeplab(19, 12).cuda()


# a = torch.randn(2,3,65,65).cuda()

# b = model(a)

# # print(b)


args = obtain_search_args()


args.cuda = True
criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)


args.crop_size = 64

dataset = CityscapesSegmentation(args, r'E:\BaiduNetdiskDownload\cityscapes', 'train')


loader = DataLoader(dataset, batch_size=4)

for i, sample in enumerate(loader):
    image, label = sample['image'].cuda(), sample['label'].cuda()
    print(image.shape)
    print(label.shape)

    prediction = model(image)

    print(criterion(prediction, label))

    if i == 0:
        exit()