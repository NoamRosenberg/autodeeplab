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
# from config_utils.search_args import obtain_search_args
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
# if __name__ == "__main__":
#     import torch
#     import time
#
#     args = obtain_search_args()
#     criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
#     model = AutoDeeplab(num_classes=19, num_layers=12, criterion=criterion, filter_multiplier=args.filter_multiplier,
#                         block_multiplier=args.block_multiplier, step=args.step, args=args).cuda()
#     input = torch.randn(2, 3, 65, 65).cuda()
#     output = model(input)
#     print(output)



if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    import numpy as np
    from dataloaders.datasets.cityscapes import CityscapesSegmentation


    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.resize = 513
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='test')

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
