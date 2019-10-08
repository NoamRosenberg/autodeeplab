# import torch.nn as nn
# # from modeling.backbone.resnet import ResNet101
# from auto_deeplab import AutoDeeplab
# import warnings
# from config_utils.search_args import obtain_search_args
# from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
#
# #
# # warnings.filterwarnings('ignore')
# # from retrain_model.build_autodeeplab import Train_Autodeeplab
# #
# #
# # def total_params(model, log=True):
# #     params = sum(p.numel() / 1000.0 for p in model.parameters())
# #     if log:
# #         print(">>> total params: {:.2f}K".format(params))
# #     return params
# #
# #
# # def each_param(model):
# #     for p in model.parameters():
# #         print(">>> {:.5f}K".format(p.numel()))
# #
# #
# import torch
# from retrain_model.new_model import get_default_net
# from retrain_model.build_autodeeplab import Retrain_Autodeeplab
# from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
#
# args = obtain_retrain_autodeeplab_args()
# args.num_classes = 19
# model = Retrain_Autodeeplab(args)
#
# x = torch.randn(2, 3, 129, 129)
# x = model(x)
# print(x.shape)


import numpy as np
from dataloaders.datasets.cityscapes import CityscapesSegmentation

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib
    import matplotlib.pyplot as plt
    import argparse
    from config_utils.search_args import obtain_search_args
    from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

    matplotlib.use('TkAgg')
    parser = argparse.ArgumentParser()
    args = obtain_retrain_autodeeplab_args()
    args.resize = 513
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CityscapesSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=False, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            print(gt)
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
