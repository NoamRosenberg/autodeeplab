# AutoML for Image Segmentation and Detection
This is an open-source project of AutoML for object detection & segmentation as well as semantic segmentation.

Currently this repo contains a pytorch implementation for AutoDeeplab / auto-deeplab.

## Requirements

* Pytorch version 1.1

* Python 3

* tensorboardX

* torchvision

* pycocotools

* tqdm

* numpy

* pandas

## Training

```
CUDA_VISIBLE_DEVICES=7 python train_autodeeplab.py --dataset kd --batch-size 2 --checkname alpha5epoch --resume /AutoDeeplabpath/checkpoint.pth.tar
```

## References
[1] : [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/abs/1901.02985)

[2] : [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[3] : [Some code for the project was taken from here](https://github.com/MenghaoGuo/AutoDeeplab)