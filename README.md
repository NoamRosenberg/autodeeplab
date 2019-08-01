# AutoML for Image Segmentation and Detection
This is an open-source project of AutoML for object detection & segmentation as well as semantic segmentation.

Currently this repo contains a pytorch implementation for [Auto-Deeplab](https://arxiv.org/abs/1901.02985). 

Since I currently don't have big enough GPUs, a smaller model is now currently undergoing testing, i.e. image input size of 224 instead of 321, and a filter_multiplier of 4 instead of 8. 
If anyone can donate any gpu resources, or has the resources to test this code according to the hyper-parameters mentioned in the paper, I would greatly appreciate it.

Following the popular trend of modern CNN architectures having a two level hierarchy. Auto-Deeplab forms a dual level search space, searching for optimal network and cell architecture.
![network and cell level search space](./images/networkandcell.png)




Auto-Deeplab acheives a better performance while minimizing the size of the final model.
![model results](./images/results.png)



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

**Start training**
```
CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py --dataset cityscapes
```

**Resume training**
```
CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py --dataset cityscapes --resume /AutoDeeplabpath/checkpoint.pth.tar
```

**Multi-GPU training**
```
CUDA_VISIBLE_DEVICES=0,1 python train_autodeeplab.py --dataset cityscapes --batch_size 2
```

## References
[1] : [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/abs/1901.02985)

[2] : [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[3] : [Some code for the project was taken from here](https://github.com/MenghaoGuo/AutoDeeplab)