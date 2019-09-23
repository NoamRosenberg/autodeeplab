class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/data/deeplearning/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'kd':
            return '/data/deeplearning/cityscapes/'
        elif dataset == 'coco':
            return '/data/deeplearning/dataset/coco2017'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
