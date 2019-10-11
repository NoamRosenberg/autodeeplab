import os
import numpy as np
from modeling.deeplab import *
from decoding_formulas import Decoder
from config_utils.decode_args import obtain_decode_args


class Loader(object):
    def __init__(self, args):
        self.args = args
        if self.args.dataset == 'cityscapes':
            self.nclass = 19

        # Resuming checkpoint
        self.best_pred = 0.0
        assert args.resume is not None, RuntimeError("No model to decode in resume path: '{:}'".format(args.resume))
        assert os.path.isfile(args.resume), RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        self._alphas = checkpoint['state_dict']['alphas']
        self._betas = checkpoint['state_dict']['betas']

        self.decoder = Decoder(alphas=self._alphas, betas=self._betas, steps=5)

    def retreive_alphas_betas(self):
        return self._alphas, self._betas

    def decode_architecture(self):
        paths, paths_space = self.decoder.viterbi_decode()
        return paths, paths_space

    def decode_cell(self):
        genotype = self.decoder.genotype_decode()
        return genotype


def get_new_network_cell():
    args = obtain_decode_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    load_model = Loader(args)
    result_paths, result_paths_space = load_model.decode_architecture()
    network_path = result_paths
    network_path_space = result_paths_space
    genotype = load_model.decode_cell()

    print('architecture search results:', network_path)
    print('new cell structure:', genotype)

    dir_name = os.path.dirname(args.resume)
    network_path_filename = os.path.join(dir_name, 'network_path')
    network_path_space_filename = os.path.join(dir_name, 'network_path_space')
    genotype_filename = os.path.join(dir_name, 'genotype')
    np.save(network_path_filename, network_path)
    np.save(network_path_space_filename, network_path_space)
    np.save(genotype_filename, genotype)

    print('saved to :', dir_name)


if __name__ == '__main__':
    get_new_network_cell()
