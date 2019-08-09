import torch
import torch.nn as nn
import numpy as np
import cell_level_search
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F
from operations import *


class newModel (nn.Module) :
    def __init__(self, new_network, new_cell, num_classes, num_layers, criterion = None, filter_multiplier = 8, block_multiplier = 5, step = 5, cell=cell_level_search.Cell):
        super(AutoDeeplab, self).__init__()

        self.cells = nn.ModuleList()
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
        self._initialize_alphas_betas ()
        C_initial = 128
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU ()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU ()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(64, C_initial, 3, stride=2, padding=1),
            nn.BatchNorm2d(C_initial),
            nn.ReLU ()
        )
