import torch
import torch.nn as nn
import numpy as np
import cell_level_search
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F
from operations import *

class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier,
                 cell_arch, network_arch,
                 filter_multiplier,
                 prev_downup, downup_sample):

        super(Cell, self).__init__()


        self.C_in = block_multiplier * filter_multiplier * block_multiplier
        self.C_out = filter_multiplier * block_multiplier
        self.C_prev = block_multiplier * prev_filter_multiplier
        self.C_prev_prev = block_multiplier * prev_prev_fmultiplier

        if downup_sample == -1:
            self.preprocess_down = FactorizedReduce(self.C_prev, self.C_out, affine=False)
        elif downup_sample == 0:
            self.preprocess_same = ReLUConvBN(self.C_prev, self.C_out, 1, 1, 0, affine=False)
        elif downup_sample == 1:
            self.preprocess_up = FactorizedIncrease(self.C_prev, self.C_out)

        if prev_downup is not None:
            if prev_downup == -2:
                self.pre_preprocess = DoubleFactorizedReduce(self.C_prev_prev, self.C_out, affine=False)
            elif prev_downup == -1:
                self.pre_preprocess = FactorizedReduce(self.C_prev_prev, self.C_out, affine=False)
            elif prev_downup == 0:
                self.pre_preprocess = ReLUConvBN(self.C_prev_prev, self.C_out, 1, 1, 0, affine=False)
            elif prev_downup == 1:
                self.pre_preprocess = FactorizedIncrease(self.C_prev_prev, self.C_out)
            elif prev_downup == 2:
                self.pre_preprocess = DoubleFactorizedIncrease(self.C_prev_prev, self.C_out)

        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()

        for x in cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out,stride=1,affine=False)
            self._ops.append(op)

        self.ReLUConvBN = ReLUConvBN (self.C_in, self.C_out, 1, 1, 0)

    def forward(self):


class newModel (nn.Module) :
    def __init__(self, network_arch, cell_arch, num_classes, num_layers, criterion = None, filter_multiplier = 8, block_multiplier = 5, step = 5, cell=Cell):
        super(newModel, self).__init__()

        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._step = step
        self._block_multiplier = block_multiplier
        self._filter_multiplier = filter_multiplier
        self._criterion = criterion
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
        #C_prev_prev = 64
        initial_fm = C_initial / self._block_multiplier
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range (self._num_layers) :
            if i==0:
                level_option = torch.sum(self.network_arch[0], dim=1)
                level = torch.argmax(level_option).item()

                three_branch_options = torch.sum(network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1

                _cell = cell (self._step, self._block_multiplier, -1,
                              initial_fm,
                              self.cell_arch, self.network_arch[i],
                              self._filter_multiplier * filter_param_dict[level],
                              None, downup_sample)


            elif i==1:
                level_option = torch.sum(network_arch[i], dim=1)
                prev_level_option = torch.sum(network_arch[i-1], dim=1)
                level = torch.argmax(level_option).item()
                prev_level = torch.argmax(prev_level_option).item()

                three_branch_options = torch.sum(network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                prev_three_branch_options = torch.sum(network_arch[i-1], dim=0)
                prev_downup_sample = torch.argmax(prev_three_branch_options).item() - 1
                total_downup = prev_downup_sample + downup_sample

                _cell = cell(self._step, self._block_multiplier, initial_fm,
                             self._filter_multiplier * filter_param_dict[prev_level],
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier * filter_param_dict[level],
                             total_downup, downup_sample)


            else:
                level_option = torch.sum(network_arch[i], dim=1)
                prev_level_option = torch.sum(network_arch[i-1], dim=1)
                prev_prev_level_option = torch.sum(network_arch[i-2], dim=1)
                level = torch.argmax(level_option).item()
                prev_level = torch.argmax(prev_level_option).item()
                prev_prev_level = torch.argmax(prev_prev_level_option).item()

                three_branch_options = torch.sum(network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                prev_three_branch_options = torch.sum(network_arch[i-1], dim=0)
                prev_downup_sample = torch.argmax(prev_three_branch_options).item() - 1
                total_downup = prev_downup_sample + downup_sample

                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier * filter_param_dict[prev_prev_level],
                             self._filter_multiplier * filter_param_dict[prev_level],
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier * filter_param_dict[level],
                             total_downup, downup_sample)

            self.cells += [_cell]

        last_level_option = torch.sum(network_arch[-1], dim=1)
        last_level = torch.argmax(last_level_option).item()
        aspp_input_channels
        self.aspp_4 = nn.Sequential (
            ASPP (self._block_multiplier * self._filter_multiplier, self._num_classes, 24, 24) #96 / 4 as in the paper
        )
        self.aspp_8 = nn.Sequential (
            ASPP (self._block_multiplier * self._filter_multiplier * 2, self._num_classes, 12, 12) #96 / 8
        )
        self.aspp_16 = nn.Sequential (
            ASPP (self._block_multiplier * self._filter_multiplier * 4, self._num_classes, 6, 6) #96 / 16
        )
        self.aspp_32 = nn.Sequential (
            ASPP (self._block_multiplier * self._filter_multiplier * 8, self._num_classes, 3, 3) #96 / 32
        )

    def forward(self):