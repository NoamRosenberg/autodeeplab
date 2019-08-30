import torch
import torch.nn as nn
import numpy as np
import cell_level_search
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F
from operations import *
import numpy as np


class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier,
                 cell_arch, network_arch,
                 filter_multiplier,
                 prev_downup, downup_sample):

        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier * block_multiplier
        self.C_out = filter_multiplier * block_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)

        if downup_sample == -1:
            self.preprocess = FactorizedReduce(
                self.C_prev, self.C_out, affine=False)
        elif downup_sample == 0:
            self.preprocess = ReLUConvBN(
                self.C_prev, self.C_out, 1, 1, 0, affine=False)
        elif downup_sample == 1:
            self.preprocess = FactorizedIncrease(self.C_prev, self.C_out)

        if prev_downup is not None:
            if prev_downup == -2:
                self.pre_preprocess = DoubleFactorizedReduce(
                    self.C_prev_prev, self.C_out, affine=False)
            elif prev_downup == -1:
                self.pre_preprocess = FactorizedReduce(
                    self.C_prev_prev, self.C_out, affine=False)
            elif prev_downup == 0:
                self.pre_preprocess = ReLUConvBN(
                    self.C_prev_prev, self.C_out, 1, 1, 0, affine=False)
            elif prev_downup == 1:
                self.pre_preprocess = FactorizedIncrease(
                    self.C_prev_prev, self.C_out)
            elif prev_downup == 2:
                self.pre_preprocess = DoubleFactorizedIncrease(
                    self.C_prev_prev, self.C_out)

        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()

        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out, stride=1, affine=False)
            self._ops.append(op)

        self.ReLUConvBN = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0)

    def forward(self,  prev_prev_input, prev_input):
        if prev_prev_input is not None:
            s0 = self.pre_preprocess(prev_prev_input)
        else:
            s0 = 0
        s1 = self.preprocess(prev_input)

        states = [s0, s1]
        offset = 0
        ops_index = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if branch_index in self.cell_arch[:, 0]:
                    if prev_prev_input is None and j == 0:
                        ops_index += 1
                        continue
                    new_state = self._ops[ops_index](h)
                    new_states.append(new_state)
                    ops_index += 1

            s = sum(new_states)
            offset += len(states)
            states.append(s)

        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
        return prev_input, self.ReLUConvBN(concat_feature)


class newModel (nn.Module):
    def __init__(self, network_arch, cell_arch, num_classes, num_layers, criterion=None, filter_multiplier=8, block_multiplier=5, step=5, cell=Cell):
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
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(64, C_initial, 3, stride=2, padding=1),
            nn.BatchNorm2d(C_initial),
            nn.ReLU()
        )
        #C_prev_prev = 64
        initial_fm = C_initial / self._block_multiplier
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):
            if i == 0:
                level_option = torch.sum(self.network_arch[0], dim=1)
                level = torch.argmax(level_option).item()

                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1

                _cell = cell(self._step, self._block_multiplier, -1,
                             initial_fm,
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier *
                             filter_param_dict[level],
                             None, downup_sample)

            elif i == 1:
                level_option = torch.sum(self.network_arch[i], dim=1)
                prev_level_option = torch.sum(self.network_arch[i-1], dim=1)
                level = torch.argmax(level_option).item()
                prev_level = torch.argmax(prev_level_option).item()

                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                prev_three_branch_options = torch.sum(
                    self.network_arch[i-1], dim=0)
                prev_downup_sample = torch.argmax(
                    prev_three_branch_options).item() - 1
                total_downup = prev_downup_sample + downup_sample

                _cell = cell(self._step, self._block_multiplier, initial_fm,
                             self._filter_multiplier *
                             filter_param_dict[prev_level],
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier *
                             filter_param_dict[level],
                             total_downup, downup_sample)

            else:
                level_option = torch.sum(self.network_arch[i], dim=1)
                prev_level_option = torch.sum(self.network_arch[i-1], dim=1)
                prev_prev_level_option = torch.sum(
                    self.network_arch[i-2], dim=1)
                level = torch.argmax(level_option).item()
                prev_level = torch.argmax(prev_level_option).item()
                prev_prev_level = torch.argmax(prev_prev_level_option).item()

                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                prev_three_branch_options = torch.sum(
                    self.network_arch[i-1], dim=0)
                prev_downup_sample = torch.argmax(
                    prev_three_branch_options).item() - 1
                total_downup = prev_downup_sample + downup_sample

                _cell = cell(self._step, self._block_multiplier, self._filter_multiplier * filter_param_dict[prev_prev_level],
                             self._filter_multiplier *
                             filter_param_dict[prev_level],
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier *
                             filter_param_dict[level],
                             total_downup, downup_sample)

            self.cells += [_cell]

        last_level_option = torch.sum(self.network_arch[-1], dim=1)
        last_level = torch.argmax(last_level_option).item()
        aspp_num_input_channels = self._block_multiplier * \
            self._filter_multiplier * filter_param_dict[last_level]
        atrous_rate = int(96 / (filter_param_dict[last_level] * 4))
        self.aspp = ASPP(aspp_num_input_channels, self._num_classes,
                         atrous_rate, atrous_rate)  # 96 / 4 as in the paper

    def forward(self, x):
        stem = self.stem0(x)
        stem = self.stem1(stem)
        stem = self.stem2(stem)
        two_last_inputs = (None, stem)
        for i in range(self._num_layers):
            two_last_inputs = self.cells[i](
                two_last_inputs[0], two_last_inputs[1])

        last_input = two_last_inputs[-1]
        aspp_result = self.aspp(last_input)
        upsample = nn.Upsample(
            size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result = upsample(aspp_result)

        return aspp_result


def network_layer_to_space(net_arch):
    for i, layer in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            space[0][layer][0] = 1
            prev = layer
        else:
            if layer == prev + 1:
                sample = 0
            elif layer == prev:
                sample = 1
            elif layer == prev - 1:
                sample = 2
            space1 = np.zeros((1, 4, 3))
            space1[0][layer][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = layer
    return space


def get_cell():
    cell = np.zeros((10, 2))
    cell[0] = [0, 7]
    cell[1] = [1, 4]
    cell[2] = [2, 4]
    cell[3] = [3, 6]
    cell[4] = [5, 4]
    cell[5] = [8, 4]
    cell[6] = [11, 5]
    cell[7] = [13, 5]
    cell[8] = [19, 7]
    cell[9] = [18, 5]
    return cell.astype('uint8')


def get_arch():

    backbone = [0, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
    network_arch = network_layer_to_space(backbone)
    cell_arch = get_cell()

    return network_arch, cell_arch


def get_default_net(filter_multiplier=8):
    net_arch, cell_arch = get_arch()
    return newModel(net_arch, cell_arch, 19, 12, filter_multiplier=filter_multiplier)
