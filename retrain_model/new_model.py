import torch
import torch.nn as nn
import numpy as np
import cell_level_search
from genotypes import PRIMITIVES
from genotypes import Genotype
import torch.nn.functional as F
import numpy as np
from operations import *
from modeling.decoder import *
from modeling.aspp import ASPP_train


class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_filter_multiplier, cell_arch, network_arch,
                 filter_multiplier, downup_sample, args=None):
        super(Cell, self).__init__()
        self.cell_arch = cell_arch

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        self.C_prev = int(block_multiplier * prev_filter_multiplier)
        self.C_prev_prev = int(block_multiplier * prev_prev_fmultiplier)
        self.downup_sample = downup_sample
        self.pre_preprocess = ReLUConvBN(self.C_prev_prev, self.C_out, 1, 1, 0, args.affine, args.use_ABN)
        self.preprocess = ReLUConvBN(self.C_prev, self.C_out, 1, 1, 0, args.affine, args.use_ABN)
        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()
        if downup_sample == -1:
            self.scale = 0.5
        elif downup_sample == 1:
            self.scale = 2
        for x in self.cell_arch:
            primitive = PRIMITIVES[x[1]]
            op = OPS[primitive](self.C_out, stride=1, affine=args.affine, use_ABN=args.use_ABN)
            self._ops.append(op)

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

    def forward(self, prev_prev_input, prev_input):
        s0 = prev_prev_input
        s1 = prev_input
        if self.downup_sample != 0:
            feature_size_h = self.scale_dimension(s1.shape[2], self.scale)
            feature_size_w = self.scale_dimension(s1.shape[3], self.scale)
            s1 = F.interpolate(s1, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        if (s0.shape[2] != s1.shape[2]) or (s0.shape[3] != s1.shape[3]):
            s0 = F.interpolate(s0, (s1.shape[2], s1.shape[3]),
                                            mode='bilinear', align_corners=True)

        s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
        s1 = self.preprocess(s1)

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

        return prev_input, concat_feature


class newModel(nn.Module):
    def __init__(self, network_arch, cell_arch, num_classes, num_layers, filter_multiplier=20, lock_multiplier=5, step=5, cell=Cell,
                 BatchNorm=NaiveBN, args=None):
        super(newModel, self).__init__()
        self.args = args
        self._step = step
        self.cells = nn.ModuleList()
        self.network_arch = torch.from_numpy(network_arch)
        self.cell_arch = torch.from_numpy(cell_arch)
        self._num_layers = num_layers
        self._num_classes = num_classes
        self._block_multiplier = args.block_multiplier
        self._filter_multiplier = args.filter_multiplier
        self.use_ABN = args.use_ABN
        initial_fm = 128 if args.initial_fm is None else args.initial_fm
        half_initial_fm = initial_fm // 2
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, half_initial_fm, 3, stride=2, padding=1),
            BatchNorm(half_initial_fm)
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_initial_fm, half_initial_fm, 3, padding=1),
            BatchNorm(half_initial_fm)
        )
        # TODO: first two channels should be set automatically
        ini_initial_fm = half_initial_fm
        self.stem2 = nn.Sequential(
            nn.Conv2d(half_initial_fm, initial_fm, 3, stride=2, padding=1),
            BatchNorm(initial_fm)
        )
        # C_prev_prev = 64
        filter_param_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        for i in range(self._num_layers):
            level_option = torch.sum(self.network_arch[i], dim=1)
            prev_level_option = torch.sum(self.network_arch[i - 1], dim=1)
            prev_prev_level_option = torch.sum(self.network_arch[i - 2], dim=1)
            level = torch.argmax(level_option).item()
            prev_level = torch.argmax(prev_level_option).item()
            prev_prev_level = torch.argmax(prev_prev_level_option).item()
            if i == 0:
                downup_sample = - torch.argmax(torch.sum(self.network_arch[0], dim=1))
                _cell = cell(self._step, self._block_multiplier, ini_initial_fm / args.block_multiplier,
                             initial_fm / args.block_multiplier,
                             self.cell_arch, self.network_arch[i],
                             self._filter_multiplier *
                             filter_param_dict[level],
                             downup_sample, self.args)
            else:
                three_branch_options = torch.sum(self.network_arch[i], dim=0)
                downup_sample = torch.argmax(three_branch_options).item() - 1
                if i == 1:
                    _cell = cell(self._step, self._block_multiplier,
                                 initial_fm / args.block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier *
                                 filter_param_dict[level],
                                 downup_sample, self.args)
                else:
                    _cell = cell(self._step, self._block_multiplier,
                                 self._filter_multiplier * filter_param_dict[prev_prev_level],
                                 self._filter_multiplier *
                                 filter_param_dict[prev_level],
                                 self.cell_arch, self.network_arch[i],
                                 self._filter_multiplier *
                                 filter_param_dict[level], downup_sample, self.args)

            self.cells += [_cell]

    def forward(self, x):
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)
        two_last_inputs = (stem0, stem1)
        for i in range(self._num_layers):
            two_last_inputs = self.cells[i](two_last_inputs[0], two_last_inputs[1])
            if i == 2:
                low_level_feature = two_last_inputs[1]
        last_output = two_last_inputs[-1]
        # else:
        return last_output, low_level_feature

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                bn_params.append(param)
        return bn_params, non_bn_params


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
    """
        return:
        network_space[layer][level][sample]:
        layer: 0 - 12
        level: sample_level {0: 1, 1: 2, 2: 4, 3: 8}
        sample: 0: down 1: None 2: Up
    """

    return space


def get_default_cell():
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


def get_default_arch():
    backbone = [1, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
    cell_arch = get_default_cell()
    return network_layer_to_space(backbone), cell_arch, backbone


def get_default_net(args=None):
    filter_multiplier = args.filter_multiplier if args is not None else 20
    path_arch, cell_arch, backbone = get_default_arch()
    return newModel(path_arch, cell_arch, 19, 12, filter_multiplier=filter_multiplier, args=args)