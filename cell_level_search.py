import torch
import torch.nn as nn
import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp (nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier,
                 prev_fmultiplier_down, prev_fmultiplier_same, prev_fmultiplier_up,
                 filter_multiplier):

        super(Cell, self).__init__()
        # self.C_in = block_multiplier * filter_multiplier * block_multiplier
        # self.C_out = filter_multiplier * block_multiplier
        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier
        # self.C_prev_prev = int(prev_prev_fmultiplier * block_multiplier)
        self.C_prev_prev = int(prev_prev_fmultiplier)
        if prev_fmultiplier_down is not None:
            # self.C_prev_down = int(prev_fmultiplier_down * block_multiplier)
            self.C_prev_down = prev_fmultiplier_down
            self.preprocess_down = FactorizedReduce(self.C_prev_down, self.C_out, affine=False)
        if prev_fmultiplier_same is not None:
            # self.C_prev_same = int(prev_fmultiplier_same * block_multiplier)
            self.C_prev_same = prev_fmultiplier_same
            self.preprocess_same = ReLUConvBN(self.C_prev_same, self.C_out, 1, 1, 0, affine=False)
        if prev_fmultiplier_up is not None:
            # self.C_prev_up = int(prev_fmultiplier_up * block_multiplier)
            self.C_prev_up = prev_fmultiplier_up
            self.preprocess_up = FactorizedIncrease(self.C_prev_up, self.C_out)


        if prev_prev_fmultiplier != -1 :
            self.pre_preprocess = ReLUConvBN(self.C_prev_prev, self.C_out, 1, 1, 0, affine=False)

        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2+i):
                stride = 1
                if prev_prev_fmultiplier == -1 and j==0:
                    op = None
                else:
                    op = MixedOp(self.C_out, stride)
                self._ops.append(op)


        self.ReLUConvBN = ReLUConvBN (self.C_in, self.C_out, 1, 1, 0)


    def forward(self, s0, s1_down, s1_same, s1_up, n_alphas):

        if s1_down is not None:
            s1_down = self.preprocess_down(s1_down)
        if s1_same is not None:
            s1_same = self.preprocess_same(s1_same)
        if s1_up is not None:
            s1_up = self.preprocess_up(s1_up)
        all_states = []
        if s0 is not None :
            s0 = self.pre_preprocess(s0)
            if s1_down is not None:
                states_down = [s0, s1_down]
                all_states.append(states_down)
            if s1_same is not None:
                states_same = [s0, s1_same]
                all_states.append(states_same)
            if s1_up is not None:
                states_up = [s0, s1_up]
                all_states.append(states_up)
        else :
            if s1_down is not None:
                states_down = [0, s1_down]
                all_states.append(states_down)
            if s1_same is not None:
                states_same = [0, s1_same]
                all_states.append(states_same)
            if s1_up is not None:
                states_up = [0, s1_up]
                all_states.append(states_up)


        final_concates = []
        for states in all_states:
            offset = 0
            for i in range(self._steps):
                new_states = []
                for j, h in enumerate(states):
                    branch_index = offset + j
                    if self._ops[branch_index] is None:
                        continue
                    new_state = self._ops[branch_index](h, n_alphas[branch_index])
                    new_states.append(new_state)

                s = sum(new_states)
                offset += len(states)
                states.append(s)


            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            final_concates.append(self.ReLUConvBN (concat_feature))
        return  final_concates



