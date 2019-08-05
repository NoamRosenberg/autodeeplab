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

    def __init__(self, steps, block_multiplier, prev_prev_fmultiplier, prev_fmultiplier, filter_multiplier, rate):

        super(Cell, self).__init__()
        self.C_prev_prev = int(prev_prev_fmultiplier * block_multiplier)
        self.C_prev = int(prev_fmultiplier * block_multiplier)
        self.C_in = block_multiplier * filter_multiplier * block_multiplier
        self.C_out = filter_multiplier * block_multiplier
        if prev_prev_fmultiplier != -1 :
            self.preprocess0 = ReLUConvBN(self.C_prev_prev, self.C_out, 1, 1, 0, affine=False)

        if rate == 2 :
            self.preprocess1 = FactorizedReduce (self.C_prev, self.C_out, affine= False)
        elif rate == 0 :
            self.preprocess1 = FactorizedIncrease (self.C_prev, self.C_out)
        else :
            self.preprocess1 = ReLUConvBN(self.C_prev, self.C_out, 1, 1, 0, affine=False)
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


    def forward(self, s0, s1, weights):
        if s0 is not None :
            s0 = self.preprocess0 (s0)
        s1 = self.preprocess1(s1)
        if s0 is not None :
            states = [s0, s1]
        else :
            states = [0, s1]
        offset = 0
        for i in range(self._steps):
            new_states = []
            for j, h in enumerate(states):
                branch_index = offset + j
                if self._ops[branch_index] is None:
                    continue
                new_state = self._ops[branch_index](h, weights[branch_index])
                new_states.append(new_state)
                #assert h!=new_state!=0
            s = sum(new_states)
            #s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)


        concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
        return  self.ReLUConvBN (concat_feature)



