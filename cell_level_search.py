import torch.nn.functional as F
from operations import *
from genotypes import PRIMITIVES


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False, False)
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

        self.C_in = block_multiplier * filter_multiplier
        self.C_out = filter_multiplier

        self.C_prev_prev = int(prev_prev_fmultiplier * block_multiplier)
        self._prev_fmultiplier_same = prev_fmultiplier_same

        if prev_fmultiplier_down is not None:
            self.C_prev_down = int(prev_fmultiplier_down * block_multiplier)
            self.preprocess_down = ReLUConvBN(
                self.C_prev_down, self.C_out, 1, 1, 0, affine=False)
        if prev_fmultiplier_same is not None:
            self.C_prev_same = int(prev_fmultiplier_same * block_multiplier)
            self.preprocess_same = ReLUConvBN(
                self.C_prev_same, self.C_out, 1, 1, 0, affine=False)
        if prev_fmultiplier_up is not None:
            self.C_prev_up = int(prev_fmultiplier_up * block_multiplier)
            self.preprocess_up = ReLUConvBN(
                self.C_prev_up, self.C_out, 1, 1, 0, affine=False)

        if prev_prev_fmultiplier != -1:
            self.pre_preprocess = ReLUConvBN(
                self.C_prev_prev, self.C_out, 1, 1, 0, affine=False)

        self._steps = steps
        self.block_multiplier = block_multiplier
        self._ops = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                if prev_prev_fmultiplier == -1 and j == 0:
                    op = None
                else:
                    op = MixedOp(self.C_out, stride)
                self._ops.append(op)

        # self.ReLUConvBN = ReLUConvBN(self.C_in, self.C_out, 1, 1, 0)

        self._initialize_weights()

    def scale_dimension(self, dim, scale):
        assert isinstance(dim, int)
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)

    def prev_feature_resize(self, prev_feature, mode):
        if mode == 'down':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 0.5)
        elif mode == 'up':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 2)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 2)

        return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear', align_corners=True)

    def forward(self, s0, s1_down, s1_same, s1_up, n_alphas):

        if s1_down is not None:
            s1_down = self.prev_feature_resize(s1_down, 'down')
            s1_down = self.preprocess_down(s1_down)
            size_h, size_w = s1_down.shape[2], s1_down.shape[3]
        if s1_same is not None:
            s1_same = self.preprocess_same(s1_same)
            size_h, size_w = s1_same.shape[2], s1_same.shape[3]
        if s1_up is not None:
            s1_up = self.prev_feature_resize(s1_up, 'up')
            s1_up = self.preprocess_up(s1_up)
            size_h, size_w = s1_up.shape[2], s1_up.shape[3]
        all_states = []
        if s0 is not None:
            # s0 = self.pre_preprocess(s0)
            s0 = F.interpolate(s0, (size_h, size_w), mode='bilinear', align_corners=True) if (s0.shape[2] != size_h) or (s0.shape[3] != size_w) else s0
            s0 = self.pre_preprocess(s0) if (s0.shape[1] != self.C_out) else s0
            if s1_down is not None:
                states_down = [s0, s1_down]
                all_states.append(states_down)
            if s1_same is not None:
                states_same = [s0, s1_same]
                all_states.append(states_same)
            if s1_up is not None:
                states_up = [s0, s1_up]
                all_states.append(states_up)
        else:
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
                    new_state = self._ops[branch_index](
                        h, n_alphas[branch_index])
                    new_states.append(new_state)

                s = sum(new_states)
                offset += len(states)
                states.append(s)

            concat_feature = torch.cat(states[-self.block_multiplier:], dim=1)
            final_concates.append(concat_feature)
        return final_concates


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
