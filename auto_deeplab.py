import torch
import torch.nn as nn
import numpy as np
import cell_level_search
from genotypes import PRIMITIVES
import torch.nn.functional as F
from operations import *
from decoding_formulas import Decoder

class AutoDeeplab (nn.Module) :
    def __init__(self, num_classes, num_layers, criterion = None, filter_multiplier = 8, block_multiplier = 5, step = 5, cell=cell_level_search.Cell):
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

        #C_prev_prev = 64
        intitial_fm = C_initial / self._block_multiplier

        ### init the self.cells array

        # layer == 0:

        self.cells += [cell (self._step, self._block_multiplier, -1,
                             None, intitial_fm, None, self._filter_multiplier)]
        self.cells += [cell (self._step, self._block_multiplier, -1,
                             intitial_fm, None, None, self._filter_multiplier * 2)]

        # layer == 1:

        self.cells += [cell (self._step, self._block_multiplier, intitial_fm,
                             None, self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier)]
        self.cells += [cell (self._step, self._block_multiplier, -1,
                             self._filter_multiplier, self._filter_multiplier * 2, None, self._filter_multiplier * 2)]
        self.cells += [cell (self._step, self._block_multiplier, -1,
                             self._filter_multiplier * 2, None, None, self._filter_multiplier * 4)]

        # layer == 2:

        self.cells += [cell (self._step, self._block_multiplier, self._filter_multiplier,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)]
        self.cells += [cell (self._step, self._block_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)]
        self.cells += [cell (self._step, self._block_multiplier, -1,
                              self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                              self._filter_multiplier * 4)]
        self.cells += [cell (self._step, self._block_multiplier, -1,
                              self._filter_multiplier * 4, None, None,
                              self._filter_multiplier * 8)]

        # layer == 3:

        self.cells += [cell (self._step, self._block_multiplier, self._filter_multiplier,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)]
        self.cells += [cell (self._step, self._block_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)]
        self.cells += [cell (self._step, self._block_multiplier, self._filter_multiplier * 4,
                              self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                              self._filter_multiplier * 4)]
        self.cells += [cell (self._step, self._block_multiplier, -1,
                              self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                              self._filter_multiplier * 8)]

        # layer > 3:

        self.cells += [cell(self._step, self._block_multiplier, self._filter_multiplier,
                            None, self._filter_multiplier, self._filter_multiplier * 2,
                            self._filter_multiplier)]
        self.cells += [cell(self._step, self._block_multiplier, self._filter_multiplier * 2,
                            self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                            self._filter_multiplier * 2)]
        self.cells += [cell(self._step, self._block_multiplier, self._filter_multiplier * 4,
                            self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                            self._filter_multiplier * 4)]
        self.cells += [cell(self._step, self._block_multiplier, self._filter_multiplier * 8,
                            self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                            self._filter_multiplier * 8)]

        # for i in range (4, self._num_layers) :
        # def __init__(self, steps, multiplier, C_prev_prev, C_initial, C, rate) : rate = 0 , 1, 2  reduce rate


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

    def forward (self, x) :
        # self._init_level_arr (x)
        cell = cell_level_search.Cell
        C_initial = 128
        intitial_fm = C_initial / self._block_multiplier

        level_2_curr = self.stem1 (self.stem0 (x))
        level_4_curr = self.stem2 (level_2_curr)

        # del level_2_curr

        if torch.cuda.device_count() > 1:
            img_device = torch.device('cuda', x.get_device())
            normalized_alphas = F.softmax(self.alphas.to(device=img_device), dim=-1)
            normalized_bottom_betas = F.softmax(self.bottom_betas.to(device=img_device), dim=-1)
            normalized_betas8 = F.softmax(self.betas8.to(device=img_device), dim=-1)
            normalized_betas16 = F.softmax(self.betas16.to(device=img_device), dim=-1)
            normalized_top_betas = F.softmax(self.top_betas.to(device=img_device), dim=-1)
        else:
            normalized_alphas = F.softmax(self.alphas, dim=-1)
            normalized_bottom_betas = F.softmax(self.bottom_betas, dim=-1)
            normalized_betas8 = F.softmax (self.betas8, dim = -1)
            normalized_betas16 = F.softmax(self.betas16, dim=-1)
            normalized_top_betas = F.softmax(self.top_betas, dim=-1)


        # layer == 0 :

        cell_curr = cell (self._step, self._block_multiplier, -1,
                             None, intitial_fm, None, self._filter_multiplier)
        level4_new, = self.cells[0] (None, None, level_4_curr, None, normalized_alphas)
        level8_new, = self.cells[1] (None, level_4_curr, None, None, normalized_alphas)

        level_4_prev = level_4_curr
        level_4_curr = level4_new
        level_8_curr = level8_new

        # layer == 1 :

        level4_new_1, level4_new_2 = self.cells[2] (level_4_prev,
                                                        None,
                                                        level_4_curr,
                                                        level_8_curr,
                                                        normalized_alphas)

        level4_new = normalized_bottom_betas[1][0] * level4_new_1 + normalized_bottom_betas[1][1] * level4_new_2

        # del level4_new_1
        # del level4_new_2

        level8_new_1, level8_new_2 = self.cells[3] (None,
                                                        level_4_curr,
                                                        level_8_curr,
                                                        None,
                                                        normalized_alphas)

        level8_new = normalized_top_betas[1][0] * level8_new_1 + normalized_top_betas[1][1] * level8_new_2

        # del level8_new_1
        # del level8_new_2

        level16_new, = self.cells[4] (None,
                                          level_8_curr,
                                          None,
                                          None,
                                          normalized_alphas)
        # level16_new = level16_new

        level_4_prev = level_4_curr
        level_4_curr = level4_new

        level_8_prev = level_8_curr
        level_8_curr = level8_new

        level_16_curr = level16_new

        # layer == 2 :

        level4_new_1, level4_new_2 = self.cells[5] (level_4_prev,
                                                        None,
                                                        level_4_curr,
                                                        level_8_curr,
                                                        normalized_alphas)

        level4_new = normalized_bottom_betas[2][0] * level4_new_1 + normalized_bottom_betas[2][1] * level4_new_2

        # del level4_new_1
        # del level4_new_2

        level8_new_1, level8_new_2, level8_new_3 = self.cells[6] (level_8_prev,
                                                                      level_4_curr,
                                                                      level_8_curr,
                                                                      level_16_curr,
                                                                      normalized_alphas)

        level8_new = normalized_betas8[2 - 1][0] * level8_new_1 + normalized_betas8[2 - 1][1] * level8_new_2 + normalized_betas8[2 - 1][2] * level8_new_3

        # del level8_new_1
        # del level8_new_2
        # del level8_new_3

        level16_new_1, level16_new_2 = self.cells[7] (None,
                                                          level_8_curr,
                                                          level_16_curr,
                                                          None,
                                                          normalized_alphas)

        level16_new = normalized_top_betas[2][0] * level16_new_1 + normalized_top_betas[2][1] * level16_new_2

        # del level16_new_1
        # del level16_new_2

        level32_new, = self.cells[8] (None,
                                          level_16_curr,
                                          None,
                                          None,
                                          normalized_alphas)

        level_4_prev = level_4_curr
        level_4_curr = level4_new

        level_8_prev = level_8_curr
        level_8_curr = level8_new

        level_16_prev = level_16_curr
        level_16_curr = level16_new

        level_32_curr = level32_new

        # layer == 3 :

        level4_new_1, level4_new_2 = self.cells[9] (level_4_prev,
                                                        None,
                                                        level_4_curr,
                                                        level_8_curr,
                                                        normalized_alphas)

        level4_new = normalized_bottom_betas[3][0] * level4_new_1 + normalized_bottom_betas[3][1] * level4_new_2

        # del level4_new_1
        # del level4_new_2

        level8_new_1, level8_new_2, level8_new_3 = self.cells[10] (level_8_prev,
                                                                      level_4_curr,
                                                                      level_8_curr,
                                                                      level_16_curr,
                                                                      normalized_alphas)

        level8_new = normalized_betas8[3 - 1][0] * level8_new_1 + normalized_betas8[3 - 1][1] * level8_new_2 + normalized_betas8[3 - 1][2] * level8_new_3

        # del level8_new_1
        # del level8_new_2
        # del level8_new_3

        level16_new_1, level16_new_2, level16_new_3 = self.cells[11] (level_16_prev,
                                                                         level_8_curr,
                                                                         level_16_curr,
                                                                         level_32_curr,
                                                                         normalized_alphas)

        level16_new = normalized_betas16[3 - 2][0] * level16_new_1 + normalized_betas16[3 - 2][1] * level16_new_2 + normalized_betas16[3 - 2][2] * level16_new_3

        # del level16_new_1
        # del level16_new_2
        # del level16_new_3

        level32_new_1, level32_new_2 = self.cells[12] (None,
                                                          level_16_curr,
                                                          level_32_curr,
                                                          None,
                                                          normalized_alphas)

        level32_new = normalized_top_betas[3][0] * level32_new_1 + normalized_top_betas[3][1] * level32_new_2

        # del level32_new_1
        # del level32_new_2

        level_4_prev = level_4_curr
        level_4_curr = level4_new

        level_8_prev = level_8_curr
        level_8_curr = level8_new

        level_16_prev = level_16_curr
        level_16_curr = level16_new

        level_32_prev = level_32_curr
        level_32_curr = level32_new

        for layer in range(4, self._num_layers - 1):

            level4_new_1, level4_new_2 = self.cells[13] (level_4_prev,
                                              None,
                                              level_4_curr,
                                              level_8_curr,
                                              normalized_alphas)

            level4_new = normalized_bottom_betas[layer][0] * level4_new_1 + normalized_bottom_betas[layer][1] * level4_new_2

            # del level4_new_1
            # del level4_new_2

            level8_new_1, level8_new_2, level8_new_3 = self.cells[14] (level_8_prev,
                                                                          level_4_curr,
                                                                          level_8_curr,
                                                                          level_16_curr,
                                                                          normalized_alphas)

            level8_new = normalized_betas8[layer - 1][0] * level8_new_1 + normalized_betas8[layer - 1][1] * level8_new_2 + normalized_betas8[layer - 1][2] * level8_new_3

            # del level8_new_1
            # del level8_new_2
            # del level8_new_3

            level16_new_1, level16_new_2, level16_new_3 = self.cells[15] (level_16_prev,
                                                                             level_8_curr,
                                                                             level_16_curr,
                                                                             level_32_curr,
                                                                             normalized_alphas)

            level16_new = normalized_betas16[layer - 2][0] * level16_new_1 + normalized_betas16[layer - 2][1] * level16_new_2 + normalized_betas16[layer - 2][2] * level16_new_3

            # del level16_new_1
            # del level16_new_2
            # del level16_new_3

            level32_new_1, level32_new_2 = self.cells[16] (level_32_prev,
                                                              level_16_curr,
                                                              level_32_curr,
                                                              None,
                                                              normalized_alphas)

            level32_new = normalized_top_betas[layer][0] * level32_new_1 + normalized_top_betas[layer][1] * level32_new_2

            # del level32_new_1
            # del level32_new_2

            level_4_prev = level_4_curr
            level_4_curr = level4_new

            level_8_prev = level_8_curr
            level_8_curr = level8_new

            level_16_prev = level_16_curr
            level_16_curr = level16_new

            level_32_prev = level_32_curr
            level_32_curr = level32_new

        aspp_result_4 = self.aspp_4 (level_4_curr)
        aspp_result_8 = self.aspp_8 (level_8_curr)
        aspp_result_16 = self.aspp_16 (level_16_curr)
        aspp_result_32 = self.aspp_32 (level_32_curr)

        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        aspp_result_4 = upsample (aspp_result_4)
        aspp_result_8 = upsample (aspp_result_8)
        aspp_result_16 = upsample (aspp_result_16)
        aspp_result_32 = upsample (aspp_result_32)


        sum_feature_map = aspp_result_4 + aspp_result_8 + aspp_result_16 + aspp_result_32


        return sum_feature_map

    def _initialize_alphas_betas(self):
        k = sum(1 for i in range(self._step) for n in range(2+i))
        num_ops = len(PRIMITIVES)
        alphas = (1e-3 * torch.randn(k, num_ops)).cuda().clone().detach().requires_grad_(True)
        bottom_betas = (1e-3 * torch.randn(self._num_layers - 1, 2)).cuda().clone().detach().requires_grad_(True)
        betas8 = (1e-3 * torch.randn(self._num_layers - 2, 3)).cuda().clone().detach().requires_grad_(True)
        betas16 = (1e-3 * torch.randn(self._num_layers - 3, 3)).cuda().clone().detach().requires_grad_(True)
        top_betas = (1e-3 * torch.randn(self._num_layers - 1, 2)).cuda().clone().detach().requires_grad_(True)

        # alphas = torch.tensor (1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        # bottom_betas = torch.tensor (1e-3 * torch.randn(self._num_layers - 1, 2).cuda(), requires_grad=True)
        # betas8 = torch.tensor (1e-3 * torch.randn(self._num_layers - 2, 3).cuda(), requires_grad=True)
        # betas16 = torch.tensor(1e-3 * torch.randn(self._num_layers - 3, 3).cuda(), requires_grad=True)
        # top_betas = torch.tensor (1e-3 * torch.randn(self._num_layers - 1, 2).cuda(), requires_grad=True)

        self._arch_parameters = [
            alphas,
            bottom_betas,
            betas8,
            betas16,
            top_betas,
        ]
        self._arch_param_names = [
            'alphas',
            'bottom_betas',
            'betas8',
            'betas16',
            'top_betas']

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in zip(self._arch_param_names, self._arch_parameters)]

    def decode_viterbi(self):
        decoder = Decoder(self.bottom_betas, self.betas8, self.betas16, self.top_betas)
        return decoder.viterbi_decode()

    def decode_dfs(self):
        decoder = Decoder(self.bottom_betas, self.betas8, self.betas16, self.top_betas)
        return decoder.dfs_decode()

    def arch_parameters (self) :
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def genotype(self):
        decoder = Decoder(self.alphas_cell, self._block_multiplier, self._step)
        return decoder.genotype_decode()

    def _loss (self, input, target) :
        logits = self (input)
        return self._criterion (logits, target)


def main () :
    model = AutoDeeplab (7, 12, None)
    x = torch.tensor (torch.ones (4, 3, 224, 224))
    resultdfs = model.decode_dfs ()
    resultviterbi = model.decode_viterbi()[0]


    print (resultviterbi)
    print (model.genotype())

if __name__ == '__main__' :
    main ()