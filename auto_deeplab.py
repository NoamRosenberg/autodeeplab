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
        C_initial = self._filter_multiplier *  self._block_multiplier
        half_C_initial = int(C_initial / 2)

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, half_C_initial, 3, stride=2, padding=1),
            nn.BatchNorm2d(half_C_initial),
            nn.ReLU ()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_C_initial, half_C_initial, 3, stride=1, padding=1),
            nn.BatchNorm2d(half_C_initial),
            nn.ReLU ()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(half_C_initial, C_initial, 3, stride=2, padding=1),
            nn.BatchNorm2d(C_initial),
            nn.ReLU ()
        )


        intitial_fm = C_initial / self._block_multiplier
        for i in range (self._num_layers) :


            if i == 0 :
                cell1 = cell (self._step, self._block_multiplier, -1,
                              None, intitial_fm, None,
                              self._filter_multiplier)
                cell2 = cell (self._step, self._block_multiplier, -1,
                              intitial_fm, None, None,
                              self._filter_multiplier * 2)
                self.cells += [cell1]
                self.cells += [cell2]
            elif i == 1 :
                cell1 = cell (self._step, self._block_multiplier, intitial_fm,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)

                cell2 = cell (self._step, self._block_multiplier, -1,
                              self._filter_multiplier, self._filter_multiplier * 2, None,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step, self._block_multiplier, -1,
                              self._filter_multiplier * 2, None, None,
                              self._filter_multiplier * 4)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]

            elif i == 2 :
                cell1 = cell (self._step, self._block_multiplier, self._filter_multiplier,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)

                cell2 = cell (self._step, self._block_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step, self._block_multiplier, -1,
                              self._filter_multiplier * 2, self._filter_multiplier * 4, None,
                              self._filter_multiplier * 4)

                cell4 = cell (self._step, self._block_multiplier, -1,
                              self._filter_multiplier * 4, None, None,
                              self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]



            elif i == 3 :
                cell1 = cell (self._step, self._block_multiplier, self._filter_multiplier,
                              None, self._filter_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier)

                cell2 = cell (self._step, self._block_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step, self._block_multiplier, self._filter_multiplier * 4,
                              self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                              self._filter_multiplier * 4)


                cell4 = cell (self._step, self._block_multiplier, -1,
                              self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                              self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

            else :
                cell1 = cell (self._step, self._block_multiplier, self._filter_multiplier,
                                None, self._filter_multiplier, self._filter_multiplier * 2,
                                self._filter_multiplier)

                cell2 = cell (self._step, self._block_multiplier, self._filter_multiplier * 2,
                              self._filter_multiplier, self._filter_multiplier * 2, self._filter_multiplier * 4,
                              self._filter_multiplier * 2)

                cell3 = cell (self._step, self._block_multiplier, self._filter_multiplier * 4,
                                self._filter_multiplier * 2, self._filter_multiplier * 4, self._filter_multiplier * 8,
                                self._filter_multiplier * 4)

                cell4 = cell (self._step, self._block_multiplier, self._filter_multiplier * 8,
                                self._filter_multiplier * 4, self._filter_multiplier * 8, None,
                                self._filter_multiplier * 8)

                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]

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
        #TODO: GET RID OF THESE LISTS, we dont need to keep everything.
        #TODO: Is this the reason for the memory issue ?

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

        count = 0

        for layer in range(self._num_layers - 1):

            if layer == 0 :

                level4_new, = self.cells[count](None, None, level_4_curr, None, normalized_alphas)
                count += 1
                level8_new, = self.cells[count](None, level_4_curr, None, None, normalized_alphas)
                count += 1

                level_4_prev = level_4_curr
                level_4_curr = level4_new
                level_8_curr = level8_new

            elif layer == 1 :

                level4_new_1, level4_new_2 = self.cells[count](level_4_prev,
                                                           None,
                                                           level_4_curr,
                                                           level_8_curr,
                                                           normalized_alphas)
                count += 1
                level4_new = normalized_bottom_betas[layer][0] * level4_new_1 + normalized_bottom_betas[layer][1] * level4_new_2

                # del level4_new_1
                # del level4_new_2

                level8_new_1, level8_new_2 = self.cells[count](None,
                                                           level_4_curr,
                                                           level_8_curr,
                                                           None,
                                                           normalized_alphas)
                count += 1
                level8_new = normalized_top_betas[layer][0] * level8_new_1 + normalized_top_betas[layer][1] * level8_new_2

                # del level8_new_1
                # del level8_new_2

                level16_new, = self.cells[count](None,
                                             level_8_curr,
                                             None,
                                             None,
                                             normalized_alphas)
                count += 1

                level_4_prev = level_4_curr
                level_4_curr = level4_new

                level_8_prev = level_8_curr
                level_8_curr = level8_new

                level_16_curr = level16_new

            elif layer == 2 :

                level4_new_1, level4_new_2 = self.cells[count](level_4_prev,
                                                           None,
                                                           level_4_curr,
                                                           level_8_curr,
                                                           normalized_alphas)
                count += 1
                level4_new = normalized_bottom_betas[layer][0] * level4_new_1 + normalized_bottom_betas[layer][1] * level4_new_2

                # del level4_new_1
                # del level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](level_8_prev,
                                                                         level_4_curr,
                                                                         level_8_curr,
                                                                         level_16_curr,
                                                                         normalized_alphas)
                count += 1
                level8_new = normalized_betas8[layer - 1][0] * level8_new_1 + normalized_betas8[layer - 1][1] * level8_new_2 + \
                             normalized_betas8[layer - 1][2] * level8_new_3

                # del level8_new_1
                # del level8_new_2
                # del level8_new_3

                level16_new_1, level16_new_2 = self.cells[count](None,
                                                             level_8_curr,
                                                             level_16_curr,
                                                             None,
                                                             normalized_alphas)
                count += 1
                level16_new = normalized_top_betas[layer][0] * level16_new_1 + normalized_top_betas[layer][1] * level16_new_2

                # del level16_new_1
                # del level16_new_2

                level32_new, = self.cells[count](None,
                                             level_16_curr,
                                             None,
                                             None,
                                             normalized_alphas)

                count += 1
                level_4_prev = level_4_curr
                level_4_curr = level4_new

                level_8_prev = level_8_curr
                level_8_curr = level8_new

                level_16_prev = level_16_curr
                level_16_curr = level16_new

                level_32_curr = level32_new

            elif layer == 3 :

                level4_new_1, level4_new_2 = self.cells[count](level_4_prev,
                                                           None,
                                                           level_4_curr,
                                                           level_8_curr,
                                                           normalized_alphas)
                count += 1
                level4_new = normalized_bottom_betas[layer][0] * level4_new_1 + normalized_bottom_betas[layer][1] * level4_new_2

                # del level4_new_1
                # del level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count](level_8_prev,
                                                                          level_4_curr,
                                                                          level_8_curr,
                                                                          level_16_curr,
                                                                          normalized_alphas)
                count += 1
                level8_new = normalized_betas8[layer - 1][0] * level8_new_1 + normalized_betas8[layer - 1][1] * level8_new_2 + \
                             normalized_betas8[layer - 1][2] * level8_new_3

                # del level8_new_1
                # del level8_new_2
                # del level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count](level_16_prev,
                                                                             level_8_curr,
                                                                             level_16_curr,
                                                                             level_32_curr,
                                                                             normalized_alphas)
                count += 1
                level16_new = normalized_betas16[layer - 2][0] * level16_new_1 + normalized_betas16[layer - 2][1] * level16_new_2 + \
                              normalized_betas16[layer - 2][2] * level16_new_3

                # del level16_new_1
                # del level16_new_2
                # del level16_new_3

                level32_new_1, level32_new_2 = self.cells[count](None,
                                                              level_16_curr,
                                                              level_32_curr,
                                                              None,
                                                              normalized_alphas)
                count += 1
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


            else:
                level4_new_1, level4_new_2 = self.cells[count] (level_4_prev,
                                                  None,
                                                  level_4_curr,
                                                  level_8_curr,
                                                  normalized_alphas)
                count += 1
                level4_new = normalized_bottom_betas[layer][0] * level4_new_1 + normalized_bottom_betas[layer][1] * level4_new_2

                # del level4_new_1
                # del level4_new_2

                level8_new_1, level8_new_2, level8_new_3 = self.cells[count] (level_8_prev,
                                                                              level_4_curr,
                                                                              level_8_curr,
                                                                              level_16_curr,
                                                                              normalized_alphas)
                count += 1
                level8_new = normalized_betas8[layer - 1][0] * level8_new_1 + normalized_betas8[layer - 1][1] * level8_new_2 + normalized_betas8[layer - 1][2] * level8_new_3

                # del level8_new_1
                # del level8_new_2
                # del level8_new_3

                level16_new_1, level16_new_2, level16_new_3 = self.cells[count] (level_16_prev,
                                                                                 level_8_curr,
                                                                                 level_16_curr,
                                                                                 level_32_curr,
                                                                                 normalized_alphas)
                count += 1
                level16_new = normalized_betas16[layer - 2][0] * level16_new_1 + normalized_betas16[layer - 2][1] * level16_new_2 + normalized_betas16[layer - 2][2] * level16_new_3

                # del level16_new_1
                # del level16_new_2
                # del level16_new_3

                level32_new_1, level32_new_2 = self.cells[count] (level_32_prev,
                                                                  level_16_curr,
                                                                  level_32_curr,
                                                                  None,
                                                                  normalized_alphas)
                count += 1
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

        # del level_4_prev
        # del level_8_prev
        # del level_16_prev
        # del level_32_prev

        aspp_result_4 = self.aspp_4(level_4_curr)
        aspp_result_8 = self.aspp_8(level_8_curr)
        aspp_result_16 = self.aspp_16(level_16_curr)
        aspp_result_32 = self.aspp_32(level_32_curr)

        # del level_4_curr
        # del level_8_curr
        # del level_16_curr
        # del level_32_curr

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
