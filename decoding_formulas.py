import numpy as np
import pdb
import torch
import torch.nn.functional as F
from retrain_model.new_model import network_layer_to_space

class Decoder(object):
    def __init__(self, alphas, betas, steps):
        self._betas = betas
        self._alphas = alphas
        self._steps = steps
        self._num_layers = self._betas.shape[0]
        self.network_space = torch.zeros(12, 4, 3)

        for layer in range(self._num_layers):
            if layer == 0:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1)  * (2/3)
            elif layer == 1:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)

            elif layer == 2:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)            
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)


            else:
                self.network_space[layer][0][1:] = F.softmax(self._betas[layer][0][1:], dim=-1) * (2/3)
                self.network_space[layer][1] = F.softmax(self._betas[layer][1], dim=-1)
                self.network_space[layer][2] = F.softmax(self._betas[layer][2], dim=-1)
                self.network_space[layer][3][:2] = F.softmax(self._betas[layer][3][:2], dim=-1) * (2/3)
        
    def viterbi_decode(self):
        prob_space = np.zeros((self.network_space.shape[:2]))
        path_space = np.zeros((self.network_space.shape[:2])).astype('int8')

        for layer in range(self.network_space.shape[0]):
            if layer == 0:
                prob_space[layer][0] = self.network_space[layer][0][1]
                prob_space[layer][1] = self.network_space[layer][0][2]
                path_space[layer][0] = 0
                path_space[layer][1] = -1
            else:
                for sample in range(self.network_space.shape[1]):
                    if layer - sample < - 1:
                        continue
                    local_prob = []
                    for rate in range(self.network_space.shape[2]):  # k[0 : ➚, 1: ➙, 2 : ➘]
                        if (sample == 0 and rate == 2) or (sample == 3 and rate == 0):
                            continue
                        else:
                            local_prob.append(prob_space[layer - 1][sample + 1 - rate] *
                                              self.network_space[layer][sample + 1 - rate][rate])
                    prob_space[layer][sample] = np.max(local_prob, axis=0)
                    rate = np.argmax(local_prob, axis=0)
                    path = 1 - rate if sample != 3 else -rate
                    path_space[layer][sample] = path  # path[1 : ➚, 0: ➙, -1 : ➘]

        output_sample = prob_space[-1, :].argmax(axis=-1)
        actual_path = np.zeros(12).astype('uint8')
        actual_path[-1] = output_sample
        for i in range(1, self._num_layers):
            actual_path[-i - 1] = actual_path[-i] + path_space[self._num_layers - i, actual_path[-i]]

        return actual_path, network_layer_to_space(actual_path)

    def dfs_decode(self):
        best_result = []
        max_prop = 0

        def _parse(weight_network, layer, curr_value, curr_result, last):
            nonlocal best_result
            nonlocal max_prop
            if layer == self._num_layers:
                if max_prop < curr_value:
                    # print (curr_result)
                    best_result = curr_result[:]
                    max_prop = curr_value
                return

            if layer == 0:
                print('begin0')
                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print('end0-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

            elif layer == 1:
                print('begin1')

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print('end1-1')

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()


            elif layer == 2:
                print('begin2')

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    print('end2-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 2
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()
            else:

                num = 0
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                num = 1
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 2
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append([num, 2])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop()

                num = 3
                if last == num:
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append([num, 0])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append([num, 1])
                    _parse(weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop()

        network_weight = F.softmax(self.last_betas_network, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse(network_weight, 0, 1, [], 0)
        print(max_prop)
        return best_result

    def genotype_decode(self):

        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x, 1:]))  # ignore none value
                top2edges = edges[:2]
                for j in top2edges:
                    best_op_index = np.argmax(alphas[j])  # this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)

        normalized_alphas = F.softmax(self._alphas, dim=-1).data.cpu().numpy()
        gene_cell = _parse(normalized_alphas, self._steps)

        return gene_cell
