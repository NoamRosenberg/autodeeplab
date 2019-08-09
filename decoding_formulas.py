import numpy as np
import torch
import torch.nn.functional as F
from genotypes import PRIMITIVES
from genotypes import Genotype

class Decoder(object):
    def __init__(self, alphas, bottom_betas, betas8, betas16, top_betas, block_multiplier, steps):

        normalized_bottom_betas = F.softmax(bottom_betas, dim=-1)
        normalized_betas8 = F.softmax (betas8, dim = -1)
        normalized_betas16 = F.softmax(betas16, dim=-1)
        normalized_top_betas = F.softmax(top_betas, dim=-1)

        self.network_space = torch.zeros(12,4,3)

        self.network_space[0, 0, 1] = 1.
        self.network_space[0, 1, 0] = 1.

        self.network_space[1:, 0, 1:] = normalized_bottom_betas
        self.network_space[1, 1, :2] = normalized_top_betas[0]
        self.network_space[1, 2, 0] = 1.

        self.network_space[2:, 1, :] = normalized_betas8
        self.network_space[2, 2, :2] = normalized_top_betas[1]
        self.network_space[2, 3, 0] = 1.

        self.network_space[3:, 2, :] = normalized_betas16
        self.network_space[3:, 3, :2] = normalized_top_betas[2:]

        self.alphas = alphas
        self.block_multiplier = block_multiplier
        self.steps = steps

    def viterbi_decode(self):
        #TODO: consider if to set single path probabilities to a third or some other value other
        #TODO: than one in order not to bias the path decodicings
        #remember best paths
        paths_space = torch.zeros(self.network_space.shape)
        paths_space[0, 0, 1] = 1.
        paths_space[0, 1, 0] = 1.
        #update total transition probability
        prob_space = torch.zeros(self.network_space.shape[:2])
        prob_space[0, 0] = 1.
        prob_space[0, 1] = 1.
        for layer in range(1, paths_space.shape[0]):
            for sample in range(paths_space.shape[1]):
                new_probs_ls = []
                for i in range(0,3):
                    prev_sample = sample + (i - 1)
                    if prev_sample < 0 or prev_sample > 3:
                        new_probs_ls.append(torch.tensor(0.))
                        continue
                    prev_prob = prob_space[layer - 1, prev_sample]
                    curr_prob = self.network_space[layer, sample, i]
                    new_prob = prev_prob * curr_prob
                    new_probs_ls.append(new_prob)
                new_probs_tls = torch.tensor(new_probs_ls)
                if new_probs_tls[new_probs_tls > 0].shape[0] == 0:
                    continue
                prob_space[layer, sample] = new_probs_tls.max()
                best_index = new_probs_tls.argmax()
                paths_space[layer, sample, best_index] = 1.

        #pick the branch with the highest probability
        all_samples = torch.tensor([i for i in range(prob_space.shape[-1])])
        best_sample = torch.argmax(prob_space[-1])
        other_samples = all_samples[all_samples != best_sample]
        paths_space[-1, other_samples] = 0.
        i = paths_space[-1, best_sample].nonzero()[0, 0]
        prev_sample = best_sample + (i - 1)

        #follow the branch and eliminate all other branches
        for layer in range(prob_space.shape[0] - 2, -1, -1):
            other_samples = all_samples[all_samples != prev_sample]
            paths_space[layer, other_samples] = 0.
            i = paths_space[layer, prev_sample].nonzero()[0, 0]
            prev_sample = prev_sample + (i - 1)

        actual_path = paths_space.nonzero()[:,1]
        return actual_path, paths_space

    def dfs_decode (self) :
        best_result = []
        max_prop = 0
        def _parse (weight_network, layer, curr_value, curr_result, last) :
            nonlocal best_result
            nonlocal max_prop
            if layer == self._num_layers :
                if max_prop < curr_value :
                    # print (curr_result)
                    best_result = curr_result[:]
                    max_prop = curr_value
                return

            if layer == 0 :
                print ('begin0')
                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end0-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

            elif layer == 1 :
                print ('begin1')

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end1-1')

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()


            elif layer == 2 :
                print ('begin2')

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    print ('end2-1')
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 2
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()
            else :

                num = 0
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                num = 1
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 0)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 2
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 1)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][2]
                    curr_result.append ([num,2])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][2]
                    curr_result.pop ()

                num = 3
                if last == num :
                    curr_value = curr_value * weight_network[layer][num][0]
                    curr_result.append ([num,0])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 2)
                    curr_value = curr_value / weight_network[layer][num][0]
                    curr_result.pop ()

                    curr_value = curr_value * weight_network[layer][num][1]
                    curr_result.append ([num,1])
                    _parse (weight_network, layer + 1, curr_value, curr_result, 3)
                    curr_value = curr_value / weight_network[layer][num][1]
                    curr_result.pop ()
        network_weight = F.softmax(self.last_betas_network, dim=-1) * 5
        network_weight = network_weight.data.cpu().numpy()
        _parse (network_weight, 0, 1, [],0)
        print (max_prop)
        return best_result

    def genotype_decode(self):

        def _parse(alphas, steps):
            gene = []
            start = 0
            n = 2
            for i in range(steps):
                end = start + n
                edges = sorted(range(start, end), key=lambda x: -np.max(alphas[x,1:])) #ignore none value
                top2edges = edges[:2]
                for j in top2edges:
                    best_op_index = np.argmax(alphas[j]) #this can include none op
                    gene.append([j, best_op_index])
                start = end
                n += 1
            return np.array(gene)
        normalized_alphas = F.softmax(self.alphas, dim=-1).data.cpu().numpy()
        gene_cell = _parse(normalized_alphas, self.steps)

        return gene_cell


