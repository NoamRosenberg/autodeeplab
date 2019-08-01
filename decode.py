import numpy as np
import torch
import torch.nn.functional as F

class ViterbiDecoder(object):
    def __init__(self):
        self._num_layers = 12
        self.bottom_betas = torch.tensor(1e-3 * torch.randn(self._num_layers - 1, 2).cuda(), requires_grad=True)
        self.betas8 = torch.tensor(1e-3 * torch.randn(self._num_layers - 2, 3).cuda(), requires_grad=True)
        self.betas16 = torch.tensor(1e-3 * torch.randn(self._num_layers - 3, 3).cuda(), requires_grad=True)
        self.top_betas = torch.tensor(1e-3 * torch.randn(self._num_layers - 1, 2).cuda(), requires_grad=True)

        normalized_bottom_betas = F.softmax(self.bottom_betas, dim=-1)
        normalized_betas8 = F.softmax (self.betas8, dim = -1)
        normalized_betas16 = F.softmax(self.betas16, dim=-1)
        normalized_top_betas = F.softmax(self.top_betas, dim=-1)

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

    def decode(self):

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

if __name__ == '__main__':
    viterbi = ViterbiDecoder()
    decode = viterbi.decode()