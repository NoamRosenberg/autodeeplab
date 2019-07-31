import numpy as np
import torch
import torch.nn.functional as F


class ViterbiDecoder(object):
    def __init__(self, initialProb, transProb, obsProb):
        self.N = initialProb.shape[0]
        self.initialProb = initialProb
        self.transProb = transProb
        self.obsProb = obsProb
        assert self.initialProb.shape == (self.N, 1)
        assert self.transProb.shape == (self.N, self.N)
        assert self.obsProb.shape[0] == self.N

    def Obs(self, obs):
        return self.obsProb[:, obs, None]

    def Decode(self, obs):
        trellis = np.zeros((self.N, len(obs)))
        backpt = np.ones((self.N, len(obs)), 'int32') * -1

        # initialization
        trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[0]))

        for t in range(1, len(obs)):
            trellis[:, t] = (trellis[:, t - 1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
            backpt[:, t] = (np.tile(trellis[:, t - 1, None], [1, self.N]) * self.transProb).argmax(0)
        # termination
        tokens = [trellis[:, -1].argmax()]
        for i in range(len(obs) - 1, 0, -1):
            tokens.append(backpt[tokens[-1], i])
        return tokens[::-1]

class ViterbiDecoder(object):
    def __init__(self, bottom_betas, betas8, betas16, top_betas):
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
        self.network_space[3:, 3, :] = normalized_top_betas[2:]

    def decode(self):

        #best_paths_space = torch.argmax(self.network_space, dim=-1)
        paths_space = torch.zeros(self.network_space.shape)
        paths_space[0, 0, 1] = 1.
        paths_space[0, 1, 0] = 1.
        prob_space = torch.zeros(self.network_space.shape[:2])
        paths = []
        prob_space[0, 0] = 1.
        prob_space[0, 1] = 1.
        for layer in range(1, paths_space.shape[0]):
            for sample in range(paths_space.shape[1]):
                new_probs_ls = []
                for i in range(0,3):
                    prev_sample = sample + (i - 1)
                    if prev_sample < 0 or prev_sample > 3:
                        continue
                    #clamp
                    #prev_sample = max(min(sample + (i-1), 0),3)
                    prev_prob = paths_space[layer - 1, prev_sample]
                    curr_prob = self.network_space[layer, sample, i]
                    new_prob = prev_prob * curr_prob
                    new_probs_ls.append(new_prob)
                    new_probs_tls = torch.tensor(new_probs_ls)
                new_probs_tls
                paths_space[layer, sample] = new_probs_tls.max()