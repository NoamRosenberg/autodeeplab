import torch
import numpy as np
from decoding_formulas import Decoder

a = torch.from_numpy(np.load('result/alpha.npy'))
b = torch.from_numpy(np.load('result/beta.npy'))
b = b.numpy()

max_min = np.max(b, axis=-1, keepdims=True) - np.min(b, axis=-1, keepdims=True)

for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        b[i, j] = (b[i, j] - np.min(b, axis=-1, keepdims=True)
                   [i, j]) / max_min[i, j]

print(b)


b = torch.from_numpy(b)

decoder = Decoder(a, b, 5)

print(decoder.network_space)

print(decoder.viterbi_decode())
