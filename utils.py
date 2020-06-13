import math
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
from pathlib import Path

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq

class MaskedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

    	# Initialize the mask with 1
        self.mask = nn.Parameter(torch.ones([out_features, in_features]), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def init_weights(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def apply_pruning(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device

        # Convert Tensors to numpy and calculate
        weight_tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(weight_tensor) < threshold, 0, mask)

        # Apply new weight and mask
        self.weight.data = torch.from_numpy(weight_tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)

def custom_dump(code_str, filename):

    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    with open(filename, 'wb') as f:
        f.write(byte_arr)

    return len(byte_arr)

def layer_encoder(arr, prefix, save_dir='./'):
    # Infer dtype
    dtype = str(arr.dtype)

    # Calculate frequency in arr
    freq_map = defaultdict(int)
    convert_map = {'float32':float, 'int32':int}
    for value in np.nditer(arr):
        value = convert_map[dtype](value)
        freq_map[value] += 1

    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)

    # Merge nodes
    while(len(heap) > 1):
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.freq + node2.freq, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = {}
    def make_codebook(node, code):
        if node is None:
            return
        if node.value is not None:
            value2code[node.value] = code
            return
            
        make_codebook(node.left, code + '0')
        make_codebook(node.right, code + '1')

    root = heappop(heap)
    make_codebook(root, '')

    # Path to save location
    directory = Path(save_dir)

    # Dump data
    data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
    datasize = custom_dump(data_encoding, directory/f'{prefix}.bin')

    return  datasize

# Encode models
def huffman_encoder(model, directory='encodings/'):
    os.makedirs(directory, exist_ok=True)
    
    original_total = 0
    compressed_total = 0
    print(f"{'Layer':<10} | {'Original':>10} | {'Compressed':>10} | {'Compression Factor'}")
    print('-'*60)
    for name, param in model.named_parameters():
        if 'bias' in name:
            # No huffman encode bias
            bias = param.data.cpu().numpy()
            bias.dump(f'{directory}/{name}.bin')

            # Print Stats
            original = bias.nbytes
            compressed = original

            print(f"{name:<10} | {original:10} | {compressed:10} | {compressed/original:>17.4f}")

        if 'weight' in name:
            weight = param.data.cpu().numpy()
            shape = weight.shape
            form = 'csr' if shape[0] < shape[1] else 'csc'
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

            # Encode
            d0 = layer_encoder(mat.data, name+f'_{form}_data', directory)
            d1 = layer_encoder(mat.indices, name+f'_{form}_indices', directory)

            index_diff = mat.indptr[1:] - mat.indptr[:-1]
            d2 = layer_encoder(index_diff, name+f'_{form}_indptr', directory)

            # Print Stats
            original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
            compressed = d0 + d1 + d2

            print(f"{name:<10} | {original:10} | {compressed:10} | {compressed/original:>17.4f}")
            
        original_total += original
        compressed_total += compressed

    print('-'*60)
    print(f"{'total':10} | {original_total:>10} | {compressed_total:>10} | {compressed_total/original_total:>17.4f}")