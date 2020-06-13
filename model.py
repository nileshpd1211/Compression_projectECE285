import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

from utils import MaskedLinear

class LeNet(nn.Module):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.apply_pruning(threshold)

    def remove_pruned_connection(self, device):
        for name, p in self.named_parameters():
            if 'mask' not in name:
                weight_tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(weight_tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
                
    def apply_weight_sharing(model, bits=3):

        for module in model.children():
            dev = module.weight.device
            weight = module.weight.data.cpu().numpy()
            shape = weight.shape
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
            min_ = min(mat.data)
            max_ = max(mat.data)
            space = np.linspace(min_, max_, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
            kmeans.fit(mat.data.reshape(-1,1))
            new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
            mat.data = new_weight
            module.weight.data = torch.from_numpy(mat.toarray()).to(dev)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
