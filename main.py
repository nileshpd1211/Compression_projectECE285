import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from model import LeNet
from utils import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned: {total - nonzero}, total: {total}, Compression Factor: {total/nonzero:10.2f}x')
    

def train(model, optimizer, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_acc = []
        epoch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            epoch_loss.append(loss.item())

            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            epoch_acc.append(pred.eq(target.data.view_as(pred)).float().mean().item())

            # zero-out all the gradients corresponding to the pruned connections
            model.remove_pruned_connection(device)

            optimizer.step()

        print(f'Epoch: {epoch:1} | Average Loss: {np.mean(epoch_loss):.4f}, Average Accuracy: {100*np.mean(epoch_acc): 3.2f}%')

def test(model, test_loader):
    model.eval()
    test_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            epoch_acc += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * epoch_acc / len(test_loader.dataset)

    return accuracy

batch_size = 50
Nepochs = 10
learning_rate = 0.01
log_interval = 10
sensitivity = 2.5
os.makedirs('saves/', exist_ok=True)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False)

# Define which model to use
model = LeNet(mask=True).to(device)

# Define Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()

# Initial training
print("--------------- Initial training ---------------")
train(model, optimizer, train_loader, Nepochs)
accuracy = test(model, test_loader)
print("Initial Accuracy:",accuracy)
torch.save(model, f"saves/initial_model.ptmodel")
print("--------------- Stats Before Pruning ---------------")
print_nonzeros(model)
print("")

# Pruning
print("--------------- Pruning ---------------")
model.prune_by_std(sensitivity)
accuracy = test(model, test_loader)
print("Accuracy After Pruning:",accuracy)
print("--------------- Stats After Pruning ---------------")
print_nonzeros(model)
print("")

# Retrain
print("--------------- Retraining ---------------")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(model, optimizer, train_loader, Nepochs)
accuracy = test(model, test_loader)
print("Accuracy After Retraining:",accuracy)
torch.save(model, f"saves/model_after_pruning.pb")
print("--------------- Stats After Retraining ---------------")
print_nonzeros(model)
print("")

# Weight Sharing
model = torch.load('./saves/model_after_pruning.pb', map_location=device)		# load model
print("--------------- Weight Sharing ---------------")
model.apply_weight_sharing()
accuracy = test(model, test_loader)
print("Accuracy After weight sharing:",accuracy)
torch.save(model, f"saves/model_after_weightsharing.pb")
print("")

# Huffman Coding
model = torch.load("./saves/model_after_weightsharing.pb", map_location=device)		# load model
print("--------------- Huffman Coding ---------------")
huffman_encoder(model)