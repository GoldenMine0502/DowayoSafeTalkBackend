import torch
from torch import nn

criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=0)

inputs = torch.tensor([-0.5, 0.5])

inputs = softmax(torch.tensor([-0.5, 0.5]))
labels = torch.tensor([0, 1.0])

print(inputs, criterion(inputs, labels))