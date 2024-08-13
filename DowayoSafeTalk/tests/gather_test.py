import torch

alpha = torch.tensor([0.75, 0.25])

targets = torch.tensor([
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

res = alpha.gather(0, targets[:, 1])

print(res)