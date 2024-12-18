# Import libraries
import torch
import torch.nn.functional as F

# True label
y = torch.tensor([1.0])

# Input
x1 = torch.tensor([1.1])

# Weight
w1 = torch.tensor([2.2])

# Bias
b = torch.tensor([0.0])

# net input
n = w1 * x1 + b

# net output
a = torch.sigmoid(n)

#
loss = F.binary_cross_entropy(a, y)

#
print(f"Input to the model: {x1}")
print(f"weight: {w1}")
print(f"bias: {b}")
print(f"Net input: {n}")
print(f"True label: {y}")
print(f"Output of the model: {a}")
print(f"Total loss: {loss}")