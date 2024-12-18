#
import torch
import torch.nn.functional as F
from torch.autograd import grad

# True label
y = torch.tensor([1.0])

# Input
x1 = torch.tensor([1.1])

# Weight
w1 = torch.tensor([2.2], requires_grad=True)

# Bias
b = torch.tensor([0.0], requires_grad=True)

# net input
n = w1 * x1 + b

# net output
a = torch.sigmoid(n)

#
loss = F.binary_cross_entropy(a, y)

#
grad_loss_w1 = grad(loss, w1, retain_graph=True)
grad_loss_b = grad(loss, b, retain_graph=True)

#
print(f"Gradient of loss wrt w1: {grad_loss_w1}")
print(f"Gradient of loss wrt b: {grad_loss_b}")