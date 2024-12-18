# Import libraries
import torch

# Check the torch version
version = torch.__version__

#
print(f"Torch version: {version}")

#
print(f"Is cude available: {torch.cuda.is_available()}")

#
print(f"Is backends mps available: {torch.backends.mps.is_available()}")

# O-dimensional tensor
tensor0d = torch.tensor(1)

print(f"Tensor 0d: {tensor0d}")

# 1-dimensional tensor
tensor1d = torch.tensor([1, 2, 3])

print(f"Tensor 1d: {tensor1d}")

# 2-dimensional tensor
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(f"Tensor 2d: {tensor2d}")

# 3-dimensional tensor
tensor3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [4, 5, 6]], [[7, 8, 9], [4, 5, 6]]])

print(f"Tensor3d: {tensor3d}")


# data type check
print(tensor3d.dtype)

#
print(tensor2d.shape)

#
tensor2d.view(3, 2)

print(tensor2d)