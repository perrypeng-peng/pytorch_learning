import torch

# Create a tensor from a list
tensor1 = torch.tensor([1, 2, 3, 4])
print("Tensor from list:", tensor1)

# Create a tensor of zeros
zeros = torch.zeros((2, 3))
print("Zeros tensor:\n", zeros)

# Create a tensor of ones
ones = torch.ones((2, 3))
print("Ones tensor:\n", ones)

# Create a random tensor
random_tensor = torch.rand((2, 3))
print("Random tensor:\n", random_tensor)

# Tensor operations
sum_tensor = tensor1 + 10
print("Tensor after addition:", sum_tensor)

# Access tensor properties
print("Shape:", tensor1.shape)
print("Data type:", tensor1.dtype)
print("Device:", tensor1.device)