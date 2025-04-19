import torch
import numpy as np
from time import time
import scipy.sparse as sp
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from devkit.dataset.imagenet_dataset import ImageNetValDataset
import os
from devkit.core.utils import load_pre_train

torch.manual_seed(2022)
np.random.seed(2022)

def create_sparse_tensor(shape, sparsity=0.9):
    # Create a tensor with random values
    tensor = torch.rand(shape)
    # Create a mask with sparsity% zeros
    mask = torch.rand(shape) > sparsity
    # Apply mask to create sparse tensor
    sparse_tensor = tensor * mask
    return sparse_tensor

def create_coo_sparse_tensor(shape, sparsity=0.9):
    # Create a COO format sparse tensor directly
    nnz = int((1 - sparsity) * shape[0] * shape[1])
    indices = torch.randint(0, shape[0], (2, nnz))
    values = torch.rand(nnz)
    return torch.sparse_coo_tensor(indices, values, shape)

# Test with 256x256 tensors with 90% zeros
print("\nCreating tensors with 90% sparsity...")
a_dense = create_sparse_tensor((256, 256), sparsity=0.9)
b_dense = create_sparse_tensor((256, 256), sparsity=0.9)

# Create COO format sparse tensors
a_coo = create_coo_sparse_tensor((256, 256), sparsity=0.9)
b_coo = create_coo_sparse_tensor((256, 256), sparsity=0.9)

# Print sparsity statistics
print(f"Dense tensor a sparsity: {(a_dense == 0).sum().item() / a_dense.numel() * 100:.2f}%")
print(f"Dense tensor b sparsity: {(b_dense == 0).sum().item() / b_dense.numel() * 100:.2f}%")
print(f"COO tensor a sparsity: {(1 - a_coo._nnz() / a_coo.numel()) * 100:.2f}%")
print(f"COO tensor b sparsity: {(1 - b_coo._nnz() / b_coo.numel()) * 100:.2f}%")

# Move to CUDA
a_dense = a_dense.cuda()
b_dense = b_dense.cuda()
a_coo = a_coo.cuda()
b_coo = b_coo.cuda()

# Test dense multiplication
print("\nTesting dense multiplication (1000 iterations)...")
dense_start_time = time()
for _ in range(1000):
    y_dense = torch.mm(a_dense, b_dense)
    torch.cuda.synchronize()
dense_end_time = time()
print(f"Average time per iteration: {(dense_end_time - dense_start_time)/100:.6f}s")

# Test sparse multiplication with COO format
print("\nTesting sparse multiplication with COO format (1000 iterations)...")
sparse_start_time = time()
for _ in range(1000):
    y_sparse = torch.sparse.mm(a_coo, b_coo)
    torch.cuda.synchronize()
sparse_end_time = time()
print(f"Average time per iteration: {(sparse_end_time - sparse_start_time)/100:.6f}s")

# Test with different sparsity levels
sparsity_levels = [0.5, 0.7, 0.9, 0.95, 0.99]
print("\nTesting different sparsity levels (1000 iterations each):")
for sparsity in sparsity_levels:
    print(f"\nSparsity: {sparsity*100:.0f}%")
    
    # Create new tensors with current sparsity
    a_dense = create_sparse_tensor((256, 256), sparsity=sparsity)
    b_dense = create_sparse_tensor((256, 256), sparsity=sparsity)
    a_coo = create_coo_sparse_tensor((256, 256), sparsity=sparsity)
    b_coo = create_coo_sparse_tensor((256, 256), sparsity=sparsity)
    
    # Move to CUDA
    a_dense = a_dense.cuda()
    b_dense = b_dense.cuda()
    a_coo = a_coo.cuda()
    b_coo = b_coo.cuda()
    
    # Test dense multiplication
    dense_start_time = time()
    for _ in range(1000):
        y_dense = torch.mm(a_dense, b_dense)
        torch.cuda.synchronize()
    dense_end_time = time()
    print(f"PyTorch dense average time: {(dense_end_time - dense_start_time)/100:.6f}s")
    
    # Test sparse multiplication
    sparse_start_time = time()
    for _ in range(1000):
        y_sparse = torch.sparse.mm(a_coo, b_coo)
        torch.cuda.synchronize()
    sparse_end_time = time()
    print(f"PyTorch sparse average time: {(sparse_end_time - sparse_start_time)/100:.6f}s")
    
    # Calculate speedup factor
    dense_time = (dense_end_time - dense_start_time)/100
    sparse_time = (sparse_end_time - sparse_start_time)/100
    print(f"Speedup factor: {dense_time/sparse_time:.2f}x")