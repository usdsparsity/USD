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

exit(0)
def convert_to_sparse(tensor, threshold=0.1):
    """Convert dense tensor to sparse format by zeroing out small values"""
    mask = torch.abs(tensor) > threshold
    sparse_tensor = tensor * mask
    return sparse_tensor.to_sparse()

def evaluate_sparse(model, val_loader, criterion, device):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # Convert input to sparse if it's a matrix
            if len(input.shape) == 4:  # For CNN inputs
                input = input.view(input.size(0), -1)
            input_sparse = input.to_sparse()

            # Forward pass with sparse operations
            output = model(input_sparse)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = "saved_models/resnet50/resnet50_sparse"  # Update with your model path
    model = models.__dict__['resnet50_sparse'](pretrained=False, N=8, M=8, search=True, num_new_classes=1000)
    model = model.to(device)
    load_pre_train(model_path, model)
    
    # Convert model weights to sparse where possible
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:  # Only convert weight matrices
            param.data = convert_to_sparse(param.data)
    
    # Define transforms and dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_dataset = ImageNetValDataset(
        csv_file='path_to_val_annotations.csv',  # Update with your validation annotations
        root_dir='path_to_val_images',  # Update with your validation images path
        transform=val_transform
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Evaluate with sparse operations
    print("Evaluating with sparse operations...")
    sparse_acc = evaluate_sparse(model, val_loader, criterion, device)
    
    # For comparison, evaluate with dense operations
    print("\nEvaluating with dense operations...")
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            param.data = param.data.to_dense()
    
    dense_acc = evaluate_sparse(model, val_loader, criterion, device)
    
    print(f"\nSparse Accuracy: {sparse_acc:.2f}%")
    print(f"Dense Accuracy: {dense_acc:.2f}%")

if __name__ == '__main__':
    main()
