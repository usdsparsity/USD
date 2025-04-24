# Model Implementations

```
   _    ___  ____  ____
  | |  / _ \|  _ \|  _ \
  | | | | | | | | | | | |
  | |_| |_| | |_| | |_| |
  |____\___/|____/|____/
  Model Implementations
```

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

This directory contains implementations of various sparse neural network architectures.

## 📋 Contents

- [Overview](#overview)
- [Supported Models](#supported-models)
- [Usage](#usage)
- [Implementation Details](#implementation-details)

## 📊 Overview

The models directory provides sparse implementations of popular neural network architectures, optimized for N:M sparsity patterns.

## 🧠 Supported Models

### ResNet
- `resnet_sparse.py`: Sparse ResNet implementation
  - Supports ResNet18, ResNet50
  - Configurable sparsity patterns
  - Weight pruning support

### Vision Transformers
- `sparse_vision_transformer.py`: Sparse ViT implementation
  - Supports ViT-B/16
  - Attention sparsity
  - MLP sparsity

- `deit_sparse_vision_transformer.py`: Sparse DeiT implementation
  - Supports DeiT-Base and Small
  - Knowledge distillation support
  - Token pruning

- `sparse_swin_transformer.py`: Sparse Swin Transformer
  - Window attention sparsity
  - Shifted window support
  - Hierarchical sparsity

### MLP
- `misc_sparse_mlp.py`: Sparse MLP implementation
  - Configurable layers
  - Activation sparsity
  - Weight pruning

## 💻 Usage

### Basic Model Creation

```python
from models.resnet_sparse import resnet50_sparse

# Create sparse ResNet50
model = resnet50_sparse(
    num_classes=1000,
    sparsity_ratio=0.75,
    M=8
)
```

### Advanced Configuration

```python
from models.sparse_vision_transformer import vit_base_patch16_224_sparse

# Create sparse ViT
model = vit_base_patch16_224_sparse(
    num_classes=1000,
    sparsity_ratio=0.75,
    M=8,
    attention_sparsity=True,
    mlp_sparsity=True
)
```

## 🔧 Implementation Details

### Sparsity Patterns
- N:M structured sparsity
- Dynamic pattern adjustment
- Layer-wise sparsity control

### Optimization
- Weight pruning
- Attention sparsity
- MLP sparsity
- Token pruning

### Performance Features
- CUDA optimization
- Memory efficiency
- Training stability

## 🏗️ Project Structure

```
models/
├── resnet_sparse.py              # Sparse ResNet
├── sparse_vision_transformer.py  # Sparse ViT
├── deit_sparse_vision_transformer.py # Sparse DeiT
├── sparse_swin_transformer.py    # Sparse Swin
├── misc_sparse_mlp.py           # Sparse MLP
└── timm/                        # Modified timm models
```

## 📚 Examples

Example usage can be found in:
- `search/find_mix_from_dense_imagenet.py`
- `inference_benchmark/test_inference.py`

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU support)
- timm (for base model implementations) 