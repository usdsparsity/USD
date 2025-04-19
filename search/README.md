# Search Module

```
   _    ___  ____  ____
  | |  / _ \|  _ \|  _ \
  | | | | | | | | | | | |
  | |_| |_| | |_| | |_| |
  |____\___/|____/|____/
  Search Module
```

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

The search module implements the core algorithms for finding optimal N:M sparsity patterns in neural networks.

## 🚀 Quick Start

```bash
# Run search on ImageNet
python find_mix_from_dense_imagenet.py \
    --M 8 \
    --batch_size 120 \
    --data [dataset_path] \
    --num_classes [num_classes] \
    --target_sparsity 0.75
```

## 📋 Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)

## 📊 Overview

This module contains the core search algorithms for:
- Unstructured pruning phase
- Structured N:M pattern optimization
- Dynamic pattern adjustment
- Weight penalty mechanisms

## ✨ Features

- 🔍 Two-phase search approach
- 🎯 Dynamic pattern optimization
- ⚙️ Configurable search parameters
- 📈 Performance monitoring
- 💾 Checkpoint management

## 💻 Usage

### Basic Search

```bash
python find_mix_from_dense_imagenet.py \
    --M 8 \
    --batch_size 120 \
    --data [dataset_path] \
    --num_classes [num_classes] \
    --target_sparsity 0.75
```

### Advanced Configuration

```bash
python find_mix_from_dense_imagenet.py \
    --M 8 \
    --batch_size 120 \
    --data [dataset_path] \
    --num_classes [num_classes] \
    --alpha -1 \
    --initialisation 1 \
    --target_sparsity 0.75 \
    --alpha_target_sparsity 0.9 \
    --with_weight_penalty 1
```

## 📝 Configuration

Configuration files are located in `script_resnet_ImageNet/configs/` and include:
- Model architectures
- Search parameters
- Training schedules
- Sparsity targets

## 🏗️ Project Structure

```
search/
├── find_mix_from_dense_imagenet.py    # Main search script
├── find_mix_from_dense_imagenet_tpu.py # TPU-optimized version
├── schemes/                           # Search schemes
├── models/                            # Model implementations
└── script_resnet_ImageNet/           # ResNet-specific scripts
```

## 📚 Examples

Example configurations and results can be found in:
- `schemes/checkpoint/` - Search checkpoints
- `results/` - Search results
- `configs/` - Example configurations

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU support)
- Additional dependencies in `requirements.txt` 