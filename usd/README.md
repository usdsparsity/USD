# USD Module


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

The USD module implements the core algorithms for finding optimal N:M sparsity patterns in neural networks.

## 📋 Table of Contents

- [USD Module](#usd-module)
  - [📋 Table of Contents](#-table-of-contents)
  - [📊 Overview](#-overview)
  - [✨ Features](#-features)
  - [💻 Usage](#-usage)
    - [Basic Search](#basic-search)
    - [Advanced Configuration](#advanced-configuration)
  - [⚙️ Configuration](#️-configuration)
  - [🏗️ Project Structure](#️-project-structure)
  - [📚 Examples](#-examples)
  - [🔧 Requirements](#-requirements)
  - [🚀 Inference Testing](#-inference-testing)

<a id="overview"></a>
## 📊 Overview

This module contains the core search algorithms for:
- Unstructured pruning phase
- Structured N:M pattern optimization
- Dynamic pattern adjustment
- Weight penalty mechanisms

<a id="features"></a>
## ✨ Features

- 🔍 Two-phase search approach
- 🎯 Dynamic pattern optimization
- ⚙️ Configurable search parameters
- 📈 Performance monitoring
- 💾 Checkpoint management

<a id="usage"></a>
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

<a id="configuration"></a>
## ⚙️ Configuration

Configuration files are located in `script_resnet_ImageNet/configs/` and include:
- Model architectures
- Search parameters
- Training schedules
- Sparsity targets

<a id="project-structure"></a>
## 🏗️ Project Structure

```
usd/
├── find_mix_from_dense_imagenet.py    # Main search script
├── find_mix_from_dense_imagenet_tpu.py # TPU-optimized version
├── schemes/                           # Search schemes
├── models/                            # Model implementations
├── script_resnet_ImageNet/           # ResNet-specific scripts
└── inference_benchmark/              # Inference testing utilities
```

<a id="examples"></a>
## 📚 Examples

Example configurations and results can be found in:
- `schemes/checkpoint/` - Search checkpoints
- `results/` - Search results
- `configs/` - Example configurations
- `inference_benchmark/` - Inference test results

<a id="requirements"></a>
## 🔧 Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU support)
- Additional dependencies in `requirements.txt`

<a id="inference-testing"></a>
## 🚀 Inference Testing

The framework includes comprehensive inference testing capabilities:

```bash
# Run inference benchmarks
python test_sparse_inference.py

# Run specific benchmark tests
./benchmark_inference.sh
```

The inference testing module supports:
- Sparse model inference
- Performance benchmarking
- Memory usage analysis
- Throughput measurements 