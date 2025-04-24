# Inference Benchmark Module

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

The Inference Benchmark module provides tools for testing and benchmarking sparse model inference performance.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)

<a id="overview"></a>
## 📊 Overview

This module contains tools for:
- Sparse model inference testing
- Performance benchmarking
- Memory usage analysis
- Throughput measurements
- Latency analysis

<a id="features"></a>
## ✨ Features

- 🔄 Comprehensive inference testing
- 🎯 Performance benchmarking
- ⚙️ Configurable test parameters
- 📈 Memory usage tracking
- 💾 Result logging and visualization

<a id="usage"></a>
## 💻 Usage

### Basic Usage

```bash
# Run inference benchmarks
python test_inference.py \
    --model_path [model_path] \
    --batch_size 32 \
    --num_iters 100 \
    --device cuda
```

### Advanced Configuration

```bash
python test_inference.py \
    --model_path [model_path] \
    --batch_size 32 \
    --num_iters 100 \
    --device cuda \
    --precision fp16 \
    --warmup_iters 10 \
    --log_interval 10
```

<a id="configuration"></a>
## ⚙️ Configuration

Configuration files are located in `configs/` and include:
- Model configurations
- Test parameters
- Benchmark settings
- Logging options

<a id="project-structure"></a>
## 🏗️ Project Structure

```
inference_benchmark/
├── configs/              # Configuration files
├── full_models/         # Full model checkpoints
├── schemes/            # Pruning schemes
└── test_inference.py   # Main test script
```

<a id="requirements"></a>
## 🔧 Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU support)
- Additional dependencies in `requirements.txt` 