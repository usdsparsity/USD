# Inference Benchmark

```
   _    ___  ____  ____
  | |  / _ \|  _ \|  _ \
  | | | | | | | | | | | |
  | |_| |_| | |_| | |_| |
  |____\___/|____/|____/
  Inference Benchmark
```

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

This directory contains tools and scripts for benchmarking model inference performance with N:M sparsity.

## 📋 Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)

## 📊 Overview

The inference benchmark module provides tools for:
- Measuring inference speed
- Comparing different sparsity patterns
- Evaluating memory usage
- Analyzing model performance

## ✨ Features

- ⚡ Fast inference testing
- 📊 Comprehensive metrics
- 🔍 Detailed profiling
- 💾 Results logging
- 📈 Performance visualization

## 💻 Usage

### Basic Benchmarking

```bash
python test_inference.py \
    --model resnet50 \
    --sparsity 0.75 \
    --M 8 \
    --batch_size 32
```

### Advanced Configuration

```bash
python test_inference.py \
    --model vit_base_patch16_224 \
    --sparsity 0.75 \
    --M 8 \
    --batch_size 32 \
    --num_runs 100 \
    --warmup 10 \
    --profile True
```

## 📝 Configuration

### Model Configuration
- Model architecture
- Sparsity ratio
- N:M pattern
- Batch size

### Benchmark Parameters
- Number of runs
- Warmup iterations
- Profiling options
- Memory tracking

## 📊 Results

Results are stored in:
- `results/` directory
- CSV format for easy analysis
- Performance metrics:
  - Inference time
  - Memory usage
  - FLOPs
  - Model size

## 🏗️ Project Structure

```
inference_benchmark/
├── test_inference.py        # Main benchmark script
├── configs/                # Benchmark configurations
├── full_models/           # Full model implementations
└── results/              # Benchmark results
```

## 📚 Examples

Example configurations:
- `validate_file_16_16.csv`: Validation config for 16x16 models
- `validate_file_16_16_old.csv`: Previous version

## 🔧 Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (for GPU support)
- Additional dependencies in `requirements.txt` 