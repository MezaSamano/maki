# Project LoRT - Low-Rank Ternary Quantization Engine

A state-of-the-art quantization engine achieving ~2.2 bits/weight with CUDA-accelerated inference.

## 📖 Documentation

- **[CLOUD_SETUP.md](CLOUD_SETUP.md)** - Complete guide for testing on cloud instances ⭐ **Start here!**
- **[EXAMPLES.md](EXAMPLES.md)** - Usage examples and common patterns
- **[UPDATES.md](UPDATES.md)** - Recent features and improvements

## Architecture

This workspace contains two binaries:

- **lort-compress**: Offline parallel solver that decomposes model weights using iterative optimization and ternary quantization
- **lort-infer**: Online inference engine with custom CUDA kernels for high-performance execution

## 🚀 Quick Start

### Cloud Testing (Recommended)

For step-by-step cloud instance setup with HuggingFace models, see **[CLOUD_SETUP.md](CLOUD_SETUP.md)**.

**TL;DR:**
```bash
# On any Ubuntu cloud instance:
git clone https://github.com/MezaSamano/maki.git
cd maki
cargo build --bin lort-compress --release
cargo run --bin lort-compress --release -- --model "Qwen/Qwen2.5-0.5B"
```

### Local Setup

### 1. Compile CUDA Kernel (Optional - for inference)

```bash
cd kernels
nvcc -ptx -arch=sm_86 lort_kernel.cu -o lort_kernel.ptx
cd ..
```

*Note: Adjust `-arch=sm_86` to match your GPU architecture (e.g., sm_80, sm_89)*

### 2. Run Compression

**From HuggingFace:**
```bash
cargo run --bin lort-compress --release -- --model "Qwen/Qwen2.5-0.5B"
```

**From local directory:**
```bash
cargo run --bin lort-compress --release -- --model /path/to/model/directory
```

**From local safetensors file:**
```bash
cargo run --bin lort-compress --release -- --model /path/to/model.safetensors
```

**With custom settings:**
```bash
cargo run --bin lort-compress --release -- \
  --model "meta-llama/Llama-2-7b-hf" \
  --output llama2.lort \
  --min-dim 256
```

This generates a compressed `.lort` file with approximately 7.2x compression ratio compared to FP16.

### 3. Run Inference

```bash
cargo run --bin lort-infer --release
```

This loads the compressed model and executes inference using custom CUDA kernels.

## How It Works

### Compression (LoRT Decomposition)

Each weight matrix W is decomposed as:

```
W ≈ α·T + A·B^T
```

Where:
- **T**: Ternary matrix (-1, 0, +1) packed to 2 bits/weight
- **α**: Scalar scale factor (FP32)
- **A, B**: Low-rank factors (FP16, rank=64)

The decomposition uses:
1. Alternating Least Squares (ALS) optimization
2. CPU-based parallel SVD (Rayon)
3. Greedy ternary quantization

### Inference

The forward pass combines:
1. **Ternary path**: Custom CUDA kernel with 2-bit unpacking
2. **LoRA path**: Standard cuBLAS matmul
3. **Fusion**: Element-wise addition

## Configuration

### Command-line Options

```bash
lort-compress [OPTIONS]

Options:
  -m, --model <MODEL>          Model ID from HuggingFace or local path [default: Qwen/Qwen2.5-0.5B]
  -o, --output <OUTPUT>        Output file path [default: model.lort]
      --min-dim <MIN_DIM>      Minimum dimension for layers to compress [default: 128]
  -h, --help                   Print help
```

### Compression Parameters

Edit constants in `lort-compress/src/main.rs`:

- `RANK`: LoRA rank (default: 64) - Higher rank = better quality, larger size
- `ITERATIONS`: ALS iterations (default: 10) - More iterations = better convergence

## Performance

- **Compression Ratio**: ~7.2x vs FP16
- **Bits per Weight**: ~2.2 bits
- **Inference Speed**: Microsecond-level forward passes on modern GPUs
- **Supported Models**: Any transformer model with safetensors format

## Supported Models

The compressor works with any model that:
- Uses safetensors format
- Has 2D weight matrices (linear layers)
- Is available on HuggingFace or stored locally

**Tested models:**
- Qwen/Qwen2.5-0.5B (default)
- meta-llama/Llama-2-7b-hf
- mistralai/Mistral-7B-v0.1
- Any GPT, BERT, T5, etc. architecture

## Requirements

- Rust 1.70+
- CUDA Toolkit 11.0+
- NVIDIA GPU with compute capability 8.0+

## File Format

`.lort` file structure:
```
Header: "LORT" (4 bytes)
Version: u32
Num Layers: u32
For each layer:
  - Alpha: f32
  - Dimensions: out_dim (u32), in_dim (u32), rank (u32)
  - LoRA matrices: A, B (FP16)
  - Packed ternary: size (u32), bytes
```
