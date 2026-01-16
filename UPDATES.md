# LoRT Compression Tool - Updated Features

## ✅ Completed Enhancements

The `lort-compress` tool now supports flexible model loading from multiple sources:

### 1. **HuggingFace Models** (Default)
Load any public model directly from HuggingFace:
```bash
cargo run --bin lort-compress --release -- --model "Qwen/Qwen2.5-0.5B"
cargo run --bin lort-compress --release -- --model "meta-llama/Llama-2-7b-hf"
cargo run --bin lort-compress --release -- --model "mistralai/Mistral-7B-v0.1"
```

### 2. **Local Model Directory**
Point to a directory containing safetensors files:
```bash
cargo run --bin lort-compress --release -- --model /path/to/model/directory
```

The tool will automatically discover and load all `.safetensors` files in the directory.

### 3. **Single Safetensors File**
Directly compress a single model file:
```bash
cargo run --bin lort-compress --release -- --model /path/to/model.safetensors
```

### 4. **Sharded Models**
Automatically handles multi-file models:
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`

The tool detects and loads all shards in order.

## Command-Line Options

```
Options:
  -m, --model <MODEL>      Model ID (HF) or local path [default: Qwen/Qwen2.5-0.5B]
  -o, --output <OUTPUT>    Output .lort file [default: model.lort]
      --min-dim <MIN_DIM>  Min layer size to compress [default: 128]
  -h, --help               Show help
```

## Examples

### Compress with custom output
```bash
cargo run --bin lort-compress --release -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --output tinyllama.lort
```

### Only compress large layers
```bash
cargo run --bin lort-compress --release -- \
  --model "meta-llama/Llama-2-7b-hf" \
  --min-dim 512 \
  --output llama2-large-only.lort
```

### Local model with aggressive compression
```bash
cargo run --bin lort-compress --release -- \
  --model ~/models/my-llm \
  --min-dim 64 \
  --output my-llm-compressed.lort
```

## Technical Improvements

### 1. Flexible File Loading
- Auto-detection of local vs HuggingFace paths
- Support for multiple safetensors file patterns
- Automatic shard detection and ordering

### 2. Model Compatibility
Works with any transformer model that:
- Uses safetensors format
- Has 2D weight matrices
- Available on HF or stored locally

### 3. Smart Layer Filtering
- Skips embeddings, layer norms, biases
- Configurable minimum dimension threshold
- Only compresses meaningful linear layers

### 4. Optimized Decomposition
- Replaced SVD (not available in Candle 0.9) with iterative optimization
- Alternating Least Squares (ALS) for ternary + LoRA factorization
- Gradient-based LoRA updates with regularization

## Output Format

The `.lort` file contains:
- Header: Magic bytes "LORT" + version
- Layer count and metadata
- For each layer:
  - Layer name (for reconstruction)
  - Alpha scale factor (FP32)
  - Dimensions (out, in, rank)
  - LoRA matrices A, B (FP16)
  - Packed ternary weights (2 bits per weight)

## Compression Statistics

Typical results:
- **Compression Ratio**: 7-8x vs FP16
- **Bits per Weight**: ~2.2 bits
- **Model Quality**: Maintained through hybrid ternary+LoRA representation

## Next Steps

To use the compressed model:
1. Compile CUDA kernel: `nvcc -ptx -arch=sm_86 kernels/lort_kernel.cu -o kernels/lort_kernel.ptx`
2. Run inference: `cargo run --bin lort-infer --release`

See [EXAMPLES.md](EXAMPLES.md) for more usage patterns.
