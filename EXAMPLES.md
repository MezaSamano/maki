# LoRT Compression Examples

## Basic Usage

### Compress Qwen 0.5B (default)
```bash
cargo run --bin lort-compress --release
```

### Compress Different HuggingFace Models

**Small models:**
```bash
# TinyLlama 1.1B
cargo run --bin lort-compress --release -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --output tinyllama.lort

# Phi-2
cargo run --bin lort-compress --release -- \
  --model "microsoft/phi-2" \
  --output phi2.lort
```

**Larger models:**
```bash
# Llama 2 7B
cargo run --bin lort-compress --release -- \
  --model "meta-llama/Llama-2-7b-hf" \
  --output llama2-7b.lort

# Mistral 7B
cargo run --bin lort-compress --release -- \
  --model "mistralai/Mistral-7B-v0.1" \
  --output mistral-7b.lort
```

## Local Models

### Compress from local directory
```bash
# If you have already downloaded a model
cargo run --bin lort-compress --release -- \
  --model ~/models/llama-2-7b \
  --output llama2.lort
```

### Compress single safetensors file
```bash
cargo run --bin lort-compress --release -- \
  --model ~/models/my-model/model.safetensors \
  --output my-model.lort
```

### Compress sharded models
```bash
# Directory with model-00001-of-00002.safetensors, model-00002-of-00002.safetensors, etc.
cargo run --bin lort-compress --release -- \
  --model ~/models/large-model \
  --output large-model.lort
```

## Advanced Options

### Adjust compression threshold
```bash
# Only compress layers with dimensions >= 256 (skip smaller layers)
cargo run --bin lort-compress --release -- \
  --model "Qwen/Qwen2.5-0.5B" \
  --min-dim 256 \
  --output qwen-selective.lort

# Compress all layers >= 64x64
cargo run --bin lort-compress --release -- \
  --model "microsoft/phi-2" \
  --min-dim 64 \
  --output phi2-aggressive.lort
```

## Typical Workflow

1. **Download model first (optional):**
   ```bash
   # Using huggingface-cli
   huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./models/qwen
   ```

2. **Compress:**
   ```bash
   cargo run --bin lort-compress --release -- \
     --model ./models/qwen \
     --output qwen.lort
   ```

3. **Check output:**
   ```bash
   ls -lh qwen.lort
   # Should be ~7x smaller than original FP16 weights
   ```

## Expected Output

```
🚀 LoRT Compression Engine
Model: Qwen/Qwen2.5-0.5B
Output: model.lort

📥 Loading model files...
  Downloading from HuggingFace: Qwen/Qwen2.5-0.5B
  Found: model.safetensors
✓ Found 1 model file(s)

📂 Loading model weights...
  Reading: /home/user/.cache/huggingface/...
  Found layer: model.layers.0.self_attn.q_proj.weight [896, 896]
  Found layer: model.layers.0.self_attn.k_proj.weight [896, 128]
  ...

✓ Loaded 42 compressible layers
Starting LoRT Decomposition (Rank=64, Iterations=10)...

[00:02:34] ████████████████████████████ 42/42 Decomposition complete!

💾 Saving to 'model.lort'...

✅ Done!
   Original size (FP16): 896.50 MB
   Compressed size:      124.31 MB
   Compression Ratio:    7.21x
   Bits per weight:      2.22
```

## Troubleshooting

### Model not found
```bash
# Make sure the model ID is correct
cargo run --bin lort-compress --release -- --model "author/model-name"

# For gated models, login first:
huggingface-cli login
```

### Out of memory
```bash
# Reduce the rank in src/main.rs:
const RANK: usize = 32;  // Instead of 64

# Or compress fewer layers:
cargo run --bin lort-compress --release -- --min-dim 512
```

### Slow compression
- SVD is CPU-intensive and uses all cores
- Larger models (7B+) may take 10-30 minutes
- Progress bar shows real-time status
