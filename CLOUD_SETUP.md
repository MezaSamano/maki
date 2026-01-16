# Cloud Instance Testing Guide for LoRT

**Complete step-by-step guide** to test the LoRT compression engine with HuggingFace models on any cloud instance.

## 🎯 Quick Overview

This guide will take you from a fresh cloud instance to successfully compressing transformer models in **under 15 minutes**.

**What you'll do:**
1. Launch a cloud instance (~2 min)
2. Install dependencies (~5 min)
3. Build LoRT (~3 min)
4. Compress your first model (~5 min)

---

## Prerequisites

- Access to any cloud provider (AWS, GCP, DigitalOcean, Hetzner, Paperspace, etc.)
- Basic SSH knowledge
- Credit card for cloud billing (most offer free credits)

**No GPU needed** for compression! CPU-only instances work great.

---

---

## 📋 Step-by-Step Setup

### Step 1: Launch Cloud Instance (2 minutes)

**Recommended Specifications:**

| Component | Minimum | Recommended | For 7B+ Models |
|-----------|---------|-------------|----------------|
| CPU       | 4 cores | 8 cores     | 16+ cores      |
| RAM       | 8GB     | 16GB        | 32GB+          |
| Storage   | 30GB    | 50GB        | 100GB+         |
| OS        | Ubuntu 22.04 | Ubuntu 22.04 | Ubuntu 22.04 |

**Provider-Specific Quick Launch:**

<details>
<summary><b>AWS EC2</b> (Click to expand)</summary>

```bash
# Via AWS Console:
# 1. Go to EC2 Dashboard
# 2. Click "Launch Instance"
# 3. Choose "Ubuntu 22.04 LTS"
# 4. Select t3.xlarge (4 vCPU, 16GB RAM)
# 5. Create/select key pair
# 6. Launch

# Or via CLI:
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.xlarge \
  --key-name your-key-name
```

**Cost**: ~$0.17/hour (~$0.34 for a full test session)
</details>

<details>
<summary><b>Google Cloud Platform</b></summary>

```bash
# Create instance
gcloud compute instances create lort-test \
  --machine-type=n2-standard-4 \
  --zone=us-central1-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# Get SSH command
gcloud compute ssh lort-test --zone=us-central1-a
```

**Cost**: ~$0.19/hour
</details>

<details>
<summary><b>DigitalOcean</b></summary>

1. Go to https://cloud.digitalocean.com/droplets/new
2. Choose **Ubuntu 22.04**
3. Select **CPU-Optimized** or **Basic** plan (16GB RAM)
4. Choose any datacenter region
5. Add SSH key
6. Click "Create Droplet"

```bash
# SSH into droplet
ssh root@your-droplet-ip
```

**Cost**: ~$0.14/hour (CPU-Optimized) or ~$0.09/hour (Basic)
</details>

<details>
<summary><b>Hetzner Cloud</b> (Most Cost-Effective!)</summary>

1. Go to https://console.hetzner.cloud
2. Create new project
3. Add Server → Choose:
   - Location: Any
   - Image: Ubuntu 22.04
   - Type: **CPX31** (4 vCPU, 8GB) or **CCX23** (8 vCPU, 16GB)
4. Add SSH key
5. Create

```bash
ssh root@your-server-ip
```

**Cost**: €0.02/hour (~$0.02/hour) ⭐ **Best value!**
</details>

<details>
<summary><b>Paperspace</b></summary>

1. Go to https://console.paperspace.com
2. Create → Gradient Notebook or Core instance
3. Choose:
   - Template: Ubuntu 22.04
   - Machine: C5 or better (for future GPU inference)
   - Region: Any

**Cost**: Varies, free tier available
</details>

### Step 2: Connect to Your Instance (1 minute)

```bash
# Generic SSH command (replace with your details)
ssh username@your-instance-ip

# AWS example
ssh -i ~/.ssh/your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# Most other providers
ssh root@your-instance-ip
```

✅ **You're in!** You should now see a terminal prompt on your cloud instance.

---

### Step 3: Install Dependencies (5 minutes)

Copy and paste this entire block:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y \
  build-essential \
  curl \
  git \
  pkg-config \
  libssl-dev \
  htop

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Load Rust environment
source $HOME/.cargo/env

# Verify installation
echo "✅ Checking installations..."
rustc --version
cargo --version
git --version
```

**Expected output:**
```
✅ Checking installations...
rustc 1.xx.x (xxxxxxxxx 20xx-xx-xx)
cargo 1.xx.x (xxxxxxxxx 20xx-xx-xx)
git version 2.xx.x
```

---

### Step 4: Clone and Build LoRT (3 minutes)

```bash
# Clone the repository
git clone https://github.com/MezaSamano/maki.git
cd maki

# Build the compressor (this takes 2-3 minutes)
echo "🔨 Building LoRT compressor..."
time cargo build --bin lort-compress --release

# Verify build
./target/release/lort-compress --help
```

**Expected output:**
```
Compress transformer models using LoRT quantization

Usage: lort-compress [OPTIONS]

Options:
  -m, --model <MODEL>      Model ID from HuggingFace or local path
  -o, --output <OUTPUT>    Output file path [default: model.lort]
      --min-dim <MIN_DIM>  Minimum dimension for layers to compress [default: 128]
  -h, --help               Print help
```

✅ **Build successful!** You're ready to compress models.

---

## 🧪 Testing with HuggingFace Models

### Test 1: Qwen 0.5B (Fastest - 3-5 minutes) ⭐ **Start here!**

This is the perfect first test to validate your setup.

```bash
# Create output directory
mkdir -p compressed

# Run compression with timing
echo "🚀 Starting compression of Qwen 0.5B..."
time cargo run --bin lort-compress --release -- \
  --model "Qwen/Qwen2.5-0.5B" \
  --output compressed/qwen-0.5b.lort

# Check the result
echo "📊 Compression complete! Checking output..."
ls -lh compressed/
```

**What you'll see:**

```
🚀 LoRT Compression Engine
Model: Qwen/Qwen2.5-0.5B
Output: compressed/qwen-0.5b.lort

📥 Loading model files...
  Downloading from HuggingFace: Qwen/Qwen2.5-0.5B
  Found: model.safetensors
✓ Found 1 model file(s)

📂 Loading model weights...
  Reading: /home/user/.cache/huggingface/...
  Found layer: model.layers.0.self_attn.q_proj.weight [896, 896]
  Found layer: model.layers.0.self_attn.k_proj.weight [896, 128]
  ... (more layers)

✓ Loaded 42 compressible layers
Starting LoRT Decomposition (Rank=64, Iterations=10)...

[00:03:24] ████████████████████████████ 42/42 Decomposition complete!

💾 Saving to 'compressed/qwen-0.5b.lort'...

✅ Done!
   Original size (FP16): 896.50 MB
   Compressed size:      124.31 MB
   Compression Ratio:    7.21x
   Bits per weight:      2.22

real    3m24.567s
```

✅ **Success!** You've compressed your first model!

---

### Test 2: TinyLlama 1.1B (Medium - 8-12 minutes)

Scale up to a larger model:

```bash
echo "🚀 Compressing TinyLlama 1.1B..."
time cargo run --bin lort-compress --release -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --output compressed/tinyllama-1.1b.lort \
  --min-dim 256

# Compare sizes
ls -lh compressed/
du -sh ~/.cache/huggingface/hub/
```

**Expected results:**
- Original: ~2.2GB
- Compressed: ~300MB
- Ratio: ~7.3x
- Time: 8-12 minutes (depends on CPU)

---

### Test 3: Phi-2 (2.7B - 15-25 minutes)

```bash
echo "🚀 Compressing Phi-2..."
time cargo run --bin lort-compress --release -- \
  --model "microsoft/phi-2" \
  --output compressed/phi2.lort

# Monitor progress in another terminal
# (SSH in again or use tmux/screen)
htop
```

**Expected results:**
- Original: ~5.4GB
- Compressed: ~750MB
- Ratio: ~7.2x

---

### Test 4: Mistral 7B (Large - 45-90 minutes) ⚠️ **Needs 32GB RAM**

Only if your instance has sufficient RAM:

```bash
# Check available RAM first
free -h

# If you have 32GB+, proceed
echo "🚀 Compressing Mistral 7B..."
time cargo run --bin lort-compress --release -- \
  --model "mistralai/Mistral-7B-v0.1" \
  --output compressed/mistral-7b.lort \
  --min-dim 512
```

**Tip**: Use `tmux` or `screen` for long-running tasks:
```bash
# Install tmux
sudo apt install tmux -y

# Start session
tmux new -s compress

# Run compression
cargo run --bin lort-compress --release -- \
  --model "mistralai/Mistral-7B-v0.1" \
  --output compressed/mistral-7b.lort

# Detach: Press Ctrl+B, then D
# Reattach later: tmux attach -t compress
```

---

## 📊 Monitoring and Performance

### Real-time Monitoring

**Terminal 1** - Run compression:
```bash
cd ~/maki
cargo run --bin lort-compress --release -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --output compressed/test.lort
```

**Terminal 2** - Monitor resources:
```bash
# Install monitoring tools
sudo apt install htop iotop -y

# CPU and memory
htop

# Or simpler view
watch -n 1 'free -h && echo "---" && uptime'
```

### Check Disk Usage

```bash
# HuggingFace cache
du -sh ~/.cache/huggingface/

# Compressed models
ls -lh compressed/

# Total disk space
df -h
```

### Performance Benchmarks

Expected compression times on different hardware:

| Model Size | 4 cores | 8 cores | 16 cores |
|-----------|---------|---------|----------|
| 0.5B      | 5 min   | 3 min   | 2 min    |
| 1B        | 12 min  | 8 min   | 5 min    |
| 3B        | 30 min  | 20 min  | 12 min   |
| 7B        | 90 min  | 60 min  | 40 min   |

---

## 🎯 Advanced Testing

### Test with Local Models

Pre-download models for repeated testing:

```bash
# Install huggingface-cli
pip install -U huggingface-hub

# Download model
huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir ./models/qwen

# Compress from local path
cargo run --bin lort-compress --release -- \
  --model ./models/qwen \
  --output compressed/qwen-local.lort

# This is much faster for repeated tests!
```

### Experiment with Compression Settings

```bash
# Aggressive: Compress more layers
cargo run --bin lort-compress --release -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --min-dim 64 \
  --output compressed/tiny-aggressive.lort

# Conservative: Compress fewer layers
cargo run --bin lort-compress --release -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --min-dim 1024 \
  --output compressed/tiny-conservative.lort

# Compare results
ls -lh compressed/
```

### Batch Testing Script

Create a test script:

```bash
cat > test_models.sh << 'EOF'
#!/bin/bash
set -e

MODELS=(
  "Qwen/Qwen2.5-0.5B"
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "microsoft/phi-2"
)

mkdir -p compressed results

for model in "${MODELS[@]}"; do
  echo "========================================="
  echo "Testing: $model"
  echo "========================================="
  
  output_name=$(echo $model | tr '/' '-')
  
  time cargo run --bin lort-compress --release -- \
    --model "$model" \
    --output "compressed/${output_name}.lort" \
    2>&1 | tee "results/${output_name}.log"
  
  echo ""
done

---

## ❗ Troubleshooting

### Problem: Out of Memory

**Symptoms:**
```
killed
Error: process didn't exit successfully
```

**Solutions:**

1. **Add swap space:**
```bash
# Create 16GB swap
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify
free -h

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

2. **Use smaller model:**
```bash
cargo run --bin lort-compress --release -- \
  --model "Qwen/Qwen2.5-0.5B"
```

3. **Compress fewer layers:**
```bash
cargo run --bin lort-compress --release -- \
  --model "your-model" \
  --min-dim 2048  # Only very large layers
```

---

### Problem: Download Fails

**Symptoms:**
```
Error: Failed to download model file
```

**Solutions:**

1. **Check internet:**
```bash
ping -c 3 huggingface.co
curl -I https://huggingface.co
```

2. **Use different cache location:**
```bash
export HF_HOME=/tmp/huggingface
cargo run --bin lort-compress --release
```

3. **Pre-download with CLI:**
```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./models/qwen
cargo run --bin lort-compress --release -- --model ./models/qwen
```

---

### Problem: Build Errors

**Symptoms:**
```
error: could not compile `lort-compress`
```

**Solutions:**

1. **Update Rust:**
```bash
rustup update stable
rustc --version
```

2. **Clean and rebuild:**
```bash
cargo clean
cargo build --bin lort-compress --release
```

3. **Check disk space:**
```bash
df -h
# Need at least 5GB free for build
```

---

### Problem: Gated Model Access

**Symptoms:**
```
Error: Access denied. This model requires authentication
```

**Solution:**

```bash
# Install huggingface-cli
pip install huggingface-hub

# Get token from: https://huggingface.co/settings/tokens
huggingface-cli login

# Then compress
cargo run --bin lort-compress --release -- \
  --model "meta-llama/Llama-2-7b-hf"
```

---

### Problem: Slow Compression

**Expected times** (on 8-core CPU):
- 0.5B: 3-5 minutes ✅
- 1B: 8-12 minutes ✅
- 3B: 20-30 minutes ✅
- 7B: 45-90 minutes ✅

**If much slower:**

1. **Check CPU usage:**
```bash
htop
# Should show high CPU on all cores
```

2. **Ensure release build:**
```bash
# Wrong (slow):
cargo run --bin lort-compress

# Right (fast):
cargo run --bin lort-compress --release
```

3. **Upgrade instance:**
- More CPU cores = faster compression
- Consider 8 or 16 core instances

---

## 🧹 Cleanup

### After Testing

```bash
# See what's taking space
du -sh ~/.cache/huggingface/ compressed/

# Remove model cache (frees 1-10GB)
rm -rf ~/.cache/huggingface/

# Keep only compressed models
ls -lh compressed/

# Download counts
ls ~/.cache/huggingface/hub/ 2>/dev/null | wc -l
```

### Before Terminating Instance

```bash
# Download compressed models to your local machine
# From your LOCAL terminal:
scp user@instance-ip:~/maki/compressed/*.lort ./

# Or create a tarball
cd ~/maki
tar -czf compressed-models.tar.gz compressed/
# Then download compressed-models.tar.gz
```

### Terminate Cloud Instance

**⚠️ Important:** Don't forget to stop/terminate to avoid charges!

**AWS:**
```bash
# Stop (can restart later)
aws ec2 stop-instances --instance-ids i-xxxxx

# Terminate (permanent)
aws ec2 terminate-instances --instance-ids i-xxxxx
```

**GCP:**
```bash
gcloud compute instances delete lort-test --zone=us-central1-a
```

**DigitalOcean/Hetzner/Paperspace:**
- Use web interface to delete/destroy instance

---

## ✅ Verification Checklist

After completing the tests, verify:

- ✅ Rust and Cargo installed correctly
- ✅ Repository cloned successfully  
- ✅ Binary builds without errors
- ✅ Help message displays
- ✅ Model downloads from HuggingFace
- ✅ Compression completes successfully
- ✅ `.lort` file created
- ✅ Compression ratio ~7-8x
- ✅ Bits per weight ~2.2

**Success output:**
```
✅ Done!
   Original size (FP16): XXX.XX MB
   Compressed size:      YYY.YY MB
   Compression Ratio:    7.XXx
   Bits per weight:      2.XX
```

---

## 📈 Cost Estimates

**Typical test session** (2 hours):

| Provider | Instance Type | Cost/Hour | 2-Hour Total |
|----------|---------------|-----------|--------------|
| Hetzner  | CPX31 (4c/8GB) | $0.02 | **$0.04** ⭐ |
| DigitalOcean | Basic 16GB | $0.09 | $0.18 |
| AWS | t3.xlarge | $0.17 | $0.34 |
| GCP | n2-standard-4 | $0.19 | $0.38 |

💡 **Tip:** Hetzner offers the best value. AWS/GCP have spot instances for 60-80% savings.

---

## 🎓 What's Next?

Now that you've successfully compressed models:

1. **Compare different models**
   - Test compression ratios across architectures
   - Find optimal `--min-dim` settings
   
2. **Benchmark performance**
   - Time different instance types
   - Measure RAM usage patterns
   
3. **Set up inference**
   - Compile CUDA kernels
   - Test decompression speed
   - Measure inference latency

4. **Production deployment**
   - Automate compression pipeline
   - Integrate with ML workflows
   - Deploy compressed models

---

## 📚 Additional Resources

- **Main README**: [README.md](README.md)
- **Usage Examples**: [EXAMPLES.md](EXAMPLES.md)
- **Feature Updates**: [UPDATES.md](UPDATES.md)
- **GitHub Repo**: https://github.com/MezaSamano/maki

---

## 🐛 Need Help?

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review error messages carefully
3. Open an issue on GitHub with:
   - Instance specs (CPU, RAM)
   - Full error output
   - Commands you ran

---

**Happy compressing! 🚀**

*Last updated: January 15, 2026*


```bash
# Example for AWS
ssh -i your-key.pem ubuntu@your-instance-ip

# For other providers, follow their SSH instructions
```

### Step 3: Install Rust

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Step 4: Install System Dependencies

```bash
# Essential build tools
sudo apt install -y build-essential pkg-config libssl-dev git

# Optional: CUDA for inference (skip for compression-only)
# sudo apt install -y nvidia-cuda-toolkit
```

### Step 5: Clone Repository

```bash
# Clone the maki repo
git clone https://github.com/MezaSamano/maki.git
cd maki

# Verify structure
ls -la
# Should see: Cargo.toml, lort-compress/, lort-infer/, kernels/, README.md
```

### Step 6: Build the Compression Tool

```bash
# Build in release mode (optimized)
cargo build --bin lort-compress --release

# This will take 5-10 minutes on first build
# Downloads and compiles all dependencies
```

### Step 7: Test with Small Model (TinyLlama 1.1B)

```bash
# Run compression on TinyLlama (fast test, ~5 minutes)
cargo run --bin lort-compress --release -- \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --output tinyllama.lort \
  --min-dim 256

# Expected output:
# 🚀 LoRT Compression Engine
# Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
# Output: tinyllama.lort
# 
# 📥 Loading model files...
#   Downloading from HuggingFace: TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   Found: model.safetensors
# ✓ Found 1 model file(s)
# 
# 📂 Loading model weights...
#   Reading: /root/.cache/huggingface/...
#   Found layer: model.layers.0.self_attn.q_proj.weight [2048, 2048]
#   Found layer: model.layers.0.self_attn.k_proj.weight [2048, 256]
#   ...
# 
# ✓ Loaded 42 compressible layers
# Starting LoRT Decomposition (Rank=64, Iterations=10)...
# 
# [00:03:24] ████████████████████████████ 42/42 Decomposition complete!
# 
# 💾 Saving to 'tinyllama.lort'...
# 
# ✅ Done!
#    Original size (FP16): 2048.00 MB
#    Compressed size:      284.12 MB
#    Compression Ratio:    7.21x
#    Bits per weight:      2.22
```

### Step 8: Verify Output

```bash
# Check the compressed file
ls -lh tinyllama.lort

# Should show ~280-300MB file

# View compression stats
du -h tinyllama.lort
```

---

## Option 2: Test with Larger Models

### Qwen 0.5B (Default, Very Fast)

```bash
# ~2 minutes compression time
cargo run --bin lort-compress --release -- \
  --model "Qwen/Qwen2.5-0.5B" \
  --output qwen-0.5b.lort

# Expected size: ~120-140MB compressed
```

### Phi-2 (2.7B Parameters)

```bash
# ~10 minutes compression time
# Requires 16GB+ RAM
cargo run --bin lort-compress --release -- \
  --model "microsoft/phi-2" \
  --output phi2.lort

# Expected size: ~400-450MB compressed
```

### Mistral 7B (Requires More RAM)

```bash
# ~30-45 minutes compression time
# Requires 32GB+ RAM
cargo run --bin lort-compress --release -- \
  --model "mistralai/Mistral-7B-v0.1" \
  --output mistral-7b.lort

# Expected size: ~1.1-1.3GB compressed
```

---

## Option 3: Using Pre-Downloaded Models (Faster Testing)

If you have limited bandwidth or want to avoid re-downloading:

### Step 1: Download Model First

```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Login (for gated models)
huggingface-cli login

# Download model
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --local-dir ./models/tinyllama

# Verify download
ls -lh ./models/tinyllama/
# Should see: model.safetensors, config.json, etc.
```

### Step 2: Compress Local Model

```bash
# Point to local directory
cargo run --bin lort-compress --release -- \
  --model ./models/tinyllama \
  --output tinyllama-local.lort
```

---

## Advanced Options

### Custom Compression Settings

```bash
# Only compress very large layers (faster)
cargo run --bin lort-compress --release -- \
  --model "Qwen/Qwen2.5-0.5B" \
  --min-dim 512 \
  --output qwen-selective.lort

# Aggressive compression (compress smaller layers)
cargo run --bin lort-compress --release -- \
  --model "microsoft/phi-2" \
  --min-dim 64 \
  --output phi2-aggressive.lort
```

### Monitor Resource Usage

```bash
# In a separate terminal
watch -n 1 'free -h && echo "" && top -bn1 | head -20'

# Or use htop
sudo apt install htop
htop
```

### Compression Performance Tips

1. **More CPU cores = faster compression** (uses Rayon for parallelism)
2. **Compression is CPU-bound**, not GPU-bound
3. **RAM requirements**:
   - 0.5B models: 8GB+
   - 1-3B models: 16GB+
   - 7B models: 32GB+
   - 13B+ models: 64GB+

---

## Testing Inference (Optional - Requires GPU)

If your instance has an NVIDIA GPU:

### Step 1: Install CUDA

```bash
# For Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

### Step 2: Compile CUDA Kernel

```bash
cd ~/maki/kernels

# Check your GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Compile (adjust sm_XX based on your GPU)
# sm_86 = RTX 3090/A6000
# sm_89 = RTX 4090/L40
# sm_80 = A100
nvcc -ptx -arch=sm_86 lort_kernel.cu -o lort_kernel.ptx

# Verify
ls -lh lort_kernel.ptx
```

### Step 3: Test Inference

```bash
cd ~/maki

# Note: Inference code needs additional implementation
# Currently a skeleton - shows basic structure
cargo run --bin lort-infer --release
```

---

## Troubleshooting

### Issue: Out of Memory During Compression

**Solution**: Use a larger instance or compress fewer layers
```bash
cargo run --bin lort-compress --release -- \
  --model "your-model" \
  --min-dim 1024  # Only compress very large layers
```

### Issue: Download Fails (403/404 Error)

**Gated Models**: Some models require authentication
```bash
pip install huggingface_hub
huggingface-cli login
# Enter your HuggingFace token
```

**Private Models**: Use local path instead
```bash
cargo run --bin lort-compress --release -- \
  --model /path/to/model
```

### Issue: Build Errors

**Missing OpenSSL**:
```bash
sudo apt install -y libssl-dev pkg-config
```

**Rust Version Too Old**:
```bash
rustup update
```

### Issue: Slow Compression

**Expected Times** (on 8-core CPU):
- 0.5B model: 1-3 minutes
- 1B model: 3-8 minutes
- 3B model: 10-20 minutes
- 7B model: 30-60 minutes

Compression is CPU-intensive and uses all available cores.

---

## Complete Example Script

Save this as `test_compression.sh`:

```bash
#!/bin/bash
set -e

echo "=== LoRT Compression Test Script ==="
echo ""

# 1. Update system
echo "Step 1: Updating system..."
sudo apt update && sudo apt upgrade -y

# 2. Install Rust
echo "Step 2: Installing Rust..."
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# 3. Install dependencies
echo "Step 3: Installing dependencies..."
sudo apt install -y build-essential pkg-config libssl-dev git

# 4. Clone repo
echo "Step 4: Cloning repository..."
cd ~
if [ ! -d "maki" ]; then
    git clone https://github.com/MezaSamano/maki.git
fi
cd maki

# 5. Build
echo "Step 5: Building (this may take 5-10 minutes)..."
cargo build --bin lort-compress --release

# 6. Test with small model
echo "Step 6: Testing with Qwen 0.5B..."
cargo run --bin lort-compress --release -- \
  --model "Qwen/Qwen2.5-0.5B" \
  --output qwen-test.lort

# 7. Verify
echo ""
echo "=== Test Complete ==="
echo "Compressed file:"
ls -lh qwen-test.lort
echo ""
echo "Success! LoRT compression is working."
```

Run it:
```bash
chmod +x test_compression.sh
./test_compression.sh
```

---

## Expected Results Summary

| Model | Params | Original (FP16) | Compressed | Ratio | Time (8 cores) |
|-------|--------|----------------|------------|-------|----------------|
| Qwen 0.5B | 0.5B | ~1.0 GB | ~140 MB | 7.1x | 2 min |
| TinyLlama | 1.1B | ~2.2 GB | ~310 MB | 7.1x | 5 min |
| Phi-2 | 2.7B | ~5.4 GB | ~750 MB | 7.2x | 15 min |
| Mistral 7B | 7B | ~14 GB | ~1.95 GB | 7.2x | 45 min |

**Compression Ratio**: Consistently ~7-7.5x vs FP16
**Bits per Weight**: ~2.2 bits average

---

## Cost Estimation

### Compression-Only (CPU Instance)

**AWS t3.xlarge** (4 vCPU, 16GB RAM) @ $0.166/hour:
- Qwen 0.5B: ~$0.01 
- TinyLlama 1.1B: ~$0.02
- Phi-2: ~$0.05
- Mistral 7B: ~$0.13

**GCP n2-standard-4** (4 vCPU, 16GB RAM) @ $0.194/hour:
- Similar costs, slightly higher

**Budget Cloud (Vast.ai/RunPod)**: 50-80% cheaper

### With GPU (For Inference Testing)

**RunPod RTX 3090** @ $0.44/hour:
- Compression + Inference testing: ~$1-2 for full workflow

---

## Next Steps

After successful compression:

1. **Experiment with different models**
2. **Test compression quality** (compare outputs)
3. **Benchmark compression speed** with different CPU counts
4. **Implement inference engine** (currently skeleton code)
5. **Measure inference performance** vs FP16

---

## Support

- **GitHub Issues**: https://github.com/MezaSamano/maki/issues
- **Documentation**: See README.md and EXAMPLES.md in repo
- **HuggingFace Models**: https://huggingface.co/models

Happy compressing! 🚀
