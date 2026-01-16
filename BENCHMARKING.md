# LoRT Benchmarking Guide

Complete guide for running and analyzing LoRT compression benchmarks in cloud environments.

## Quick Start

```bash
# Install dependencies (if not already installed)
# Ubuntu/Debian:
apt update && apt install -y bc time

# RHEL/CentOS/Fedora:
yum install -y bc time

# Alpine:
apk add bc coreutils

# Make script executable
chmod +x benchmark.sh

# Run standard benchmark (Qwen 0.5B, TinyLlama 1.1B, Phi-2)
./benchmark.sh

# Results saved to: benchmark_results/results_TIMESTAMP.csv
```

---

## Benchmark Modes

### 1. Quick Test (1 model, ~2-3 minutes)

Perfect for validating setup:

```bash
./benchmark.sh --quick
```

**Tests**: Qwen 0.5B only

**Use case**: Verify environment, test script functionality

---

### 2. Standard Benchmark (3 models, ~20-30 minutes)

Recommended for most testing:

```bash
./benchmark.sh
```

**Tests**:
- Qwen 0.5B (0.5B params)
- TinyLlama 1.1B (1.1B params)
- Phi-2 (2.7B params)

**Use case**: Compare performance across small-to-medium models

---

### 3. Large Models (4 models, ~60-90 minutes)

For comprehensive testing (requires 32GB+ RAM):

```bash
./benchmark.sh --large
```

**Tests**:
- Qwen 0.5B
- TinyLlama 1.1B
- Phi-2 (2.7B)
- Mistral 7B (7B params) ⚠️ **Needs 32GB+ RAM**

**Use case**: Full performance analysis, production validation

---

### 4. Custom Model

Test specific model:

```bash
./benchmark.sh --model "microsoft/phi-2"
./benchmark.sh --model "./local/models/custom-model"
```

**Use case**: Targeted testing, local models

---

## What Gets Measured

Each benchmark tracks:

| Metric | Description |
|--------|-------------|
| **Duration** | Total compression time (seconds) |
| **Memory Used** | RAM increase during compression (MB) |
| **Peak Memory** | Maximum memory footprint (MB) |
| **Original Size** | Uncompressed model size in FP16 (MB) |
| **Compressed Size** | Final .lort file size (MB) |
| **Compression Ratio** | Original / Compressed (e.g., 7.2x) |
| **Bits per Weight** | Average bits per parameter (~2.2) |
| **File Size** | Actual .lort file size on disk (MB) |
| **Cache Size** | HuggingFace cache usage (MB) |
| **Status** | Success or failure |

---

## Output Files

All results saved to `benchmark_results/`:

```
benchmark_results/
├── results_20260115_143022.csv        # Raw data
├── benchmark_20260115_143022.log      # Full log output
└── compressed/                         # .lort files
    ├── Qwen-Qwen2.5-0.5B.lort
    ├── TinyLlama-TinyLlama-1.1B-Chat-v1.0.lort
    └── microsoft-phi-2.lort
```

### CSV Format

```csv
model,duration_sec,mem_used_mb,peak_mem_mb,original_mb,compressed_mb,compression_ratio,bits_per_weight,file_size_mb,cache_mb,status
Qwen/Qwen2.5-0.5B,124,856,2048.5,896.5,124.3,7.21,2.22,124.31,924,success
```

---

## Analyzing Results

### Using Python Analyzer

```bash
# Make analyzer executable
chmod +x analyze_benchmark.py

# Analyze latest results
python3 analyze_benchmark.py benchmark_results/results_20260115_143022.csv
```

**Output**:

```
==================================================
BENCHMARK RESULTS
==================================================
Model                                    Time         Original     Compressed   Ratio    Bits/W   Status    
----------------------------------------------------
Qwen2.5-0.5B                             0:02:04      896.50 MB    124.31 MB    7.21x    2.22     ✓         
TinyLlama-1.1B-Chat-v1.0                 0:08:15      2048.00 MB   284.12 MB    7.21x    2.22     ✓         
phi-2                                    0:18:42      5400.00 MB   750.34 MB    7.20x    2.23     ✓         

==================================================
SUMMARY STATISTICS
==================================================

Tests Run: 3
  Successful: 3
  Failed: 0

Time Statistics:
  Total: 0:29:01
  Average: 0:09:40
  Min: 0:02:04 (Qwen/Qwen2.5-0.5B)
  Max: 0:18:42 (microsoft/phi-2)

Size Statistics:
  Total Original: 8.06 GB
  Total Compressed: 1.13 GB
  Overall Compression Ratio: 7.14x
  Space Saved: 6.93 GB (86.0%)

Compression Ratio:
  Average: 7.21x
  Min: 7.20x (microsoft/phi-2)
  Max: 7.21x (Qwen/Qwen2.5-0.5B)

Bits per Weight:
  Average: 2.22 bits
  Min: 2.22 bits
  Max: 2.23 bits

Memory Usage:
  Average Used: 1456.33 MB
  Average Peak: 3250.50 MB
  Max Peak: 5800.25 MB (microsoft/phi-2)
```

### Generate Plots

Requires matplotlib:

```bash
# Install matplotlib
pip3 install matplotlib

# Generate visualizations
python3 analyze_benchmark.py benchmark_results/results_*.csv --plot
```

**Generates**: `benchmark_results/analysis.png` with 4 plots:
1. Compression time comparison
2. Compression ratio comparison
3. File size comparison (original vs compressed)
4. Memory usage (used vs peak)

---

## Comparing Multiple Runs

Test different configurations:

```bash
# Run benchmark on instance A
./benchmark.sh
# Results: results_instanceA.csv

# Run benchmark on instance B (different CPU/RAM)
./benchmark.sh
# Results: results_instanceB.csv

# Compare
python3 analyze_benchmark.py results_instanceA.csv results_instanceB.csv --compare
```

**Output**:

```
Model: Qwen/Qwen2.5-0.5B
Run                            Time            Ratio      Bits/W     Status
--------------------------------------------------------------------------------
instanceA (8 cores)            0:02:04         7.21x      2.22       ✓
instanceB (16 cores)           0:01:15         7.21x      2.22       ✓

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
instanceA (8 cores)            0:08:15         7.21x      2.22       ✓
instanceB (16 cores)           0:04:30         7.21x      2.22       ✓
```

---

## Cloud Provider Testing

### Recommended Workflow

**1. Launch instance** (see [CLOUD_SETUP.md](CLOUD_SETUP.md))

**2. One-command setup + benchmark:**

```bash
# Download and run
curl -sSL https://raw.githubusercontent.com/MezaSamano/maki/main/benchmark.sh | bash -s -- --quick
```

**3. Or full setup:**

```bash
# Clone repo
git clone https://github.com/MezaSamano/maki.git
cd maki

# Build
cargo build --bin lort-compress --release

# Run benchmark
./benchmark.sh --large
```

**4. Download results:**

```bash
# From your local machine
scp user@instance-ip:~/maki/benchmark_results/*.csv ./
scp user@instance-ip:~/maki/benchmark_results/*.log ./
```

---

## Performance Baselines

Expected results on different hardware:

### AWS t3.xlarge (4 vCPU, 16GB RAM)

| Model | Time | Ratio | Bits/W |
|-------|------|-------|--------|
| Qwen 0.5B | 3-5 min | 7.2x | 2.2 |
| TinyLlama 1.1B | 10-15 min | 7.2x | 2.2 |
| Phi-2 (2.7B) | 25-35 min | 7.2x | 2.2 |

### AWS c5.2xlarge (8 vCPU, 16GB RAM)

| Model | Time | Ratio | Bits/W |
|-------|------|-------|--------|
| Qwen 0.5B | 2-3 min | 7.2x | 2.2 |
| TinyLlama 1.1B | 6-8 min | 7.2x | 2.2 |
| Phi-2 (2.7B) | 15-20 min | 7.2x | 2.2 |

### Hetzner CCX33 (8 vCPU, 32GB RAM)

| Model | Time | Ratio | Bits/W |
|-------|------|-------|--------|
| Qwen 0.5B | 2 min | 7.2x | 2.2 |
| TinyLlama 1.1B | 5-7 min | 7.2x | 2.2 |
| Phi-2 (2.7B) | 12-18 min | 7.2x | 2.2 |
| Mistral 7B | 40-50 min | 7.2x | 2.2 |

**Performance scales linearly** with CPU core count.

---

## Advanced Usage

### Custom Metrics Collection

Modify `benchmark.sh` to add custom metrics:

```bash
# Add to benchmark_model() function

# GPU usage (if available)
if command -v nvidia-smi &> /dev/null; then
    local gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "GPU Memory: ${gpu_mem} MB"
fi

# Disk I/O
local io_read=$(iostat -d -k 1 1 | awk '/sda/ {print $3}')
local io_write=$(iostat -d -k 1 1 | awk '/sda/ {print $4}')
```

### Automated Testing Pipeline

```bash
#!/bin/bash
# cloud_benchmark_pipeline.sh

INSTANCES=(
    "ubuntu@aws-instance-1"
    "root@hetzner-instance"
    "user@gcp-instance"
)

for instance in "${INSTANCES[@]}"; do
    echo "Testing on: $instance"
    
    # Run benchmark remotely
    ssh $instance "cd ~/maki && ./benchmark.sh --standard"
    
    # Download results
    scp $instance:~/maki/benchmark_results/results_*.csv ./results_$(echo $instance | cut -d@ -f2).csv
done

# Compare all results
python3 analyze_benchmark.py results_*.csv --compare
```

### Continuous Benchmarking

Set up cron job for regular testing:

```bash
# Add to crontab
crontab -e

# Run benchmark every day at 2 AM
0 2 * * * cd /home/user/maki && ./benchmark.sh --quick >> /var/log/lort_benchmark.log 2>&1
```

---

## Troubleshooting

### Missing Dependencies

**Symptoms:**
```
./benchmark.sh: line 197: /usr/bin/time: No such file or directory
./benchmark.sh: line 223: bc: command not found
```

**Solution:**

```bash
# Ubuntu/Debian
apt update && apt install -y bc time

# RHEL/CentOS/Fedora
yum install -y bc time

# Alpine
apk add bc coreutils

# Verify installation
which bc time
```

---

### Script Fails to Run

```bash
# Ensure executable
chmod +x benchmark.sh

# Check dependencies
which cargo rustc bc time

# Run with verbose output
bash -x benchmark.sh --quick
```

### Out of Memory

```bash
# Use quick mode
./benchmark.sh --quick

# Or add swap (see CLOUD_SETUP.md)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Compression Fails

Check logs in `benchmark_results/benchmark_TIMESTAMP.log`:

```bash
# View latest log
tail -f benchmark_results/benchmark_*.log

# Search for errors
grep -i error benchmark_results/benchmark_*.log
```

---

## Cost Estimation

Running benchmarks on different providers:

| Provider | Instance | Standard (3 models, ~30min) | Large (4 models, ~90min) |
|----------|----------|----------------------------|--------------------------|
| **Hetzner** | CCX23 (8c/32GB) | $0.03 | $0.08 |
| **DigitalOcean** | CPU-Opt 16GB | $0.07 | $0.21 |
| **AWS** | c5.2xlarge | $0.17 | $0.51 |
| **GCP** | c2-standard-8 | $0.19 | $0.57 |
| **RunPod** | CPU 8c/16GB | $0.05 | $0.15 |

💡 **Tip**: Use spot/preemptible instances for 60-80% savings

---

## Best Practices

1. **Always run --quick first** to validate setup
2. **Use release builds** (`cargo build --release`)
3. **Monitor resources** with `htop` during benchmarks
4. **Save results** before terminating instances
5. **Compare apples-to-apples**: Same models, different hardware
6. **Run multiple times** for statistical significance
7. **Document instance specs** in results

---

## Example Benchmark Session

```bash
# 1. Launch cloud instance (16GB RAM, 8 cores)
ssh user@cloud-instance

# 2. Setup
git clone https://github.com/MezaSamano/maki.git
cd maki
cargo build --bin lort-compress --release

# 3. Quick validation
./benchmark.sh --quick
# ✓ Qwen 0.5B: 2min, 7.2x ratio

# 4. Full benchmark
./benchmark.sh --standard
# ✓ 3 models in 28 minutes

# 5. Analyze
python3 analyze_benchmark.py benchmark_results/results_*.csv
# Shows detailed stats

# 6. Download results (from local machine)
scp user@cloud-instance:~/maki/benchmark_results/*.csv ./

# 7. Terminate instance
# Don't forget!
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday 2 AM

jobs:
  benchmark:
    runs-on: ubuntu-latest-8-cores
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Build
        run: cargo build --bin lort-compress --release
      
      - name: Run Benchmark
        run: ./benchmark.sh --quick
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results/
```

---

## Next Steps

- **Optimize compression**: Experiment with `--min-dim` settings
- **Profile code**: Use `cargo flamegraph` for bottleneck analysis
- **Test edge cases**: Unusual model architectures
- **Benchmark inference**: Once implemented (see roadmap)

---

**Happy benchmarking! 🚀**

*For cloud setup instructions, see [CLOUD_SETUP.md](CLOUD_SETUP.md)*  
*For usage examples, see [EXAMPLES.md](EXAMPLES.md)*  
*For memory mapping details, see [MEMORY_MAPPING.md](MEMORY_MAPPING.md)*
