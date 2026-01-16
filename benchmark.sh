#!/bin/bash
#
# LoRT Compression Benchmark Suite
# 
# Automatically tests compression across multiple models and tracks:
# - Compression time
# - Memory usage
# - Compression ratio
# - File sizes
# - System specs
#
# Usage:
#   ./benchmark.sh                    # Run all benchmarks
#   ./benchmark.sh --quick            # Quick test (small models only)
#   ./benchmark.sh --large            # Include 7B+ models
#   ./benchmark.sh --model "Qwen/..."  # Single model test

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/results_${TIMESTAMP}.csv"
LOG_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.log"
COMPRESSED_DIR="${RESULTS_DIR}/compressed"

# Model sets
QUICK_MODELS=(
    "Qwen/Qwen2.5-0.5B"
)

STANDARD_MODELS=(
    "Qwen/Qwen2.5-0.5B"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "microsoft/phi-2"
)

LARGE_MODELS=(
    "Qwen/Qwen2.5-0.5B"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "microsoft/phi-2"
    "mistralai/Mistral-7B-v0.1"
)

# Parse arguments
MODE="standard"
CUSTOM_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --large)
            MODE="large"
            shift
            ;;
        --model)
            CUSTOM_MODEL="$2"
            MODE="custom"
            shift 2
            ;;
        --help)
            echo "LoRT Compression Benchmark Suite"
            echo ""
            echo "Usage: ./benchmark.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Run quick benchmark (Qwen 0.5B only)"
            echo "  --large           Include large models (7B+)"
            echo "  --model MODEL     Benchmark specific model"
            echo "  --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  ./benchmark.sh                           # Standard benchmark"
            echo "  ./benchmark.sh --quick                   # Quick test"
            echo "  ./benchmark.sh --model 'Qwen/Qwen2.5-0.5B'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Select model set
if [ "$MODE" == "quick" ]; then
    MODELS=("${QUICK_MODELS[@]}")
elif [ "$MODE" == "large" ]; then
    MODELS=("${LARGE_MODELS[@]}")
elif [ "$MODE" == "custom" ]; then
    MODELS=("$CUSTOM_MODEL")
else
    MODELS=("${STANDARD_MODELS[@]}")
fi

# Functions
log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $1" | tee -a "$LOG_FILE"
}

# Get system info
get_system_info() {
    local cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
    local cpu_cores=$(nproc)
    local total_ram=$(free -h | awk '/^Mem:/ {print $2}')
    local available_ram=$(free -h | awk '/^Mem:/ {print $7}')
    local os_info=$(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")
    
    echo "CPU: $cpu_model"
    echo "CPU Cores: $cpu_cores"
    echo "Total RAM: $total_ram"
    echo "Available RAM: $available_ram"
    echo "OS: $os_info"
    echo "Rust: $(rustc --version)"
    echo "Cargo: $(cargo --version)"
}

# Get current memory usage
get_memory_usage() {
    local mem_used=$(free -m | awk '/^Mem:/ {print $3}')
    echo "$mem_used"
}

# Calculate file size in MB
get_file_size_mb() {
    local file=$1
    if [ -f "$file" ]; then
        local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        echo "scale=2; $size / 1048576" | bc
    else
        echo "0"
    fi
}

# Get model cache size
get_model_cache_size() {
    local model=$1
    local cache_dir="$HOME/.cache/huggingface/hub"
    local model_hash=$(echo "$model" | sed 's/\//-/g')
    
    # Find matching directory
    local model_dir=$(find "$cache_dir" -type d -name "*${model_hash}*" 2>/dev/null | head -n 1)
    
    if [ -n "$model_dir" ]; then
        local size=$(du -sm "$model_dir" 2>/dev/null | cut -f1)
        echo "$size"
    else
        echo "0"
    fi
}

# Benchmark a single model
benchmark_model() {
    local model=$1
    local output_name=$(echo "$model" | tr '/' '-')
    local output_file="${COMPRESSED_DIR}/${output_name}.lort"
    
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Benchmarking: ${YELLOW}${model}${NC}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Memory before
    local mem_before=$(get_memory_usage)
    
    # Run compression with timing
    local start_time=$(date +%s)
    
    log "Starting compression..."
    
    # Capture both stdout and stderr, and time output
    local compress_output=$(mktemp)
    local time_output=$(mktemp)
    
    if /usr/bin/time -v cargo run --bin lort-compress --release -- \
        --model "$model" \
        --output "$output_file" \
        2>&1 | tee "$compress_output" 2>&1; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Memory after
        local mem_after=$(get_memory_usage)
        local mem_used=$((mem_after - mem_before))
        
        # Extract metrics from output
        local original_size=$(grep "Original size" "$compress_output" | grep -oP '\d+\.\d+' | head -1 || echo "0")
        local compressed_size=$(grep "Compressed size" "$compress_output" | grep -oP '\d+\.\d+' | head -1 || echo "0")
        local compression_ratio=$(grep "Compression Ratio" "$compress_output" | grep -oP '\d+\.\d+' | head -1 || echo "0")
        local bits_per_weight=$(grep "Bits per weight" "$compress_output" | grep -oP '\d+\.\d+' | head -1 || echo "0")
        
        # File size verification
        local actual_file_size=$(get_file_size_mb "$output_file")
        
        # Model cache size
        local cache_size=$(get_model_cache_size "$model")
        
        # Peak memory from /usr/bin/time
        local peak_mem=$(grep "Maximum resident set size" "$compress_output" | grep -oP '\d+' || echo "0")
        peak_mem=$(echo "scale=2; $peak_mem / 1024" | bc)  # Convert to MB
        
        # Save results to CSV
        echo "${model},${duration},${mem_used},${peak_mem},${original_size},${compressed_size},${compression_ratio},${bits_per_weight},${actual_file_size},${cache_size},success" >> "$RESULTS_FILE"
        
        log_success "Compression complete!"
        log "  Time: ${duration}s ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
        log "  Original: ${original_size} MB"
        log "  Compressed: ${compressed_size} MB (actual: ${actual_file_size} MB)"
        log "  Ratio: ${compression_ratio}x"
        log "  Bits/weight: ${bits_per_weight}"
        log "  Memory used: ${mem_used} MB (peak: ${peak_mem} MB)"
        
    else
        log_error "Compression failed for $model"
        echo "${model},0,0,0,0,0,0,0,0,0,failed" >> "$RESULTS_FILE"
    fi
    
    # Cleanup
    rm -f "$compress_output" "$time_output"
    
    log ""
}

# Generate summary report
generate_report() {
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Benchmark Summary"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Calculate totals
    local total_time=0
    local total_original=0
    local total_compressed=0
    local successful=0
    local failed=0
    
    while IFS=, read -r model duration mem peak orig comp ratio bits actual cache status; do
        if [ "$status" == "success" ]; then
            total_time=$((total_time + duration))
            total_original=$(echo "$total_original + $orig" | bc)
            total_compressed=$(echo "$total_compressed + $comp" | bc)
            ((successful++))
        else
            ((failed++))
        fi
    done < <(tail -n +2 "$RESULTS_FILE")
    
    local total_tests=$((successful + failed))
    local avg_time=0
    if [ $successful -gt 0 ]; then
        avg_time=$(echo "scale=2; $total_time / $successful" | bc)
    fi
    
    local total_ratio=0
    if [ $(echo "$total_original > 0" | bc) -eq 1 ]; then
        total_ratio=$(echo "scale=2; $total_original / $total_compressed" | bc)
    fi
    
    log "Total Tests: $total_tests"
    log "Successful: ${GREEN}${successful}${NC}"
    [ $failed -gt 0 ] && log "Failed: ${RED}${failed}${NC}"
    log ""
    log "Total Time: $(printf '%02d:%02d:%02d' $((total_time/3600)) $((total_time%3600/60)) $((total_time%60)))"
    log "Average Time: ${avg_time}s per model"
    log ""
    log "Total Original Size: ${total_original} MB"
    log "Total Compressed Size: ${total_compressed} MB"
    log "Overall Compression Ratio: ${total_ratio}x"
    log ""
    log "Results saved to: ${RESULTS_FILE}"
    log "Log saved to: ${LOG_FILE}"
    log "Compressed files in: ${COMPRESSED_DIR}"
}

# Main execution
main() {
    # Setup
    mkdir -p "$RESULTS_DIR" "$COMPRESSED_DIR"
    
    # Header
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   LoRT Compression Benchmark Suite        ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
    echo ""
    
    # System info
    log "System Information:"
    get_system_info | while read line; do
        log "  $line"
    done
    log ""
    
    log "Benchmark Mode: ${YELLOW}${MODE}${NC}"
    log "Models to test: ${#MODELS[@]}"
    for model in "${MODELS[@]}"; do
        log "  - $model"
    done
    log ""
    
    # CSV header
    echo "model,duration_sec,mem_used_mb,peak_mem_mb,original_mb,compressed_mb,compression_ratio,bits_per_weight,file_size_mb,cache_mb,status" > "$RESULTS_FILE"
    
    # Check if binary exists
    if [ ! -f "target/release/lort-compress" ]; then
        log_warning "Release binary not found. Building..."
        cargo build --bin lort-compress --release
    fi
    
    # Run benchmarks
    local overall_start=$(date +%s)
    
    for model in "${MODELS[@]}"; do
        benchmark_model "$model"
    done
    
    local overall_end=$(date +%s)
    local overall_duration=$((overall_end - overall_start))
    
    # Generate report
    generate_report
    
    log ""
    log "Total benchmark time: $(printf '%02d:%02d:%02d' $((overall_duration/3600)) $((overall_duration%3600/60)) $((overall_duration%60)))"
    log_success "Benchmark complete!"
    echo ""
}

# Run main
main
