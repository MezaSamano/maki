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

# Check dependencies
check_dependencies() {
    local missing=()
    
    command -v cargo >/dev/null 2>&1 || missing+=("cargo (Rust toolchain)")
    command -v bc >/dev/null 2>&1 || missing+=("bc")
    command -v time >/dev/null 2>&1 || missing+=("time")
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required dependencies:"
        for dep in "${missing[@]}"; do
            log_error "  - $dep"
        done
        echo ""
        log "Install with:"
        log "  Ubuntu/Debian: apt update && apt install -y bc time"
        log "  RHEL/CentOS:   yum install -y bc time"
        log "  Alpine:        apk add bc coreutils"
        echo ""
        return 1
    fi
    return 0
}

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
SKIP_COMPRESS=false
MEASURE_PPL=false

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
        --skip-compress)
            SKIP_COMPRESS=true
            shift
            ;;
        --benchmark-only)
            SKIP_COMPRESS=true
            shift
            ;;
        --measure-ppl)
            MEASURE_PPL=true
            shift
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
            echo "  --skip-compress   Skip compression, use existing .lort files"
            echo "  --benchmark-only  Same as --skip-compress"
            echo "  --measure-ppl     Measure perplexity (requires Python & transformers)"
            echo "  --help            Show this help"
            echo ""
            echo "Examples:"
            echo "  ./benchmark.sh                           # Standard benchmark"
            echo "  ./benchmark.sh --quick                   # Quick test"
            echo "  ./benchmark.sh --model 'Qwen/Qwen2.5-0.5B'"
            echo "  ./benchmark.sh --skip-compress           # Benchmark existing files"
            echo "  ./benchmark.sh --measure-ppl             # Include perplexity metrics"
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
        if command -v bc >/dev/null 2>&1; then
            echo "scale=2; $size / 1048576" | bc
        else
            awk "BEGIN {printf \"%.2f\", $size / 1048576}"
        fi
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
    
    # Check if skipping compression
    if [ "$SKIP_COMPRESS" = true ]; then
        if [ ! -f "$output_file" ]; then
            log_error "Compressed file not found: $output_file"
            log_error "Run without --skip-compress first to create the file"
            echo "${model},0,0,0,0,0,0,0,0,0,not_found" >> "$RESULTS_FILE"
            return 1
        fi
        
        log "Using existing compressed file: $output_file"
        
        # Get file info from existing file
        local actual_file_size=$(get_file_size_mb "$output_file")
        local cache_size=$(get_model_cache_size "$model")
        
        # Extract info from filename or use defaults
        log "Skipping compression, analyzing existing file..."
        
        # Save minimal results
        echo "${model},0,0,0,0,${actual_file_size},0,0,${actual_file_size},${cache_size},skipped" >> "$RESULTS_FILE"
        
        log_success "File analysis complete!"
        log "  Compressed file: ${actual_file_size} MB"
        log ""
        
        # Measure perplexity if requested
        if [ "$MEASURE_PPL" = true ]; then
            measure_perplexity "$model" "$output_file"
        fi
        
        return 0
    fi
    
    # 
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Benchmarking: ${YELLOW}${model}${NC}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Memory before
    local mem_before=$(get_memory_usage)
    
    # Run compression with timing
    local start_time=$(date +%s)
    
    log "Starting compression..."
    
    # Capture both stdout and stderr
    local compress_output=$(mktemp)
    
    if cargo run --bin lort-compress --release -- \
        --model "$model" \
        --output "$output_file" \
        2>&1 | tee "$compress_output"; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Memory after
        local mem_after=$(get_memory_usage)
        local mem_used=$((mem_after - mem_before))
        
        # Extract metrics from output
        local original_size=$(grep "Original size:" "$compress_output" | grep -oP '\d+\.\d+' | head -1 || echo "0")
        local compressed_size=$(grep "Compressed size:" "$compress_output" | grep -oP '\d+\.\d+' | head -1 || echo "0")
        local compression_ratio=$(grep "Compression ratio:" "$compress_output" | grep -oP '\d+\.\d+' | head -1 || echo "0")
        local bits_per_weight=$(grep "Bits per weight:" "$compress_output" | grep -oP '\d+\.\d+' | head -1 || echo "0")
        
        # Actual file size
        local actual_file_size=$(get_file_size_mb "$output_file")
        
        # Model cache size
        local cache_size=$(get_model_cache_size "$model")
        
        # Peak memory (estimate from system if time -v unavailable)
        local peak_mem="0"
        if command -v bc >/dev/null 2>&1; then
            # Try to get from /usr/bin/time output if available
            local peak_kb=$(grep "Maximum resident set size" "$compress_output" | grep -oP '\d+' 2>/dev/null || echo "0")
            if [ "$peak_kb" != "0" ]; then
                peak_mem=$(echo "scale=2; $peak_kb / 1024" | bc)
            else
                # Estimate from memory usage delta
                peak_mem=$(echo "scale=2; $mem_used * 1.2" | bc)
            fi
        else
            # Fallback: use current memory delta * 1.2 as estimate
            peak_mem=$(awk "BEGIN {printf \"%.2f\", $mem_used * 1.2}")
        fi
        
        # Save results to CSV
        echo "${model},${duration},${mem_used},${peak_mem},${original_size},${compressed_size},${compression_ratio},${bits_per_weight},${actual_file_size},${cache_size},success" >> "$RESULTS_FILE"
        
        log_success "Compression complete!"
        log "  Time: ${duration}s ($(printf '%02d:%02d:%02d' $((duration/3600)) $((duration%3600/60)) $((duration%60))))"
        log "  Original: ${original_size} MB"
        log "  Compressed: ${compressed_size} MB (actual: ${actual_file_size} MB)"
        log "  Ratio: ${compression_ratio}x"
        log "  Bits/weight: ${bits_per_weight}"
        log "  Memory used: ${mem_used} MB (peak: ${peak_mem} MB)"
        
        # Measure perplexity if requested
        if [ "$MEASURE_PPL" = true ]; then
            measure_perplexity "$model" "$output_file"
        fi
        
    else
        log_error "Compression failed for $model"
        echo "${model},0,0,0,0,0,0,0,0,0,failed" >> "$RESULTS_FILE"
    fi
    
    # Cleanup
    rm -f "$compress_output" "$time_output"
    
    log ""
}

# Measure perplexity (requires Python script)
measure_perplexity() {
    local model=$1
    local compressed_file=$2
    
    log "📊 Measuring perplexity..."
    
    # Check if Python is available
    if ! command -v python3 >/dev/null 2>&1; then
        log_warning "Python3 not found, skipping perplexity measurement"
        return 1
    fi
    
    # Check if evaluation script exists
    if [ ! -f "evaluate_ppl.py" ]; then
        log_warning "evaluate_ppl.py not found, skipping perplexity measurement"
        log "  Create evaluate_ppl.py to enable perplexity testing"
        return 1
    fi
    
    # Run perplexity evaluation
    local ppl_output=$(mktemp)
    if python3 evaluate_ppl.py --model "$model" --compressed "$compressed_file" 2>&1 | tee "$ppl_output"; then
        local original_ppl=$(grep "Original PPL:" "$ppl_output" | grep -oP '\d+\.\d+' || echo "N/A")
        local compressed_ppl=$(grep "Compressed PPL:" "$ppl_output" | grep -oP '\d+\.\d+' || echo "N/A")
        local ppl_delta=$(grep "Delta:" "$ppl_output" | grep -oP '\d+\.\d+' || echo "N/A")
        
        log "  Original PPL: ${original_ppl}"
        log "  Compressed PPL: ${compressed_ppl}"
        log "  Delta: ${ppl_delta}"
    else
        log_warning "Perplexity measurement failed"
    fi
    
    rm -f "$ppl_output"
}

# Benchmark wrapper - processes models list
run_benchmarks() {
    local models=("$@")
    
    for model in "${models[@]}"; do
        benchmark_model "$model"
    done
}

# Generate summary report
generate_report() {
    if [ ! -f "$RESULTS_FILE" ]; then
        log_error "No results file found: $RESULTS_FILE"
        return 1
    fi
    
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "${YELLOW}BENCHMARK SUMMARY${NC}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Count successful vs failed
    local total=$(wc -l < "$RESULTS_FILE")
    local successful=$(grep -c "success\|skipped" "$RESULTS_FILE" || echo "0")
    local failed=$((total - successful))
    
    log "Models tested: ${total}"
    log "Successful: ${GREEN}${successful}${NC}"
    if [ "$failed" -gt 0 ]; then
        log "Failed: ${RED}${failed}${NC}"
    fi
    
    log ""
    
    # Calculate totals (skip header and failed runs)
    if command -v bc >/dev/null 2>&1; then
        local total_time=$(awk -F',' '$11=="success" {sum+=$2} END {print sum}' "$RESULTS_FILE")
        local total_original=$(awk -F',' '$11=="success" {sum+=$5} END {print sum}' "$RESULTS_FILE")
        local total_compressed=$(awk -F',' '$11=="success" {sum+=$6} END {print sum}' "$RESULTS_FILE")
        local avg_ratio=$(echo "scale=2; $total_original / $total_compressed" | bc)
        
        log "Total compression time: ${total_time}s"
        log "Total original size: ${total_original} MB"
        log "Total compressed size: ${total_compressed} MB"
        log "Average compression ratio: ${avg_ratio}x"
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
    
    [ "$SKIP_COMPRESS" = true ] && log "Skip Compression: ${YELLOW}YES${NC} (using existing files)"
    [ "$MEASURE_PPL" = true ] && log "Measure Perplexity: ${YELLOW}YES${NC}"
    # Calculate totals
    local total_time=0
    local total_original=0
    local total_compressed=0
    local successful=0
    local failed=0
    
    while IFS=, read -r model duration mem peak orig comp ratio bits actual cache status; do
        if [ "$status" == "success" ]; then
            total_time=$((total_time + duration))
            if command -v bc >/dev/null 2>&1; then
                total_original=$(echo "$total_original + $orig" | bc)
                total_compressed=$(echo "$total_compressed + $comp" | bc)
            else
                total_original=$(awk "BEGIN {printf \"%.2f\", $total_original + $orig}")
                total_compressed=$(awk "BEGIN {printf \"%.2f\", $total_compressed + $comp}")
            fi
            ((successful++))
        else
            ((failed++))
        fi
    done < <(tail -n +2 "$RESULTS_FILE")
    
    local total_tests=$((successful + failed))
    local avg_time="0.00"
    if [ $successful -gt 0 ]; then
        if command -v bc >/dev/null 2>&1; then
            avg_time=$(echo "scale=2; $total_time / $successful" | bc)
        else
            avg_time=$(awk "BEGIN {printf \"%.2f\", $total_time / $successful}")
        fi
    fi
    
    local total_ratio="0.00"
    if command -v bc >/dev/null 2>&1; then
        if [ $(echo "$total_original > 0" | bc) -eq 1 ]; then
            total_ratio=$(echo "scale=2; $total_original / $total_compressed" | bc)
        fi
    else
        if awk "BEGIN {exit !($total_original > 0)}"; then
            total_ratio=$(awk "BEGIN {printf \"%.2f\", $total_original / $total_compressed}")
        fi
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
    
    # Check dependencies
    if ! check_dependencies; then
        log_error "Cannot proceed without required dependencies."
        exit 1
    fi
    
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
