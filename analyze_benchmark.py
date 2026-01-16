#!/usr/bin/env python3
"""
LoRT Benchmark Results Analyzer

Analyzes benchmark CSV files and generates detailed reports with visualizations.

Usage:
    python analyze_benchmark.py results.csv
    python analyze_benchmark.py results.csv --plot
    python analyze_benchmark.py results.csv --compare results2.csv
"""

import sys
import csv
import argparse
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Any


def format_time(seconds: int) -> str:
    """Format seconds into human-readable time."""
    return str(timedelta(seconds=seconds))


def format_size(mb: float) -> str:
    """Format MB into human-readable size."""
    if mb < 1024:
        return f"{mb:.2f} MB"
    else:
        gb = mb / 1024
        return f"{gb:.2f} GB"


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load benchmark results from CSV."""
    results = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['duration_sec'] = int(row['duration_sec'])
            row['mem_used_mb'] = int(row['mem_used_mb'])
            row['peak_mem_mb'] = float(row['peak_mem_mb'])
            row['original_mb'] = float(row['original_mb'])
            row['compressed_mb'] = float(row['compressed_mb'])
            row['compression_ratio'] = float(row['compression_ratio'])
            row['bits_per_weight'] = float(row['bits_per_weight'])
            row['file_size_mb'] = float(row['file_size_mb'])
            row['cache_mb'] = int(row['cache_mb'])
            results.append(row)
    return results


def print_table(results: List[Dict[str, Any]]):
    """Print results as formatted table."""
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)
    
    # Header
    print(f"{'Model':<40} {'Time':<12} {'Original':<12} {'Compressed':<12} {'Ratio':<8} {'Bits/W':<8} {'Status':<10}")
    print("-" * 120)
    
    # Rows
    for r in results:
        model_short = r['model'].split('/')[-1][:38]  # Truncate long names
        status = "✓" if r['status'] == 'success' else "✗"
        
        print(f"{model_short:<40} {format_time(r['duration_sec']):<12} "
              f"{format_size(r['original_mb']):<12} {format_size(r['compressed_mb']):<12} "
              f"{r['compression_ratio']:.2f}x{'':<4} {r['bits_per_weight']:.2f}{'':<4} {status:<10}")
    
    print("-" * 120)


def print_summary(results: List[Dict[str, Any]]):
    """Print summary statistics."""
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"\nTests Run: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if not successful:
        print("\nNo successful benchmarks to analyze.")
        return
    
    # Time statistics
    total_time = sum(r['duration_sec'] for r in successful)
    avg_time = total_time / len(successful)
    min_time = min(r['duration_sec'] for r in successful)
    max_time = max(r['duration_sec'] for r in successful)
    
    print(f"\nTime Statistics:")
    print(f"  Total: {format_time(total_time)}")
    print(f"  Average: {format_time(int(avg_time))}")
    print(f"  Min: {format_time(min_time)} ({min(successful, key=lambda x: x['duration_sec'])['model']})")
    print(f"  Max: {format_time(max_time)} ({max(successful, key=lambda x: x['duration_sec'])['model']})")
    
    # Size statistics
    total_original = sum(r['original_mb'] for r in successful)
    total_compressed = sum(r['compressed_mb'] for r in successful)
    overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
    
    print(f"\nSize Statistics:")
    print(f"  Total Original: {format_size(total_original)}")
    print(f"  Total Compressed: {format_size(total_compressed)}")
    print(f"  Overall Compression Ratio: {overall_ratio:.2f}x")
    print(f"  Space Saved: {format_size(total_original - total_compressed)} ({((1 - total_compressed/total_original) * 100):.1f}%)")
    
    # Compression ratio statistics
    avg_ratio = sum(r['compression_ratio'] for r in successful) / len(successful)
    min_ratio = min(r['compression_ratio'] for r in successful)
    max_ratio = max(r['compression_ratio'] for r in successful)
    
    print(f"\nCompression Ratio:")
    print(f"  Average: {avg_ratio:.2f}x")
    print(f"  Min: {min_ratio:.2f}x ({min(successful, key=lambda x: x['compression_ratio'])['model']})")
    print(f"  Max: {max_ratio:.2f}x ({max(successful, key=lambda x: x['compression_ratio'])['model']})")
    
    # Bits per weight statistics
    avg_bits = sum(r['bits_per_weight'] for r in successful) / len(successful)
    
    print(f"\nBits per Weight:")
    print(f"  Average: {avg_bits:.2f} bits")
    print(f"  Min: {min(r['bits_per_weight'] for r in successful):.2f} bits")
    print(f"  Max: {max(r['bits_per_weight'] for r in successful):.2f} bits")
    
    # Memory statistics
    avg_mem = sum(r['mem_used_mb'] for r in successful) / len(successful)
    avg_peak = sum(r['peak_mem_mb'] for r in successful) / len(successful)
    
    print(f"\nMemory Usage:")
    print(f"  Average Used: {format_size(avg_mem)}")
    print(f"  Average Peak: {format_size(avg_peak)}")
    print(f"  Max Peak: {format_size(max(r['peak_mem_mb'] for r in successful))} ({max(successful, key=lambda x: x['peak_mem_mb'])['model']})")


def print_performance_analysis(results: List[Dict[str, Any]]):
    """Analyze performance patterns."""
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        return
    
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Time per GB
    print("\nTime Efficiency (seconds per GB original):")
    for r in sorted(successful, key=lambda x: x['duration_sec'] / (x['original_mb'] / 1024)):
        gb = r['original_mb'] / 1024
        sec_per_gb = r['duration_sec'] / gb if gb > 0 else 0
        print(f"  {r['model'].split('/')[-1]:<40} {sec_per_gb:>8.1f} sec/GB")
    
    # Compression efficiency
    print("\nCompression Efficiency (ratio vs time):")
    for r in sorted(successful, key=lambda x: x['compression_ratio'] / x['duration_sec'], reverse=True):
        efficiency = r['compression_ratio'] / r['duration_sec'] if r['duration_sec'] > 0 else 0
        print(f"  {r['model'].split('/')[-1]:<40} {efficiency:>8.4f} ratio/sec")


def compare_results(files: List[str]):
    """Compare results from multiple benchmark runs."""
    all_results = {}
    for filepath in files:
        run_name = Path(filepath).stem
        all_results[run_name] = load_results(filepath)
    
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON")
    print("=" * 100)
    
    # Find common models
    all_models = set()
    for results in all_results.values():
        all_models.update(r['model'] for r in results)
    
    for model in sorted(all_models):
        print(f"\nModel: {model}")
        print(f"{'Run':<30} {'Time':<15} {'Ratio':<10} {'Bits/W':<10} {'Status'}")
        print("-" * 80)
        
        for run_name, results in all_results.items():
            model_results = [r for r in results if r['model'] == model]
            if model_results:
                r = model_results[0]
                status = "✓" if r['status'] == 'success' else "✗"
                print(f"{run_name:<30} {format_time(r['duration_sec']):<15} "
                      f"{r['compression_ratio']:.2f}x{'':<6} {r['bits_per_weight']:.2f}{'':<6} {status}")
            else:
                print(f"{run_name:<30} {'N/A':<15} {'N/A':<10} {'N/A':<10} -")


def plot_results(results: List[Dict[str, Any]]):
    """Generate plots (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n⚠ Matplotlib not installed. Install with: pip install matplotlib")
        return
    
    successful = [r for r in results if r['status'] == 'success']
    if not successful:
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LoRT Benchmark Results', fontsize=16, fontweight='bold')
    
    models = [r['model'].split('/')[-1] for r in successful]
    
    # 1. Compression Time
    times = [r['duration_sec'] / 60 for r in successful]  # Convert to minutes
    ax1.barh(models, times, color='steelblue')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_title('Compression Time')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Compression Ratio
    ratios = [r['compression_ratio'] for r in successful]
    ax2.barh(models, ratios, color='forestgreen')
    ax2.set_xlabel('Compression Ratio (x)')
    ax2.set_title('Compression Ratio')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. File Sizes
    original = [r['original_mb'] for r in successful]
    compressed = [r['compressed_mb'] for r in successful]
    x = np.arange(len(models))
    width = 0.35
    ax3.bar(x - width/2, original, width, label='Original', color='coral')
    ax3.bar(x + width/2, compressed, width, label='Compressed', color='lightseagreen')
    ax3.set_ylabel('Size (MB)')
    ax3.set_title('File Sizes')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Memory Usage
    mem_used = [r['mem_used_mb'] for r in successful]
    peak_mem = [r['peak_mem_mb'] for r in successful]
    x = np.arange(len(models))
    ax4.bar(x - width/2, mem_used, width, label='Used', color='mediumpurple')
    ax4.bar(x + width/2, peak_mem, width, label='Peak', color='tomato')
    ax4.set_ylabel('Memory (MB)')
    ax4.set_title('Memory Usage')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'benchmark_results/analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze LoRT benchmark results')
    parser.add_argument('results', nargs='+', help='Benchmark results CSV file(s)')
    parser.add_argument('--plot', action='store_true', help='Generate plots (requires matplotlib)')
    parser.add_argument('--compare', action='store_true', help='Compare multiple benchmark runs')
    
    args = parser.parse_args()
    
    if args.compare and len(args.results) > 1:
        compare_results(args.results)
    else:
        results = load_results(args.results[0])
        print_table(results)
        print_summary(results)
        print_performance_analysis(results)
        
        if args.plot:
            plot_results(results)


if __name__ == '__main__':
    main()
