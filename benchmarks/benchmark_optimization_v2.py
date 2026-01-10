#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization Performance: CUDA vs Numba
Compare CUDA and Numba strategies over 20 iterations of optimization
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
import time
from pathlib import Path

# Set UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Add paths
sys.path.insert(0, 'src')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

from LCNv1.api import LCNSolver


def benchmark_optimization(strategy, json_path, iterations=20, runs=3):
    """Benchmark a strategy with optimization iterations"""
    try:
        times = []
        final_k = None
        
        for _ in range(runs):
            solver = LCNSolver(strategy=strategy)
            solver.load_from_json(json_path)
            
            start = time.perf_counter()
            result = solver.optimize(iterations=iterations)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if final_k is None:
                final_k = result.k
        
        avg_time = sum(times) / len(times)
        
        return {
            'avg_time': avg_time,
            'min_time': min(times),
            'max_time': max(times),
            'k': final_k,
        }
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_optimization_benchmark():
    """Run optimization benchmark for CUDA vs Numba"""
    
    instances = [
        ('15-nodes', 'live-2025-example-instances/15-nodes.json', 15),
        ('70-nodes', 'live-2025-example-instances/70-nodes.json', 70),
        ('100-nodes', 'live-2025-example-instances/100-nodes.json', 100),
    ]
    
    iterations = 20
    
    results = {
        'numba': [],
        'cuda': []
    }
    node_counts = []
    instance_names = []
    
    print("="*80)
    print(f"Optimization Performance Benchmark: CUDA vs Numba")
    print(f"Iterations: {iterations}")
    print("="*80)
    
    for name, path, nodes in instances:
        if not Path(path).exists():
            print(f"\n⚠️  Skipping missing file: {path}")
            continue
        
        node_counts.append(nodes)
        instance_names.append(name)
        
        print(f"\n{name} ({nodes} nodes, {iterations} iterations):")
        print("-" * 40)
        
        # Benchmark Numba
        print("  Testing Numba...", end=' ', flush=True)
        numba_result = benchmark_optimization('numba', path, iterations=iterations)
        if numba_result:
            results['numba'].append(numba_result['avg_time'])
            print(f"✓ {numba_result['avg_time']:.3f}s (K={numba_result['k']})")
        else:
            results['numba'].append(0)
            print("✗ Failed")
        
        # Benchmark CUDA
        print("  Testing CUDA...", end=' ', flush=True)
        cuda_result = benchmark_optimization('cuda', path, iterations=iterations)
        if cuda_result:
            results['cuda'].append(cuda_result['avg_time'])
            print(f"✓ {cuda_result['avg_time']:.3f}s (K={cuda_result['k']})")
        else:
            results['cuda'].append(0)
            print("✗ Failed")
        
        # Calculate speedup
        if numba_result and cuda_result and cuda_result['avg_time'] > 0:
            speedup = numba_result['avg_time'] / cuda_result['avg_time']
            print(f"  Speedup: {speedup:.2f}x (CUDA vs Numba)")
    
    return node_counts, instance_names, results


def plot_optimization_results(node_counts, instance_names, results):
    """Plot optimization benchmark results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Grouped Bar Chart
    x = np.arange(len(node_counts))
    width = 0.35
    
    colors = {
        'numba': '#F18F01',
        'cuda': '#06A77D'
    }
    
    numba_bars = ax1.bar(x - width/2, results['numba'], width, 
                         label='Numba', color=colors['numba'], alpha=0.9)
    cuda_bars = ax1.bar(x + width/2, results['cuda'], width, 
                        label='CUDA', color=colors['cuda'], alpha=0.9)
    
    # Add value labels
    for bars in [numba_bars, cuda_bars]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}s',
                        ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Optimization Time: CUDA vs Numba (20 iterations)\n(Lower is Better)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(node_counts)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 2: Speedup Chart
    speedups = []
    for i in range(len(node_counts)):
        if results['cuda'][i] > 0 and results['numba'][i] > 0:
            speedup = results['numba'][i] / results['cuda'][i]
            speedups.append(speedup)
        else:
            speedups.append(0)
    
    bars = ax2.bar(x, speedups, color='#2E86AB', alpha=0.9)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add reference line at 1x
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No speedup (1x)')
    
    ax2.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax2.set_title('CUDA Speedup over Numba\n(Higher is Better)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(node_counts)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'benchmark_cuda_vs_numba_optimization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Nodes':<10} {'Numba (s)':<15} {'CUDA (s)':<15} {'Speedup':<10}")
    print("-"*80)
    for i in range(len(node_counts)):
        numba_time = results['numba'][i]
        cuda_time = results['cuda'][i]
        speedup = speedups[i]
        print(f"{node_counts[i]:<10} {numba_time:<15.4f} {cuda_time:<15.4f} {speedup:<10.2f}x")
    print("="*80)
    
    plt.show()


def main():
    """Main function"""
    node_counts, instance_names, results = run_optimization_benchmark()
    plot_optimization_results(node_counts, instance_names, results)


if __name__ == '__main__':
    main()
