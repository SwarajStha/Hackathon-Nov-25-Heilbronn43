#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K-Value Calculation Performance Benchmark
Plot comparison of Legacy, New, Numba, and CUDA strategies
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
import planar_cuda


def benchmark_k_calculation():
    """Run K-value calculation benchmark for all strategies"""
    
    instances = [
        ('15-nodes', 'live-2025-example-instances/15-nodes.json'),
        ('70-nodes', 'live-2025-example-instances/70-nodes.json'),
        ('100-nodes', 'live-2025-example-instances/100-nodes.json'),
    ]
    
    strategies = ['legacy', 'new', 'numba', 'cuda']
    results = {s: [] for s in strategies}
    node_counts = []
    
    print("="*80)
    print("K-Value Calculation Performance Benchmark")
    print("="*80)
    
    for name, path in instances:
        if not Path(path).exists():
            print(f"\n⚠️  Skipping missing file: {path}")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        num_nodes = len(data['nodes'])
        node_counts.append(num_nodes)
        
        print(f"\n{name} ({num_nodes} nodes):")
        print("-" * 40)
        
        # Test each strategy
        for strategy in strategies:
            times = []
            
            try:
                if strategy == 'cuda':
                    # CUDA strategy
                    nodes_x = [n['x'] for n in data['nodes']]
                    nodes_y = [n['y'] for n in data['nodes']]
                    edges = [(e['source'], e['target']) for e in data['edges']]
                    
                    for _ in range(3):
                        start = time.perf_counter()
                        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
                        k = solver.calculate_k_value()
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)
                else:
                    # Legacy, New, Numba strategies
                    for _ in range(3):
                        solver = LCNSolver(strategy=strategy)
                        solver.load_from_json(path)
                        
                        start = time.perf_counter()
                        stats = solver.get_stats()
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)
                
                avg_time = sum(times) / len(times)
                results[strategy].append(avg_time * 1000)  # Convert to ms
                print(f"  {strategy.upper():<10}: {avg_time*1000:6.2f} ms")
                
            except Exception as e:
                print(f"  {strategy.upper():<10}: ERROR - {e}")
                results[strategy].append(0)
    
    return node_counts, results


def plot_results(node_counts, results):
    """Plot benchmark results as grouped bar chart"""
    
    # Filter out strategies with all zeros
    active_strategies = {k: v for k, v in results.items() if any(v)}
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(node_counts))
    width = 0.2
    multiplier = 0
    
    colors = {
        'legacy': '#2E86AB',
        'new': '#A23B72',
        'numba': '#F18F01',
        'cuda': '#06A77D'
    }
    
    # Plot bars for each strategy
    for strategy, times in active_strategies.items():
        offset = width * multiplier
        bars = ax.bar(x + offset, times, width, 
                     label=strategy.upper(), 
                     color=colors.get(strategy, '#999999'),
                     alpha=0.9)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        multiplier += 1
    
    # Customize the plot
    ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_title('K-Value Calculation Performance Comparison\n(Lower is Better)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * (len(active_strategies) - 1) / 2)
    ax.set_xticklabels(node_counts)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis to logarithmic scale if values vary greatly
    max_time = max(max(times) for times in active_strategies.values() if times)
    min_time = min(min(t for t in times if t > 0) for times in active_strategies.values() if any(times))
    
    if max_time / min_time > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Time (milliseconds, log scale)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'benchmark_k_calculation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    plt.show()


def main():
    """Main function"""
    node_counts, results = benchmark_k_calculation()
    plot_results(node_counts, results)


if __name__ == '__main__':
    main()
