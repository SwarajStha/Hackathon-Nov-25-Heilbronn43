#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略性能对比测试
测试 legacy, new, numba, cuda 四种策略在不同节点数下的执行时间
"""
import json
import sys
import os
import time
import csv
from pathlib import Path
from datetime import datetime

# 设置 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

# 添加路径
sys.path.insert(0, 'src')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

from LCNv1.core.geometry import Point, GeometryCore
import planar_cuda


def load_instance(filepath):
    """加载实例文件"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    nodes = [Point(n['x'], n['y']) for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    return nodes, edges, data


def benchmark_legacy(nodes, edges):
    """测试 Legacy 策略"""
    geo = GeometryCore(nodes, edges)
    
    start = time.perf_counter()
    k_value = geo.calculate_k_value_legacy()
    elapsed = time.perf_counter() - start
    
    return k_value, elapsed


def benchmark_new(nodes, edges):
    """测试 New 策略"""
    geo = GeometryCore(nodes, edges)
    
    start = time.perf_counter()
    k_value = geo.calculate_k_value_new()
    elapsed = time.perf_counter() - start
    
    return k_value, elapsed


def benchmark_numba(nodes, edges):
    """测试 Numba 策略"""
    geo = GeometryCore(nodes, edges)
    
    start = time.perf_counter()
    k_value = geo.calculate_k_value_numba()
    elapsed = time.perf_counter() - start
    
    return k_value, elapsed


def benchmark_cuda(nodes_x, nodes_y, edges):
    """测试 CUDA 策略"""
    start = time.perf_counter()
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    k_value = solver.calculate_k_value()
    
    elapsed = time.perf_counter() - start
    
    return k_value, elapsed


def run_benchmark(instance_path, num_runs=3):
    """
    对单个实例运行所有策略的benchmark
    
    Args:
        instance_path: 实例文件路径
        num_runs: 每个策略重复运行次数（取平均值）
    
    Returns:
        结果字典
    """
    instance_name = Path(instance_path).stem
    
    # 加载实例
    nodes, edges, data = load_instance(instance_path)
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    # 准备CUDA数据
    nodes_x = [n.x for n in nodes]
    nodes_y = [n.y for n in nodes]
    
    print(f"\n{'='*80}")
    print(f"测试实例: {instance_name}")
    print(f"节点数: {num_nodes}, 边数: {num_edges}")
    print(f"{'='*80}\n")
    
    results = {
        'instance': instance_name,
        'nodes': num_nodes,
        'edges': num_edges
    }
    
    # 测试每个策略
    strategies = [
        ('Legacy', lambda: benchmark_legacy(instance_path)),
        ('New', lambda: benchmark_new(instance_path)),
        ('Numba', lambda: benchmark_numba(instance_path)),
        ('CUDA', lambda: benchmark_cuda(instance_path)),
    ]
    
    for strategy_name, benchmark_func in strategies:
        print(f"\n[{strategy_name}] ", end="", flush=True)
        
        times = []
        k_values = []
        
        try:
            # 预热运行（第一次可能较慢）
            benchmark_func()
            
            # 正式运行
            for run in range(num_runs):
                k_val, elapsed = benchmark_func()
                times.append(elapsed)
                k_values.append(k_val)
                print(".", end="", flush=True)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            k_value = k_values[0]  # 应该都一样
            
            results[f'{strategy_name.lower()}_k'] = k_value
            results[f'{strategy_name.lower()}_time_avg'] = avg_time
            results[f'{strategy_name.lower()}_time_min'] = min_time
            results[f'{strategy_name.lower()}_time_max'] = max_time
            
            print(f" K={k_value}, 平均={avg_time:.4f}s, 最小={min_time:.4f}s, 最大={max_time:.4f}s")
            
        except Exception as e:
            print(f" ❌ 错误: {e}")
            results[f'{strategy_name.lower()}_k'] = None
            results[f'{strategy_name.lower()}_time_avg'] = None
            results[f'{strategy_name.lower()}_time_min'] = None
            results[f'{strategy_name.lower()}_time_max'] = None
    
    return results


def save_to_csv(all_results, output_file):
    """保存结果到CSV文件"""
    if not all_results:
        print("没有结果可保存")
        return
    
    # CSV列名
    fieldnames = [
        'instance', 'nodes', 'edges',
        'legacy_k', 'legacy_time_avg', 'legacy_time_min', 'legacy_time_max',
        'new_k', 'new_time_avg', 'new_time_min', 'new_time_max',
        'numba_k', 'numba_time_avg', 'numba_time_min', 'numba_time_max',
        'cuda_k', 'cuda_time_avg', 'cuda_time_min', 'cuda_time_max'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\n✅ 结果已保存到: {output_file}")


def print_summary(all_results):
    """打印汇总表格"""
    print(f"\n{'='*120}")
    print("性能对比汇总 (平均时间)")
    print(f"{'='*120}")
    print(f"{'实例':<20} {'节点':<8} {'Legacy(s)':<12} {'New(s)':<12} {'Numba(s)':<12} {'CUDA(s)':<12} {'最快':<12}")
    print(f"{'-'*120}")
    
    for r in all_results:
        instance = r['instance']
        nodes = r['nodes']
        
        times = {
            'Legacy': r.get('legacy_time_avg'),
            'New': r.get('new_time_avg'),
            'Numba': r.get('numba_time_avg'),
            'CUDA': r.get('cuda_time_avg')
        }
        
        # 找出最快的策略
        valid_times = {k: v for k, v in times.items() if v is not None}
        if valid_times:
            fastest = min(valid_times, key=valid_times.get)
            fastest_time = valid_times[fastest]
        else:
            fastest = "N/A"
            fastest_time = None
        
        legacy_str = f"{times['Legacy']:.4f}" if times['Legacy'] else "N/A"
        new_str = f"{times['New']:.4f}" if times['New'] else "N/A"
        numba_str = f"{times['Numba']:.4f}" if times['Numba'] else "N/A"
        cuda_str = f"{times['CUDA']:.4f}" if times['CUDA'] else "N/A"
        
        print(f"{instance:<20} {nodes:<8} {legacy_str:<12} {new_str:<12} {numba_str:<12} {cuda_str:<12} {fastest:<12}")
    
    print(f"{'='*120}")


def main():
    print("="*80)
    print("策略性能对比测试")
    print("测试策略: Legacy, New, Numba, CUDA")
    print("="*80)
    
    # 测试实例列表 (15, 70, 100, 150 nodes)
    instances = [
        'live-2025-example-instances/15-nodes.json',
        'live-2025-example-instances/70-nodes.json',
        'live-2025-example-instances/100-nodes.json',
        'live-2025-example-instances/150-nodes.json',
    ]
    
    # 每个策略运行3次取平均
    num_runs = 3
    
    all_results = []
    
    for instance_path in instances:
        if not Path(instance_path).exists():
            print(f"\n⚠️  文件不存在: {instance_path}")
            continue
        
        result = run_benchmark(instance_path, num_runs)
        all_results.append(result)
    
    # 保存CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.csv"
    save_to_csv(all_results, output_file)
    
    # 打印汇总
    print_summary(all_results)
    
    # 计算加速比
    print(f"\n{'='*80}")
    print("加速比分析 (相对于Legacy)")
    print(f"{'='*80}")
    print(f"{'实例':<20} {'节点':<8} {'New':<12} {'Numba':<12} {'CUDA':<12}")
    print(f"{'-'*80}")
    
    for r in all_results:
        instance = r['instance']
        nodes = r['nodes']
        legacy_time = r.get('legacy_time_avg')
        
        if legacy_time:
            new_speedup = legacy_time / r['new_time_avg'] if r.get('new_time_avg') else None
            numba_speedup = legacy_time / r['numba_time_avg'] if r.get('numba_time_avg') else None
            cuda_speedup = legacy_time / r['cuda_time_avg'] if r.get('cuda_time_avg') else None
            
            new_str = f"{new_speedup:.2f}x" if new_speedup else "N/A"
            numba_str = f"{numba_speedup:.2f}x" if numba_speedup else "N/A"
            cuda_str = f"{cuda_speedup:.2f}x" if cuda_speedup else "N/A"
            
            print(f"{instance:<20} {nodes:<8} {new_str:<12} {numba_str:<12} {cuda_str:<12}")
        else:
            print(f"{instance:<20} {nodes:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print(f"{'='*80}")
    print(f"\n✅ 测试完成!")


if __name__ == "__main__":
    main()
