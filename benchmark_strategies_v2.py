#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略性能对比测试 v2
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

from LCNv1.api import LCNSolver
import planar_cuda


def load_instance(json_path):
    """加载实例数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def benchmark_strategy(strategy_name, json_path, num_runs=3, timeout=10.0):
    """
    测试指定策略
    
    Args:
        strategy_name: 'legacy', 'new', 'numba', 'cuda'
        json_path: JSON 文件路径
        num_runs: 重复运行次数
        timeout: 单次运行超时时间（秒）
    
    Returns:
        (k_value, avg_time, min_time, max_time)
    """
    times = []
    k_value = None
    
    if strategy_name == 'cuda':
        # CUDA 使用特殊接口
        data = load_instance(json_path)
        nodes_x = [n['x'] for n in data['nodes']]
        nodes_y = [n['y'] for n in data['nodes']]
        edges = [(e['source'], e['target']) for e in data['edges']]
        
        for _ in range(num_runs):
            start = time.perf_counter()
            solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
            k_value = solver.calculate_k_value()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # 检查是否超时
            if elapsed > timeout:
                print(f"⚠️  超时 (>{timeout}s), ", end='')
                break
    else:
        # Legacy, New, Numba 使用统一 API
        for i in range(num_runs):
            start_total = time.perf_counter()
            
            solver = LCNSolver(strategy=strategy_name)
            solver.load_from_json(json_path)
            
            start = time.perf_counter()
            stats = solver.get_stats()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if k_value is None:
                k_value = stats['k']
            
            # 检查是否超时
            total_elapsed = time.perf_counter() - start_total
            if total_elapsed > timeout:
                print(f"⚠️  超时 (>{timeout}s), ", end='')
                break
    
    if not times:
        raise TimeoutError(f"策略 {strategy_name} 在超时前未完成任何运行")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return k_value, avg_time, min_time, max_time


def run_benchmark(instance_path, strategies=['legacy', 'new', 'numba', 'cuda'], num_runs=3):
    """
    对单个实例运行所有策略的benchmark
    
    Args:
        instance_path: 实例文件路径
        strategies: 要测试的策略列表
        num_runs: 每个策略重复运行次数
    
    Returns:
        结果字典
    """
    instance_name = Path(instance_path).stem
    
    # 加载实例信息
    data = load_instance(instance_path)
    num_nodes = len(data['nodes'])
    num_edges = len(data['edges'])
    
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
    for strategy in strategies:
        strategy_upper = strategy.upper() if strategy != 'cuda' else 'CUDA'
        print(f"[{strategy.capitalize()}] ", end='', flush=True)
        
        try:
            k_value, avg_time, min_time, max_time = benchmark_strategy(
                strategy, instance_path, num_runs
            )
            
            results[f'{strategy}_k'] = k_value
            results[f'{strategy}_avg'] = avg_time
            results[f'{strategy}_min'] = min_time
            results[f'{strategy}_max'] = max_time
            
            print(f"... K={k_value}, 平均={avg_time:.4f}s, 最小={min_time:.4f}s, 最大={max_time:.4f}s")
            
        except Exception as e:
            print(f" ❌ 错误: {e}")
            results[f'{strategy}_k'] = None
            results[f'{strategy}_avg'] = None
            results[f'{strategy}_min'] = None
            results[f'{strategy}_max'] = None
    
    return results


def save_to_csv(all_results, output_path):
    """保存结果到 CSV 文件"""
    strategies = ['legacy', 'new', 'numba', 'cuda']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ['实例', '节点数', '边数']
        for strategy in strategies:
            header.extend([
                f'{strategy.capitalize()}_K',
                f'{strategy.capitalize()}_平均时间(s)',
                f'{strategy.capitalize()}_最小时间(s)',
                f'{strategy.capitalize()}_最大时间(s)'
            ])
        writer.writerow(header)
        
        # 写入数据
        for result in all_results:
            row = [result['instance'], result['nodes'], result['edges']]
            for strategy in strategies:
                row.extend([
                    result.get(f'{strategy}_k', 'N/A'),
                    result.get(f'{strategy}_avg', 'N/A'),
                    result.get(f'{strategy}_min', 'N/A'),
                    result.get(f'{strategy}_max', 'N/A')
                ])
            writer.writerow(row)
    
    print(f"\n✅ 结果已保存到: {output_path}")


def print_summary(all_results):
    """打印性能对比摘要"""
    strategies = ['legacy', 'new', 'numba', 'cuda']
    
    print(f"\n{'='*80}")
    print(f"{'性能对比汇总 (平均时间)':^80}")
    print(f"{'='*80}")
    print(f"{'实例':<20} {'节点':<8} {'Legacy(s)':<12} {'New(s)':<12} {'Numba(s)':<12} {'CUDA(s)':<12} {'最快':<10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        times = {}
        for strategy in strategies:
            avg = result.get(f'{strategy}_avg')
            times[strategy] = avg if avg is not None else float('inf')
        
        fastest = min(times.items(), key=lambda x: x[1])[0] if any(t != float('inf') for t in times.values()) else 'N/A'
        
        print(f"{result['instance']:<20} {result['nodes']:<8} ", end='')
        for strategy in strategies:
            avg = result.get(f'{strategy}_avg')
            if avg is not None:
                print(f"{avg:<12.4f} ", end='')
            else:
                print(f"{'N/A':<12} ", end='')
        print(f"{fastest.upper():<10}")
    
    print(f"{'='*80}\n")
    
    # 加速比分析
    print(f"{'='*80}")
    print("加速比分析 (相对于Legacy)")
    print(f"{'='*80}")
    print(f"{'实例':<20} {'节点':<8} {'New':<12} {'Numba':<12} {'CUDA':<12}")
    print(f"{'-'*80}")
    
    for result in all_results:
        legacy_time = result.get('legacy_avg')
        if legacy_time and legacy_time > 0:
            print(f"{result['instance']:<20} {result['nodes']:<8} ", end='')
            for strategy in ['new', 'numba', 'cuda']:
                avg = result.get(f'{strategy}_avg')
                if avg and avg > 0:
                    speedup = legacy_time / avg
                    print(f"{speedup:<12.2f}x ", end='')
                else:
                    print(f"{'N/A':<12} ", end='')
            print()
        else:
            print(f"{result['instance']:<20} {result['nodes']:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print(f"{'='*80}\n")


def main():
    """主函数"""
    print("="*80)
    print("策略性能对比测试")
    print("测试策略: Legacy, New, Numba, CUDA")
    print("="*80)
    
    # 测试实例列表
    instances = [
        'live-2025-example-instances/15-nodes.json',
        'live-2025-example-instances/70-nodes.json',
        'live-2025-example-instances/100-nodes.json',
        'live-2025-example-instances/150-nodes.json',
    ]
    
    # 运行所有测试
    all_results = []
    for instance_path in instances:
        if not Path(instance_path).exists():
            print(f"\n⚠️  跳过不存在的文件: {instance_path}")
            continue
        
        result = run_benchmark(instance_path, num_runs=3)
        all_results.append(result)
    
    # 保存到 CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'benchmark_results_{timestamp}.csv'
    save_to_csv(all_results, csv_path)
    
    # 打印摘要
    print_summary(all_results)
    
    print("✅ 测试完成!")


if __name__ == '__main__':
    main()
