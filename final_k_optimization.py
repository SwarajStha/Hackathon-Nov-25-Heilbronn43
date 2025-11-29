#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终 K 值优化 - 使用瓶颈惩罚策略
多次运行，选择最佳结果，自动保存
"""
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# 设置 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

# 添加CUDA DLL路径
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def optimize_instance(instance_path, num_runs=10, iterations=30000, power=3):
    """
    多次优化一个实例，选择 K 值最小的结果
    
    Args:
        instance_path: 实例文件路径
        num_runs: 运行次数
        iterations: 每次SA迭代数
        power: 瓶颈惩罚幂次 (推荐 2 或 3)
    
    Returns:
        最佳结果字典
    """
    instance_name = Path(instance_path).name
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    print(f"\n{'='*80}")
    print(f"优化实例: {instance_name}")
    print(f"{'='*80}")
    print(f"节点数: {len(nodes_x)}, 边数: {len(edges)}")
    print(f"策略: 瓶颈惩罚 (p={power}), 运行 {num_runs} 次, 每次 {iterations} 迭代")
    
    best_k = float('inf')
    best_result = None
    results = []
    
    for run in range(num_runs):
        print(f"\n[Run {run+1}/{num_runs}] ", end="", flush=True)
        
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                          cell_size=100, width=width, height=height)
        
        start = time.time()
        stats = solver.run_sa_optimization(iterations, 100.0, 0.95, f"bottleneck_p{power}")
        elapsed = time.time() - start
        
        k_final = int(stats['final_k'])
        total_final = int(stats['final_crossings'])
        
        results.append({
            'run': run + 1,
            'k': k_final,
            'total': total_final,
            'time': elapsed
        })
        
        print(f"K={k_final}, 总交叉={total_final}, 时间={elapsed:.2f}s", end="")
        
        if k_final < best_k or (k_final == best_k and total_final < best_result['total']):
            best_k = k_final
            final_nodes_x, final_nodes_y = solver.get_coordinates()
            best_result = {
                'k': k_final,
                'total': total_final,
                'nodes_x': final_nodes_x,
                'nodes_y': final_nodes_y,
                'time': elapsed,
                'run': run + 1,
                'stats': stats
            }
            print(" ⭐ 新最佳!")
        else:
            print()
    
    # 显示统计
    print(f"\n{'-'*80}")
    print(f"{'Run':<8} {'K值':<8} {'总交叉数':<12} {'时间(s)':<10}")
    print(f"{'-'*80}")
    for r in results:
        marker = " ⭐" if r['run'] == best_result['run'] else ""
        print(f"{r['run']:<8} {r['k']:<8} {r['total']:<12} {r['time']:<10.2f}{marker}")
    
    print(f"\n{'='*80}")
    print(f"✅ 最佳结果: Run {best_result['run']}")
    print(f"   K 值: {best_result['k']}")
    print(f"   总交叉数: {best_result['total']}")
    print(f"   时间: {best_result['time']:.2f}s")
    
    # 评估是否达到目标
    num_nodes = len(nodes_x)
    target_k = 5 if num_nodes <= 15 else 10
    
    if best_result['k'] <= target_k:
        print(f"   🎯 达到目标: K={best_result['k']} ≤ {target_k} ✅")
    else:
        diff = best_result['k'] - target_k
        print(f"   ⚠️  未达到目标: K={best_result['k']} > {target_k} (差 {diff})")
    
    return best_result


def save_result(instance_path, result, base_dir='results'):
    """
    保存结果到 JSON 文件
    格式: results/HH-MM-SS/number-nodes-cu-kN.json
    
    Args:
        instance_path: 原始实例路径
        result: 优化结果字典
        base_dir: 结果保存基础目录
    
    Returns:
        保存的文件路径
    """
    # 加载原始实例
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    # 更新节点坐标
    for i, node in enumerate(data['nodes']):
        node['x'] = int(result['nodes_x'][i])
        node['y'] = int(result['nodes_y'][i])
    
    # 创建输出目录 - 使用当前时间的 HH-MM-SS 格式
    now = datetime.now()
    time_folder = f"{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
    output_path = Path(base_dir) / time_folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 构建文件名: number-nodes-cu-kN.json
    num_nodes = len(data['nodes'])
    k_value = result['k']
    filename = f"{num_nodes}-nodes-cu-k{k_value}.json"
    output_file = output_path / filename
    
    # 保存
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n[SAVED] {output_file}")
    return output_file


def main():
    print("="*80)
    print("最终 K 值优化 - 瓶颈惩罚策略 (p=3)")
    print("="*80)
    
    # 测试实例
    instances = [
        ('live-2025-example-instances/15-nodes.json', 10, 20000, 3),  # 小实例：15次运行
        ('live-2025-example-instances/225-nodes.json', 20, 30000, 3),  # 中实例：15次运行
        ('live-2025-example-instances/625-nodes.json', 20, 30000, 3), # 大实例：20次运行
    ]
    
    results_dir = r'D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\results'
    
    all_results = []
    
    for instance_path, num_runs, iterations, power in instances:
        if not Path(instance_path).exists():
            print(f"\n⚠️  文件不存在: {instance_path}")
            continue
        
        # 优化
        result = optimize_instance(instance_path, num_runs, iterations, power)
        
        # 保存结果
        output_file = save_result(instance_path, result, results_dir)
        
        all_results.append({
            'instance': Path(instance_path).name,
            'k': result['k'],
            'total': result['total'],
            'time': result['time'],
            'output': str(output_file)
        })
    
    # 最终总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print(f"\n{'实例':<25} {'K值':<8} {'目标':<8} {'状态':<10} {'总交叉':<10} {'时间(s)':<10}")
    print(f"{'-'*80}")
    
    for r in all_results:
        num_nodes = int(r['instance'].split('-')[0])
        target = 5 if num_nodes <= 15 else 10
        status = "✅ 达标" if r['k'] <= target else f"❌ +{r['k'] - target}"
        
        print(f"{r['instance']:<25} {r['k']:<8} ≤{target:<7} {status:<10} {r['total']:<10} {r['time']:<10.2f}")
    
    print(f"\n✅ 所有结果已保存到: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
