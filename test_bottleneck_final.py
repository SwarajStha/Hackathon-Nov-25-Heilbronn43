#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终测试：使用立方惩罚 (p=3) 优化 70 和 100 节点实例
多次运行，选择 K 值最小的结果
"""
import json
import sys
import os
import time

# 设置 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

# 添加CUDA DLL路径
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def run_optimization(instance_path, iterations=20000, cost_function="bottleneck_p3"):
    """运行一次优化"""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                      cell_size=100, width=width, height=height)
    
    start_time = time.time()
    stats = solver.run_sa_optimization(iterations, 100.0, 0.95, cost_function)
    elapsed = time.time() - start_time
    
    final_nodes_x, final_nodes_y = solver.get_coordinates()
    
    return {
        'k': int(stats['final_k']),
        'crossings': int(stats['final_crossings']),
        'accepted': int(stats['accepted_moves']),
        'time': elapsed,
        'nodes_x': final_nodes_x,
        'nodes_y': final_nodes_y
    }


def test_multiple_runs(instance_path, num_runs=10, iterations=20000):
    """多次运行，选择最佳结果"""
    instance_name = os.path.basename(instance_path)
    
    print(f"\n{'='*80}")
    print(f"测试实例: {instance_name}")
    print(f"策略: 立方惩罚 (p=3)，运行 {num_runs} 次")
    print(f"{'='*80}\n")
    
    best_k = float('inf')
    best_result = None
    results = []
    
    for run in range(num_runs):
        print(f"[Run {run+1}/{num_runs}] ", end="", flush=True)
        
        result = run_optimization(instance_path, iterations, "bottleneck_p3")
        
        results.append({
            'run': run + 1,
            'k': result['k'],
            'crossings': result['crossings'],
            'time': result['time']
        })
        
        print(f"K={result['k']}, 总交叉={result['crossings']}, 时间={result['time']:.2f}s", end="")
        
        if result['k'] < best_k:
            best_k = result['k']
            best_result = result
            best_result['run'] = run + 1
            print(" ⭐ 新最佳!")
        else:
            print()
    
    # 显示统计
    print(f"\n{'-'*80}")
    print(f"{'Run':<8} {'K值':<8} {'总交叉':<12} {'时间(s)':<10}")
    print(f"{'-'*80}")
    for r in results:
        marker = " ⭐" if r['run'] == best_result['run'] else ""
        print(f"{r['run']:<8} {r['k']:<8} {r['crossings']:<12} {r['time']:<10.2f}{marker}")
    
    # K值分布
    k_values = [r['k'] for r in results]
    avg_k = sum(k_values) / len(k_values)
    min_k = min(k_values)
    max_k = max(k_values)
    
    print(f"\n{'='*80}")
    print(f"✅ 最佳结果: Run {best_result['run']}")
    print(f"   K值: {best_result['k']} (目标: ≤10)")
    print(f"   总交叉数: {best_result['crossings']}")
    print(f"   时间: {best_result['time']:.2f}s")
    print(f"\nK值统计:")
    print(f"   最小: {min_k}")
    print(f"   平均: {avg_k:.1f}")
    print(f"   最大: {max_k}")
    print(f"   标准差: {(sum((k - avg_k)**2 for k in k_values) / len(k_values))**0.5:.1f}")
    
    return best_result


def main():
    print("="*80)
    print("最终测试：立方惩罚 (p=3) 策略")
    print("目标: K ≤ 10")
    print("="*80)
    
    # 测试 70 节点
    result_70 = test_multiple_runs('live-2025-example-instances/70-nodes.json', 
                                   num_runs=10, iterations=20000)
    
    # 测试 100 节点
    print("\n\n")
    result_100 = test_multiple_runs('live-2025-example-instances/100-nodes.json', 
                                    num_runs=10, iterations=20000)
    
    # 最终总结
    print(f"\n{'='*80}")
    print("最终总结")
    print(f"{'='*80}\n")
    print(f"{'实例':<20} {'最佳K值':<10} {'目标':<10} {'状态':<10}")
    print(f"{'-'*80}")
    
    for instance, k in [('70-nodes', result_70['k']), ('100-nodes', result_100['k'])]:
        status = "✅ 通过" if k <= 10 else "❌ 未达标"
        print(f"{instance:<20} {k:<10} ≤10       {status}")
    
    print(f"\n{'='*80}")
    print("测试完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
