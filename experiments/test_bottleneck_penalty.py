#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试瓶颈惩罚策略 (Bottleneck Penalty Strategy)
比较不同的 cost function：
1. total_crossings (p=1): 最小化总交叉数
2. bottleneck_p2 (p=2): 平方惩罚，对高交叉边加权
3. bottleneck_p3 (p=3): 立方惩罚，强烈偏向减少最大交叉
4. k_value: 直接最小化 max(crossings)
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


def test_cost_functions(instance_path, iterations=10000):
    """比较不同 cost function 的效果"""
    # 加载实例
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    num_nodes = len(nodes_x)
    num_edges = len(edges)
    
    print(f"\n{'='*80}")
    print(f"测试实例: {os.path.basename(instance_path)}")
    print(f"节点数: {num_nodes}, 边数: {num_edges}")
    print(f"{'='*80}\n")
    
    cost_functions = [
        ("total_crossings", "总交叉数 (p=1)"),
        ("bottleneck_p2", "平方惩罚 (p=2)"),
        ("bottleneck_p3", "立方惩罚 (p=3)"),
        ("k_value", "直接最小化K值"),
    ]
    
    results = []
    
    for cost_func, description in cost_functions:
        print(f"\n{'-'*80}")
        print(f"Cost Function: {description}")
        print(f"{'-'*80}")
        
        # 创建新的求解器实例
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                          cell_size=100, width=width, height=height)
        
        # 获取初始状态
        initial_k = solver.calculate_k_value()
        initial_crossings = solver.calculate_total_crossings()
        
        print(f"初始状态: K={initial_k}, 总交叉={initial_crossings}")
        
        # 运行优化
        start_time = time.time()
        stats = solver.run_sa_optimization(
            iterations=iterations,
            start_temp=100.0,
            cooling_rate=0.95,
            cost_function=cost_func
        )
        elapsed = time.time() - start_time
        
        final_k = int(stats['final_k'])
        final_crossings = int(stats['final_crossings'])
        accepted = int(stats['accepted_moves'])
        
        k_improvement = initial_k - final_k
        k_reduction_pct = (k_improvement / initial_k * 100) if initial_k > 0 else 0
        
        print(f"\n结果:")
        print(f"  K值: {initial_k} → {final_k} (改善 {k_improvement}, {k_reduction_pct:.1f}%)")
        print(f"  总交叉: {initial_crossings} → {final_crossings}")
        print(f"  接受移动: {accepted}/{iterations} ({accepted/iterations*100:.1f}%)")
        print(f"  时间: {elapsed:.2f}s")
        
        results.append({
            'cost_function': cost_func,
            'description': description,
            'initial_k': initial_k,
            'final_k': final_k,
            'k_improvement': k_improvement,
            'final_crossings': final_crossings,
            'accepted_moves': accepted,
            'time': elapsed
        })
    
    # 总结比较
    print(f"\n{'='*80}")
    print("总结比较")
    print(f"{'='*80}\n")
    
    print(f"{'Cost Function':<25} {'最终K值':<10} {'K改善':<10} {'总交叉':<10} {'接受率':<10} {'时间(s)':<10}")
    print(f"{'-'*80}")
    
    best_k = min(r['final_k'] for r in results)
    
    for r in results:
        marker = " ⭐" if r['final_k'] == best_k else ""
        accept_rate = r['accepted_moves'] / iterations * 100
        print(f"{r['description']:<25} {r['final_k']:<10} {r['k_improvement']:<10} "
              f"{r['final_crossings']:<10} {accept_rate:<9.1f}% {r['time']:<10.2f}{marker}")
    
    print(f"\n✅ 最佳方法: {[r['description'] for r in results if r['final_k'] == best_k][0]} (K={best_k})")
    
    return results


def main():
    print("="*80)
    print("测试瓶颈惩罚策略 (Bottleneck Penalty)")
    print("比较 p=1 (总交叉), p=2 (平方), p=3 (立方), 和直接K值优化")
    print("="*80)
    
    # 测试 15 节点实例
    test_cost_functions('live-2025-example-instances/15-nodes.json', iterations=5000)
    
    # 测试 70 节点实例
    print("\n\n")
    test_cost_functions('live-2025-example-instances/70-nodes.json', iterations=10000)
    
    print(f"\n{'='*80}")
    print("测试完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
