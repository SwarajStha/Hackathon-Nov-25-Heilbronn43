#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 70 和 100 节点实例的 K 值优化
使用多次运行选择最佳结果的策略
"""
import json
import sys
import os
import time
from pathlib import Path

# 设置 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

# 添加CUDA DLL路径
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')
sys.path.insert(0, 'src')

import planar_cuda
from LCNv1.core.geometry import Point, GeometryCore


def calculate_k_value(nodes_x, nodes_y, edges):
    """计算K值 (每条边的最大交叉数)"""
    nodes = {i: {'x': nodes_x[i], 'y': nodes_y[i]} for i in range(len(nodes_x))}
    
    edge_crossings = {}
    
    for i in range(len(edges)):
        src1, tgt1 = edges[i]
        p1 = Point(nodes[src1]['x'], nodes[src1]['y'])
        p2 = Point(nodes[tgt1]['x'], nodes[tgt1]['y'])
        
        crossings = 0
        for j in range(len(edges)):
            if i == j:
                continue
            
            src2, tgt2 = edges[j]
            
            # 跳过共享端点
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = Point(nodes[src2]['x'], nodes[src2]['y'])
            q2 = Point(nodes[tgt2]['x'], nodes[tgt2]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                crossings += 1
        
        edge_crossings[i] = crossings
    
    k = max(edge_crossings.values()) if edge_crossings else 0
    total = sum(edge_crossings.values()) // 2
    
    return k, total, edge_crossings


def run_optimization(instance_path, iterations=20000, cell_size=100):
    """运行一次CUDA优化"""
    # 加载实例
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    # 创建求解器
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                      cell_size=cell_size,
                                      width=width, height=height)
    
    # 运行SA
    start_time = time.time()
    stats = solver.run_sa_optimization(iterations, 100.0, 0.95)
    elapsed = time.time() - start_time
    
    # 获取结果
    final_nodes_x, final_nodes_y = solver.get_coordinates()
    
    # 计算K值
    k, total, edge_crossings = calculate_k_value(final_nodes_x, final_nodes_y, edges)
    
    return k, total, final_nodes_x, final_nodes_y, elapsed, edge_crossings


def test_instance(instance_path, num_runs=10, iterations=20000):
    """多次测试一个实例，选择K值最小的结果"""
    instance_name = Path(instance_path).name
    print(f"\n{'='*80}")
    print(f"测试实例: {instance_name}")
    print(f"{'='*80}")
    print(f"策略: 运行 {num_runs} 次，每次 {iterations} 迭代")
    
    best_k = float('inf')
    best_result = None
    results = []
    
    for run in range(num_runs):
        print(f"\n[Run {run+1}/{num_runs}] ", end="", flush=True)
        
        k, total, nodes_x, nodes_y, elapsed, edge_crossings = run_optimization(
            instance_path, iterations
        )
        
        results.append({'run': run+1, 'k': k, 'total': total, 'time': elapsed})
        
        print(f"K={k}, 总交叉={total}, 时间={elapsed:.2f}s", end="")
        
        if k < best_k or (k == best_k and total < best_result['total']):
            best_k = k
            best_result = {
                'k': k,
                'total': total,
                'nodes_x': nodes_x,
                'nodes_y': nodes_y,
                'time': elapsed,
                'edge_crossings': edge_crossings,
                'run': run + 1
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
    print(f"   K值: {best_result['k']}")
    print(f"   总交叉数: {best_result['total']}")
    print(f"   时间: {best_result['time']:.2f}s")
    
    # 显示交叉最多的边
    sorted_edges = sorted(best_result['edge_crossings'].items(), 
                         key=lambda x: x[1], reverse=True)[:5]
    print(f"\n   交叉数最多的5条边:")
    for idx, count in sorted_edges:
        print(f"     边 {idx}: {count} 次交叉")
    
    return best_result


def main():
    print("="*80)
    print("测试 70 和 100 节点实例的 K 值优化")
    print("="*80)
    
    instances = [
        'live-2025-example-instances/70-nodes.json',
        'live-2025-example-instances/100-nodes.json',
    ]
    
    all_results = []
    
    for instance_path in instances:
        if not Path(instance_path).exists():
            print(f"\n⚠️  文件不存在: {instance_path}")
            continue
        
        result = test_instance(instance_path, num_runs=10, iterations=20000)
        
        all_results.append({
            'instance': Path(instance_path).name,
            'k': result['k'],
            'total': result['total'],
            'time': result['time']
        })
    
    # 最终总结
    print(f"\n{'='*80}")
    print("测试总结")
    print(f"{'='*80}")
    print(f"\n{'实例':<25} {'最佳K值':<10} {'总交叉':<10} {'时间(s)':<10}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['instance']:<25} {r['k']:<10} {r['total']:<10} {r['time']:<10.2f}")
    
    print(f"\n✅ 测试完成!")


if __name__ == "__main__":
    main()
