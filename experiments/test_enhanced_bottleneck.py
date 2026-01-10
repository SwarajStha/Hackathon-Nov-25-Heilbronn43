#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加强版测试：更多迭代 + 更多运行次数
目标：K ≤ 10
"""
import json
import sys
import os
import time

sys.stdout.reconfigure(encoding='utf-8')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def run_optimization(instance_path, iterations=30000, start_temp=150.0, cooling_rate=0.97):
    """运行一次优化（增强参数）"""
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
    stats = solver.run_sa_optimization(iterations, start_temp, cooling_rate, "bottleneck_p3")
    elapsed = time.time() - start_time
    
    return {
        'k': int(stats['final_k']),
        'crossings': int(stats['final_crossings']),
        'time': elapsed
    }


def test_enhanced(instance_path, num_runs=15, iterations=30000):
    """加强版测试"""
    instance_name = os.path.basename(instance_path)
    
    print(f"\n{'='*80}")
    print(f"测试实例: {instance_name}")
    print(f"策略: 立方惩罚 (p=3)")
    print(f"参数: {num_runs}次运行, {iterations}迭代, temp=150, cooling=0.97")
    print(f"{'='*80}\n")
    
    best_k = float('inf')
    results = []
    
    for run in range(num_runs):
        print(f"[Run {run+1}/{num_runs}] ", end="", flush=True)
        
        result = run_optimization(instance_path, iterations)
        results.append({'run': run + 1, **result})
        
        print(f"K={result['k']}, 交叉={result['crossings']}, {result['time']:.1f}s", end="")
        
        if result['k'] < best_k:
            best_k = result['k']
            print(" ⭐")
        else:
            print()
    
    # 统计
    k_values = [r['k'] for r in results]
    k_below_target = sum(1 for k in k_values if k <= 10)
    
    print(f"\n{'-'*80}")
    for r in sorted(results, key=lambda x: x['k']):
        marker = " ⭐" if r['k'] == best_k else ""
        target_mark = " ✅" if r['k'] <= 10 else ""
        print(f"Run {r['run']:<3}: K={r['k']:<3} 交叉={r['crossings']:<5} {r['time']:.1f}s{marker}{target_mark}")
    
    print(f"\n{'='*80}")
    print(f"最佳K值: {best_k} {'✅ 达标!' if best_k <= 10 else '❌ 未达标'}")
    print(f"达标次数: {k_below_target}/{num_runs} ({k_below_target/num_runs*100:.1f}%)")
    print(f"平均K值: {sum(k_values)/len(k_values):.1f}")
    print(f"K值范围: [{min(k_values)}, {max(k_values)}]")
    
    return best_k


def main():
    print("="*80)
    print("加强版测试：立方惩罚 (p=3)")
    print("更多迭代 (30k) + 更多运行次数 (15次)")
    print("="*80)
    
    # 测试 70 节点
    k_70 = test_enhanced('live-2025-example-instances/70-nodes.json')
    
    # 测试 100 节点
    k_100 = test_enhanced('live-2025-example-instances/100-nodes.json')
    
    # 总结
    print(f"\n{'='*80}")
    print("最终结果")
    print(f"{'='*80}")
    print(f"70-nodes:  K={k_70} {'✅' if k_70 <= 10 else '❌'}")
    print(f"100-nodes: K={k_100} {'✅' if k_100 <= 10 else '❌'}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
