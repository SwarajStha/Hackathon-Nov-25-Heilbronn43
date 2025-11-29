#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 K-value cost function vs Total Crossings cost function
比较两种优化目标的效果
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

import planar_cuda


def test_cost_functions(instance_path, iterations=20000):
    """测试两种 cost function 的效果"""
    instance_name = Path(instance_path).name
    
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    print(f"\n{'='*80}")
    print(f"测试实例: {instance_name}")
    print(f"节点数: {len(nodes_x)}, 边数: {len(edges)}")
    print(f"{'='*80}")
    
    # ========== 测试 1: Total Crossings Cost Function ==========
    print(f"\n{'─'*80}")
    print("方法 1: Total Crossings Cost Function (当前默认)")
    print(f"{'─'*80}")
    
    solver1 = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                       cell_size=100, width=width, height=height)
    
    start = time.time()
    stats1 = solver1.run_sa_optimization(iterations, 100.0, 0.95, "total_crossings")
    elapsed1 = time.time() - start
    
    print(f"初始状态:")
    print(f"  总交叉数: {int(stats1['initial_crossings'])}")
    print(f"  K值: {int(stats1['initial_k'])}")
    
    print(f"\n最终状态:")
    print(f"  总交叉数: {int(stats1['final_crossings'])} "
          f"({int(stats1['final_crossings'] - stats1['initial_crossings']):+d})")
    print(f"  K值: {int(stats1['final_k'])} "
          f"({int(stats1['final_k'] - stats1['initial_k']):+d})")
    
    print(f"\n统计:")
    print(f"  迭代次数: {int(stats1['iterations'])}")
    print(f"  接受移动: {int(stats1['accepted_moves'])}")
    print(f"  拒绝移动: {int(stats1['rejected_moves'])}")
    print(f"  运行时间: {elapsed1:.2f}s")
    
    # ========== 测试 2: K-Value Cost Function ==========
    print(f"\n{'─'*80}")
    print("方法 2: K-Value Cost Function (新实现)")
    print(f"{'─'*80}")
    
    solver2 = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                       cell_size=100, width=width, height=height)
    
    start = time.time()
    stats2 = solver2.run_sa_optimization(iterations, 100.0, 0.95, "k_value")
    elapsed2 = time.time() - start
    
    print(f"初始状态:")
    print(f"  总交叉数: {int(stats2['initial_crossings'])}")
    print(f"  K值: {int(stats2['initial_k'])}")
    
    print(f"\n最终状态:")
    print(f"  总交叉数: {int(stats2['final_crossings'])} "
          f"({int(stats2['final_crossings'] - stats2['initial_crossings']):+d})")
    print(f"  K值: {int(stats2['final_k'])} "
          f"({int(stats2['final_k'] - stats2['initial_k']):+d})")
    
    print(f"\n统计:")
    print(f"  迭代次数: {int(stats2['iterations'])}")
    print(f"  接受移动: {int(stats2['accepted_moves'])}")
    print(f"  拒绝移动: {int(stats2['rejected_moves'])}")
    print(f"  运行时间: {elapsed2:.2f}s")
    
    # ========== 比较结果 ==========
    print(f"\n{'='*80}")
    print("结果比较")
    print(f"{'='*80}")
    
    print(f"\n{'指标':<25} {'Total Crossings':<20} {'K-Value':<20} {'差异':<15}")
    print(f"{'-'*80}")
    
    k1_final = int(stats1['final_k'])
    k2_final = int(stats2['final_k'])
    k_diff = k2_final - k1_final
    k_winner = "✅ K-Value" if k2_final < k1_final else ("✅ Total Crossings" if k1_final < k2_final else "平局")
    
    total1_final = int(stats1['final_crossings'])
    total2_final = int(stats2['final_crossings'])
    total_diff = total2_final - total1_final
    total_winner = "✅ Total Crossings" if total1_final < total2_final else ("✅ K-Value" if total2_final < total1_final else "平局")
    
    print(f"{'最终 K 值':<25} {k1_final:<20} {k2_final:<20} {k_diff:+d} ({k_winner})")
    print(f"{'最终总交叉数':<25} {total1_final:<20} {total2_final:<20} {total_diff:+d} ({total_winner})")
    print(f"{'运行时间(s)':<25} {elapsed1:<20.2f} {elapsed2:<20.2f} {elapsed2-elapsed1:+.2f}")
    
    print(f"\n{'='*80}")
    print("结论:")
    print(f"{'='*80}")
    
    if k2_final < k1_final:
        improvement = ((k1_final - k2_final) / k1_final * 100)
        print(f"✅ K-Value cost function 效果更好!")
        print(f"   K 值改善: {k1_final} → {k2_final} ({improvement:.1f}% 降低)")
        if k2_final <= 10:
            print(f"   🎯 达到目标: K ≤ 10!")
        else:
            print(f"   ⚠️  未达到目标: K={k2_final} > 10")
    else:
        print(f"⚠️  K-Value cost function 未能改善 K 值")
        print(f"   需要调整参数: 温度、冷却率、迭代次数")
    
    return {
        'instance': instance_name,
        'total_crossings_k': k1_final,
        'k_value_k': k2_final,
        'total_crossings_total': total1_final,
        'k_value_total': total2_final,
        'time1': elapsed1,
        'time2': elapsed2
    }


def main():
    print("="*80)
    print("测试 K-Value Cost Function")
    print("对比 Total Crossings vs K-Value 优化目标")
    print("="*80)
    
    instances = [
        'live-2025-example-instances/70-nodes.json',
        'live-2025-example-instances/100-nodes.json',
    ]
    
    results = []
    
    for instance_path in instances:
        if not Path(instance_path).exists():
            print(f"\n⚠️  文件不存在: {instance_path}")
            continue
        
        result = test_cost_functions(instance_path, iterations=20000)
        results.append(result)
    
    # 总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    
    print(f"\n{'实例':<25} {'方法':<20} {'K值':<10} {'总交叉':<10} {'时间(s)':<10}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['instance']:<25} {'Total Crossings':<20} "
              f"{r['total_crossings_k']:<10} {r['total_crossings_total']:<10} {r['time1']:<10.2f}")
        print(f"{'':<25} {'K-Value':<20} "
              f"{r['k_value_k']:<10} {r['k_value_total']:<10} {r['time2']:<10.2f}")
        print()
    
    print("="*80)
    print("✅ 测试完成!")
    print("="*80)


if __name__ == "__main__":
    main()
