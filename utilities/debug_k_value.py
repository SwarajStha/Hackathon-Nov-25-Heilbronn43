#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 compute_delta_k() - 验证计算是否正确
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def debug_delta_k():
    """详细调试 delta_k 计算"""
    print("="*80)
    print("调试 compute_delta_k() 计算")
    print("="*80)
    
    # 简单的 X 型图案
    nodes_x = [0, 10, 0, 10]
    nodes_y = [0, 10, 10, 0]
    edges = [(0, 1), (2, 3)]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    print("\n初始配置: X 型图案")
    print(f"节点: (0,0), (10,10), (0,10), (10,0)")
    print(f"边: (0,1), (2,3)")
    
    k_initial = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    print(f"\n初始 K 值: {k_initial}")
    print(f"每条边交叉数: {crossings}")
    
    # 测试移动节点 0 到不同位置
    test_positions = [
        (0, 5, "减少交叉 - 边(0,1)不再与(2,3)交叉"),
        (5, 5, "中心位置"),
        (0, 0, "原位置 - 不应该有变化"),
    ]
    
    for new_x, new_y, desc in test_positions:
        print(f"\n{'-'*80}")
        print(f"测试移动: 节点 0 从 (0,0) 到 ({new_x},{new_y})")
        print(f"预期: {desc}")
        
        delta_k = solver.compute_delta_k(0, new_x, new_y)
        print(f"compute_delta_k() 返回: {delta_k}")
        
        # 实际应用并验证
        old_k = solver.calculate_k_value()
        solver.update_node_position(0, new_x, new_y)
        new_k = solver.calculate_k_value()
        actual_delta = new_k - old_k
        
        print(f"实际 K 值变化: {old_k} → {new_k} (delta = {actual_delta})")
        print(f"预测准确: {delta_k == actual_delta}")
        
        if delta_k != actual_delta:
            print(f"❌ 错误! 预测={delta_k}, 实际={actual_delta}")
            crossings_after = solver.get_edge_crossings()
            print(f"   每条边交叉数变化: {crossings} → {crossings_after}")
        else:
            print(f"✅ 正确!")
        
        # 重置以便下次测试
        solver.reset_to_initial()


def debug_acceptance_pattern():
    """调试SA接受模式"""
    print("\n" + "="*80)
    print("调试 SA 接受模式")
    print("="*80)
    
    # 加载一个小实例
    import json
    with open('live-2025-example-instances/15-nodes.json', 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                      cell_size=100, width=width, height=height)
    
    print(f"\n15-nodes 实例")
    k_init = solver.calculate_k_value()
    total_init = solver.calculate_total_crossings()
    print(f"初始 K 值: {k_init}")
    print(f"初始总交叉: {total_init}")
    
    # 运行短时间的优化
    print(f"\n运行 1000 次 K-value 优化...")
    stats = solver.run_sa_optimization(1000, 100.0, 0.95, "k_value")
    
    print(f"\n结果:")
    print(f"  初始 K: {int(stats['initial_k'])}")
    print(f"  最终 K: {int(stats['final_k'])} (delta = {int(stats['final_k'] - stats['initial_k'])})")
    print(f"  初始总交叉: {int(stats['initial_crossings'])}")
    print(f"  最终总交叉: {int(stats['final_crossings'])}")
    print(f"  接受移动: {int(stats['accepted_moves'])}")
    print(f"  拒绝移动: {int(stats['rejected_moves'])}")
    
    accept_rate = stats['accepted_moves'] / stats['iterations'] * 100
    print(f"  接受率: {accept_rate:.1f}%")


if __name__ == "__main__":
    debug_delta_k()
    debug_acceptance_pattern()
