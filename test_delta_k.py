#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 compute_delta_k() 功能
验证增量 K 值计算是否正确
"""
import sys
import os

# 设置 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

# 添加CUDA DLL路径
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def test_delta_k_simple():
    """测试简单情况：X 型图案"""
    print("="*80)
    print("测试 1: X 型图案 (两条交叉的边)")
    print("="*80)
    
    # X 型图案: (0,0)-(10,10) 交叉 (0,10)-(10,0)
    nodes_x = [0, 10, 0, 10]
    nodes_y = [0, 10, 10, 0]
    edges = [(0, 1), (2, 3)]  # 两条边交叉
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    # 初始 K 值（应该是 1，每条边都交叉 1 次）
    k_before = solver.calculate_k_value()
    print(f"初始 K 值: {k_before}")
    print(f"预期: 1 (每条边交叉 1 次)")
    
    # 测试移动节点 0 到 (0, 5) - 应该减少交叉
    delta_k = solver.compute_delta_k(0, 0, 5)
    print(f"\n移动节点 0 从 (0,0) 到 (0,5)")
    print(f"Delta K: {delta_k}")
    
    # 实际应用移动
    solver.update_node_position(0, 0, 5)
    k_after = solver.calculate_k_value()
    print(f"实际 K 值变化: {k_before} -> {k_after} (delta = {k_after - k_before})")
    print(f"预测正确: {delta_k == (k_after - k_before)}")
    
    assert delta_k == (k_after - k_before), f"Delta K 预测错误! 预测={delta_k}, 实际={k_after - k_before}"
    print("✅ 测试通过!")


def test_delta_k_star():
    """测试星型图案"""
    print("\n" + "="*80)
    print("测试 2: 星型图案 (中心节点连接所有外围节点)")
    print("="*80)
    
    # 星型: 中心节点(5,5)连接到4个角
    nodes_x = [5, 0, 10, 0, 10]  # center, TL, TR, BL, BR
    nodes_y = [5, 0, 0, 10, 10]
    edges = [
        (0, 1),  # center - TL
        (0, 2),  # center - TR
        (0, 3),  # center - BL
        (0, 4),  # center - BR
    ]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    k_before = solver.calculate_k_value()
    print(f"初始 K 值: {k_before}")
    
    edge_crossings = solver.get_edge_crossings()
    print(f"每条边的交叉数: {edge_crossings}")
    
    # 测试移动中心节点
    delta_k = solver.compute_delta_k(0, 6, 6)
    print(f"\n移动中心节点从 (5,5) 到 (6,6)")
    print(f"Delta K: {delta_k}")
    
    solver.update_node_position(0, 6, 6)
    k_after = solver.calculate_k_value()
    print(f"实际 K 值变化: {k_before} -> {k_after} (delta = {k_after - k_before})")
    print(f"预测正确: {delta_k == (k_after - k_before)}")
    
    assert delta_k == (k_after - k_before), f"Delta K 预测错误! 预测={delta_k}, 实际={k_after - k_before}"
    print("✅ 测试通过!")


def test_delta_k_k4():
    """测试 K4 完全图"""
    print("\n" + "="*80)
    print("测试 3: K4 完全图 (4个节点，6条边)")
    print("="*80)
    
    # K4: 正方形排列
    nodes_x = [0, 10, 10, 0]
    nodes_y = [0, 0, 10, 10]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 外围
        (0, 2), (1, 3)  # 对角线（会交叉）
    ]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    k_before = solver.calculate_k_value()
    print(f"初始 K 值: {k_before}")
    
    edge_crossings = solver.get_edge_crossings()
    for i, count in enumerate(edge_crossings):
        print(f"  边 {i} ({edges[i]}): {count} 次交叉")
    
    # 测试移动节点
    delta_k = solver.compute_delta_k(0, 1, 1)
    print(f"\n移动节点 0 从 (0,0) 到 (1,1)")
    print(f"Delta K: {delta_k}")
    
    solver.update_node_position(0, 1, 1)
    k_after = solver.calculate_k_value()
    print(f"实际 K 值变化: {k_before} -> {k_after} (delta = {k_after - k_before})")
    print(f"预测正确: {delta_k == (k_after - k_before)}")
    
    edge_crossings_after = solver.get_edge_crossings()
    print(f"\n移动后每条边的交叉数:")
    for i, count in enumerate(edge_crossings_after):
        print(f"  边 {i} ({edges[i]}): {count} 次交叉 (之前: {edge_crossings[i]})")
    
    assert delta_k == (k_after - k_before), f"Delta K 预测错误! 预测={delta_k}, 实际={k_after - k_before}"
    print("✅ 测试通过!")


def test_performance():
    """测试性能：大规模图"""
    print("\n" + "="*80)
    print("测试 4: 性能测试 (30 节点)")
    print("="*80)
    
    import time
    
    # 生成 30 节点的随机图
    import random
    random.seed(42)
    
    num_nodes = 30
    nodes_x = [random.randint(0, 100) for _ in range(num_nodes)]
    nodes_y = [random.randint(0, 100) for _ in range(num_nodes)]
    
    # 生成一些随机边
    edges = []
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
    for i in range(0, num_nodes - 1, 3):
        if i + 5 < num_nodes:
            edges.append((i, i + 5))
    
    print(f"节点数: {num_nodes}")
    print(f"边数: {len(edges)}")
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    # 测试 compute_delta_k 性能
    start = time.time()
    num_tests = 100
    for _ in range(num_tests):
        node_id = random.randint(0, num_nodes - 1)
        new_x = random.randint(0, 100)
        new_y = random.randint(0, 100)
        delta_k = solver.compute_delta_k(node_id, new_x, new_y)
    
    elapsed = time.time() - start
    avg_time = elapsed / num_tests * 1000
    
    print(f"\n{num_tests} 次 compute_delta_k 调用")
    print(f"总时间: {elapsed:.3f}s")
    print(f"平均时间: {avg_time:.3f} ms/次")
    print("✅ 性能测试完成!")


def main():
    print("测试 compute_delta_k() 功能")
    print("验证增量 K 值计算的正确性\n")
    
    test_delta_k_simple()
    test_delta_k_star()
    test_delta_k_k4()
    test_performance()
    
    print("\n" + "="*80)
    print("✅ 所有测试通过!")
    print("="*80)


if __name__ == "__main__":
    main()
