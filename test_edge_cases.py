#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试共端点和端点触碰情况
验证这些情况是否被正确排除
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def test_shared_endpoints():
    """测试共享端点的情况"""
    print("="*80)
    print("测试 1: 共享端点")
    print("="*80)
    
    # 两条边共享一个端点: (0,0)-(1,0) 和 (1,0)-(2,0)
    nodes_x = [0, 1, 2]
    nodes_y = [0, 0, 0]
    edges = [(0, 1), (1, 2)]  # 共享节点 1
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    
    print(f"边: {edges}")
    print(f"总交叉数: {total} (预期: 0)")
    print(f"K 值: {k} (预期: 0)")
    print(f"每条边交叉: {crossings}")
    
    assert total == 0, f"错误! 共享端点不应该算交叉，但 total={total}"
    assert k == 0, f"错误! 共享端点不应该算交叉，但 K={k}"
    print("✅ 通过!\n")


def test_endpoint_touching():
    """测试端点碰到另一条线段的情况"""
    print("="*80)
    print("测试 2: 端点碰到线段 (T 型)")
    print("="*80)
    
    # T 型: (0,0)-(2,0) 和 (1,0)-(1,1)
    # 节点 1 在线段 (0,0)-(2,0) 上
    nodes_x = [0, 1, 2, 1]
    nodes_y = [0, 0, 0, 1]
    edges = [(0, 2), (1, 3)]  # 边(1,3)的端点1在边(0,2)上
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    
    print(f"边: {edges}")
    print(f"节点: (0,0), (1,0), (2,0), (1,1)")
    print(f"总交叉数: {total} (预期: 0 - 端点触碰不算交叉)")
    print(f"K 值: {k} (预期: 0)")
    print(f"每条边交叉: {crossings}")
    
    if total != 0:
        print(f"⚠️  注意: 端点触碰被算作交叉了!")
    else:
        print("✅ 通过! 端点触碰正确排除\n")


def test_collinear_overlapping():
    """测试共线重叠的情况"""
    print("="*80)
    print("测试 3: 共线重叠")
    print("="*80)
    
    # 共线重叠: (0,0)-(2,0) 和 (1,0)-(3,0)
    nodes_x = [0, 2, 1, 3]
    nodes_y = [0, 0, 0, 0]
    edges = [(0, 1), (2, 3)]  # 重叠区间 [1,2]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    
    print(f"边: {edges}")
    print(f"节点: (0,0), (2,0), (1,0), (3,0)")
    print(f"总交叉数: {total} (预期: 0 - 共线不算交叉)")
    print(f"K 值: {k} (预期: 0)")
    print(f"每条边交叉: {crossings}")
    
    if total != 0:
        print(f"⚠️  注意: 共线重叠被算作交叉了!")
    else:
        print("✅ 通过! 共线重叠正确排除\n")


def test_proper_crossing():
    """测试真正的交叉"""
    print("="*80)
    print("测试 4: 真正的交叉 (X 型)")
    print("="*80)
    
    # X 型: (0,0)-(10,10) 和 (0,10)-(10,0)
    nodes_x = [0, 10, 0, 10]
    nodes_y = [0, 10, 10, 0]
    edges = [(0, 1), (2, 3)]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    
    print(f"边: {edges}")
    print(f"总交叉数: {total} (预期: 1)")
    print(f"K 值: {k} (预期: 1)")
    print(f"每条边交叉: {crossings}")
    
    assert total == 1, f"错误! 应该有 1 个交叉，但 total={total}"
    assert k == 1, f"错误! 应该 K=1，但 K={k}"
    print("✅ 通过!\n")


def test_endpoint_on_segment():
    """测试端点在另一条线段的延长线上（不相交）"""
    print("="*80)
    print("测试 5: 端点在延长线上（不相交）")
    print("="*80)
    
    # 线段1: (0,0)-(1,0)
    # 线段2: (2,0)-(3,0) - 在延长线上但不相交
    nodes_x = [0, 1, 2, 3]
    nodes_y = [0, 0, 0, 0]
    edges = [(0, 1), (2, 3)]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    
    print(f"边: {edges}")
    print(f"总交叉数: {total} (预期: 0)")
    print(f"K 值: {k} (预期: 0)")
    
    assert total == 0, f"错误! 延长线不应该相交，但 total={total}"
    print("✅ 通过!\n")


def main():
    print("\n测试共端点和端点触碰情况\n")
    
    test_shared_endpoints()
    test_endpoint_touching()
    test_collinear_overlapping()
    test_proper_crossing()
    test_endpoint_on_segment()
    
    print("="*80)
    print("✅ 所有测试完成!")
    print("="*80)


if __name__ == "__main__":
    main()
