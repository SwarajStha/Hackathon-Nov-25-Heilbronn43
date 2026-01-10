#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试共线边的特殊情况
验证共线的多条边是否被正确处理
"""
import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def test_vertical_collinear_edges():
    """测试垂直共线的多条边"""
    print("="*80)
    print("测试: 垂直共线的多条边")
    print("="*80)
    
    # 多个点在同一垂直线 x=100 上
    # 边: (0,10)-(0,20), (0,15)-(0,25), (0,30)-(0,40)
    nodes_x = [100, 100, 100, 100, 100, 100]
    nodes_y = [10,  20,  15,  25,  30,  40]
    edges = [
        (0, 1),  # (100,10)-(100,20)
        (2, 3),  # (100,15)-(100,25) - 与第一条重叠
        (4, 5),  # (100,30)-(100,40) - 不重叠
    ]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    
    print(f"节点在垂直线 x=100 上，y 坐标: {nodes_y}")
    print(f"边: {edges}")
    print(f"总交叉数: {total} (预期: 0 - 共线边不应该相交)")
    print(f"K 值: {k} (预期: 0)")
    print(f"每条边交叉: {crossings}")
    
    if total != 0:
        print(f"❌ 错误! 共线边被算作交叉了!")
        return False
    else:
        print("✅ 通过!\n")
        return True


def test_horizontal_collinear_edges():
    """测试水平共线的多条边"""
    print("="*80)
    print("测试: 水平共线的多条边")
    print("="*80)
    
    # 多个点在同一水平线 y=50 上
    nodes_x = [10, 20, 15, 25, 30, 40]
    nodes_y = [50, 50, 50, 50, 50, 50]
    edges = [
        (0, 1),  # (10,50)-(20,50)
        (2, 3),  # (15,50)-(25,50) - 与第一条重叠
        (4, 5),  # (30,50)-(40,50) - 不重叠
    ]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    
    print(f"节点在水平线 y=50 上，x 坐标: {nodes_x}")
    print(f"边: {edges}")
    print(f"总交叉数: {total} (预期: 0 - 共线边不应该相交)")
    print(f"K 值: {k} (预期: 0)")
    print(f"每条边交叉: {crossings}")
    
    if total != 0:
        print(f"❌ 错误! 共线边被算作交叉了!")
        return False
    else:
        print("✅ 通过!\n")
        return True


def test_actual_file():
    """测试实际的 15-nodes-cu-k1.json 文件"""
    print("="*80)
    print("测试: 15-nodes-cu-k1.json 实际文件")
    print("="*80)
    
    file_path = r'results\06-18-34\15-nodes-cu-k1.json'
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return True
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    # 检查有多少节点在同一垂直线上
    from collections import Counter
    x_counts = Counter(nodes_x)
    collinear_x = {x: count for x, count in x_counts.items() if count > 1}
    
    print(f"节点数: {len(nodes_x)}, 边数: {len(edges)}")
    print(f"共线节点 (相同 x 坐标): {collinear_x}")
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                      cell_size=100, width=width, height=height)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    
    print(f"\n结果:")
    print(f"总交叉数: {total}")
    print(f"K 值: {k}")
    
    # 找出交叉数最多的边
    if k > 0:
        max_idx = crossings.index(max(crossings))
        print(f"\n交叉最多的边: 边 {max_idx} ({edges[max_idx]}), 交叉数 = {crossings[max_idx]}")
        
        # 检查这条边是否涉及共线节点
        src, tgt = edges[max_idx]
        if nodes_x[src] == nodes_x[tgt]:
            print(f"  ⚠️  这是一条垂直边! 节点 {src} 和 {tgt} 都在 x={nodes_x[src]}")
            print(f"     y 坐标: ({nodes_y[src]}, {nodes_y[tgt]})")
    
    return k <= 1


def test_diagonal_vs_vertical():
    """测试对角线与垂直线的交叉"""
    print("="*80)
    print("测试: 对角线与垂直边的交叉")
    print("="*80)
    
    # 垂直边: (5,0)-(5,10)
    # 对角线: (0,5)-(10,5) - 应该在 (5,5) 相交
    nodes_x = [5, 5, 0, 10]
    nodes_y = [0, 10, 5, 5]
    edges = [
        (0, 1),  # 垂直边
        (2, 3),  # 水平边
    ]
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    
    print(f"边1 (垂直): (5,0)-(5,10)")
    print(f"边2 (水平): (0,5)-(10,5)")
    print(f"总交叉数: {total} (预期: 1 - 应该在 (5,5) 相交)")
    print(f"K 值: {k} (预期: 1)")
    
    if total == 1 and k == 1:
        print("✅ 通过!\n")
        return True
    else:
        print(f"❌ 错误! 应该有 1 个交叉")
        return False


def main():
    print("\n测试共线边的特殊情况\n")
    
    results = []
    
    results.append(("垂直共线", test_vertical_collinear_edges()))
    results.append(("水平共线", test_horizontal_collinear_edges()))
    results.append(("对角线 vs 垂直", test_diagonal_vs_vertical()))
    results.append(("实际文件", test_actual_file()))
    
    print("="*80)
    print("测试总结")
    print("="*80)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:<20} {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 有测试失败!")


if __name__ == "__main__":
    main()
