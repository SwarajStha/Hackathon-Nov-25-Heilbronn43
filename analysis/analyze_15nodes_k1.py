#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析 15-nodes-cu-k1.json 的交叉情况
"""
import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, 'build_artifacts')

import planar_cuda


def segments_intersect_python(p1, p2, p3, p4):
    """Python 版本的线段相交判断（用于验证）"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # 检查是否共享端点
    if (p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4):
        return False
    
    # 叉积计算
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    # 严格相交: d1*d2 < 0 AND d3*d4 < 0
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def main():
    file_path = r'results\06-18-34\15-nodes-cu-k1.json'
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    print("="*80)
    print(f"分析文件: {file_path}")
    print("="*80)
    print(f"节点数: {len(nodes_x)}, 边数: {len(edges)}")
    print(f"画布大小: {width} x {height}\n")
    
    # 显示所有节点
    print("节点坐标:")
    for i, (x, y) in enumerate(zip(nodes_x, nodes_y)):
        print(f"  节点 {i:2d}: ({x:3d}, {y:3d})")
    
    # 统计共线节点
    from collections import defaultdict
    x_groups = defaultdict(list)
    y_groups = defaultdict(list)
    
    for i, (x, y) in enumerate(zip(nodes_x, nodes_y)):
        x_groups[x].append((i, y))
        y_groups[y].append((i, x))
    
    print(f"\n共线分析 (相同 x 坐标):")
    for x, nodes in sorted(x_groups.items()):
        if len(nodes) > 1:
            print(f"  x={x}: {len(nodes)} 个节点")
            for node_id, y in sorted(nodes, key=lambda n: n[1]):
                print(f"    节点 {node_id}: y={y}")
    
    # 用 CUDA 计算
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges, 
                                      cell_size=100, width=width, height=height)
    
    total = solver.calculate_total_crossings()
    k = solver.calculate_k_value()
    crossings = solver.get_edge_crossings()
    
    print(f"\nCUDA 结果:")
    print(f"  总交叉数: {total}")
    print(f"  K 值: {k}")
    
    # 显示有交叉的边
    print(f"\n交叉详情:")
    for i, count in enumerate(crossings):
        if count > 0:
            src, tgt = edges[i]
            p1 = (nodes_x[src], nodes_y[src])
            p2 = (nodes_x[tgt], nodes_y[tgt])
            print(f"  边 {i:2d}: {edges[i]} = {p1} → {p2}, 交叉数 = {count}")
    
    # 手工验证：找出所有实际相交的边对
    print(f"\n手工验证 (Python):")
    crossing_pairs = []
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            src1, tgt1 = edges[i]
            src2, tgt2 = edges[j]
            
            p1 = (nodes_x[src1], nodes_y[src1])
            p2 = (nodes_x[tgt1], nodes_y[tgt1])
            p3 = (nodes_x[src2], nodes_y[src2])
            p4 = (nodes_x[tgt2], nodes_y[tgt2])
            
            if segments_intersect_python(p1, p2, p3, p4):
                crossing_pairs.append((i, j))
                print(f"  边 {i} × 边 {j}:")
                print(f"    边{i}: {p1} → {p2}")
                print(f"    边{j}: {p3} → {p4}")
    
    print(f"\nPython 手工计算:")
    print(f"  相交边对数: {len(crossing_pairs)}")
    print(f"  总交叉数: {len(crossing_pairs) * 2}  (每对算两次)")
    
    # 计算每条边的交叉数
    python_crossings = [0] * len(edges)
    for i, j in crossing_pairs:
        python_crossings[i] += 1
        python_crossings[j] += 1
    
    python_k = max(python_crossings) if python_crossings else 0
    print(f"  K 值 (最大边交叉): {python_k}")
    
    # 比较结果
    print(f"\n结果比较:")
    print(f"  CUDA 总交叉: {total}, Python 总交叉: {len(crossing_pairs) * 2}")
    print(f"  CUDA K值: {k}, Python K值: {python_k}")
    
    if total == len(crossing_pairs) * 2 and k == python_k:
        print(f"  ✅ CUDA 和 Python 结果一致!")
    else:
        print(f"  ❌ CUDA 和 Python 结果不一致!")


if __name__ == "__main__":
    main()
