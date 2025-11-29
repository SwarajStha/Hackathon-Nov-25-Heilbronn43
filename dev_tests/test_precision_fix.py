#!/usr/bin/env python3
"""
验证CUDA精度修复 - 测试225节点文件
修复内容: 避免d1*d2溢出，改用符号比较
"""
import sys
import os
sys.path.insert(0, 'src')

# 添加CUDA DLL路径
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build_artifacts'))

import json
import planar_cuda

def ccw(A, B, C):
    """Python官方几何计算方法"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    """Python官方几何计算方法"""
    if A == C or A == D or B == C or B == D:
        return False
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

print("=" * 80)
print("CUDA精度修复验证")
print("=" * 80)
print()

# 加载数据
filepath = 'results/07-06-39/225-nodes-NEW-k303.json'
with open(filepath, 'r') as f:
    data = json.load(f)

nodes = [(n['x'], n['y']) for n in data['nodes']]
edges = [(e['source'], e['target']) for e in data['edges']]
nodes_x = [n['x'] for n in data['nodes']]
nodes_y = [n['y'] for n in data['nodes']]

print(f"文件: {filepath}")
print(f"节点数: {len(nodes)}")
print(f"边数: {len(edges)}")
print(f"坐标范围: 0 ~ {max(max(nodes_x), max(nodes_y))}")
print()

# Python计算K值
print("[Python几何计算]")
edge_crossings = []
for i, (s1, t1) in enumerate(edges):
    count = 0
    p1, p2 = nodes[s1], nodes[t1]
    for j, (s2, t2) in enumerate(edges):
        if i == j:
            continue
        q1, q2 = nodes[s2], nodes[t2]
        if segments_intersect(p1, p2, q1, q2):
            count += 1
    edge_crossings.append(count)

k_python = max(edge_crossings)
max_edge_idx = edge_crossings.index(k_python)
print(f"  K值: {k_python}")
print(f"  最坏边: 边{max_edge_idx} (节点{edges[max_edge_idx][0]} -> {edges[max_edge_idx][1]})")
print()

# CUDA计算K值
print("[CUDA计算]")
try:
    # 创建solver实例 - 只需要nodes_x, nodes_y, edges, cell_size, width, height
    max_coord = max(max(nodes_x), max(nodes_y))
    solver = planar_cuda.PlanarSolver(
        nodes_x, nodes_y, edges,
        -1,  # cell_size (auto)
        max_coord + 1000,  # width
        max_coord + 1000   # height
    )
    k_cuda = solver.calculate_k_value()
    print(f"  K值: {k_cuda}")
    print()
    
    print("=" * 80)
    print("结果对比")
    print("=" * 80)
    print(f"Python K值: {k_python}")
    print(f"CUDA K值:   {k_cuda}")
    print(f"差异:       {k_cuda - k_python}")
    print()
    
    if k_cuda == k_python:
        print("✅ 精度修复成功！CUDA计算与Python一致")
        print()
        print("修复细节:")
        print("  问题: 6位数坐标(~100万)导致cross product乘积超过long long范围")
        print("  示例: d1*d2 ≈ 1.8×10^21 > 2^63-1 (9.2×10^18)")
        print("  修复: 将 (d1*d2 < 0) 改为 (d1<0 && d2>0) || (d1>0 && d2<0)")
        print("  效果: 避免溢出，保持符号判断的正确性")
    else:
        print(f"❌ 仍有差异: {abs(k_cuda - k_python)}")
        
except Exception as e:
    print(f"❌ CUDA计算失败: {e}")
    import traceback
    traceback.print_exc()
