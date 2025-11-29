#!/usr/bin/env python3
"""
深入分析一条边的交叉计算差异
"""
import json
import sys
import os

sys.path.insert(0, 'src')
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
import planar_cuda

# 官方几何
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segments_intersect(A, B, C, D):
    if A == C or A == D or B == C or B == D:
        return False
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# 加载文件
with open('results/07-06-39/225-nodes-NEW-k303.json') as f:
    data = json.load(f)

nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
edges = [(e['source'], e['target']) for e in data['edges']]

# CUDA计算
nodes_x = [n['x'] for n in data['nodes']]
nodes_y = [n['y'] for n in data['nodes']]
solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
cuda_crossings = solver.get_edge_crossings()

# 找一条差异大的边
edge_idx = 124  # 官方=1, CUDA=257
src, tgt = edges[edge_idx]

print(f"分析边{edge_idx}: {src} -> {tgt}")
print(f"  起点: {nodes[src]}")
print(f"  终点: {nodes[tgt]}")
print()
print(f"CUDA计算: {cuda_crossings[edge_idx]}次交叉")
print()

# 手动计算官方方法
p1, p2 = nodes[src], nodes[tgt]
crossing_edges = []

for j, (src2, tgt2) in enumerate(edges):
    if j == edge_idx:
        continue
    
    p3, p4 = nodes[src2], nodes[tgt2]
    
    if segments_intersect(p1, p2, p3, p4):
        crossing_edges.append(j)

print(f"官方计算: {len(crossing_edges)}次交叉")
if len(crossing_edges) > 0:
    print(f"  与以下边相交:")
    for j in crossing_edges[:5]:
        src2, tgt2 = edges[j]
        print(f"    边{j}: {src2}->{tgt2} at {nodes[src2]}, {nodes[tgt2]}")

print()
print(f"差异: CUDA多算了 {cuda_crossings[edge_idx] - len(crossing_edges)} 个交叉")

# 检查是否是坐标问题
print()
print("检查坐标一致性:")
print(f"  CUDA nodes_x[{src}] = {nodes_x[src]}, nodes_y[{src}] = {nodes_y[src]}")
print(f"  官方 nodes[{src}] = {nodes[src]}")
print(f"  CUDA nodes_x[{tgt}] = {nodes_x[tgt]}, nodes_y[{tgt}] = {nodes_y[tgt]}")
print(f"  官方 nodes[{tgt}] = {nodes[tgt]}")
