#!/usr/bin/env python3
"""
使用官方几何方法计算K值
Grid granularity = 10000
"""
import json
import sys

def ccw(A, B, C):
    """判断三点的方向 (counter-clockwise)"""
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segments_intersect(A, B, C, D):
    """判断线段AB和CD是否真正相交（不包括端点）"""
    # 共享端点不算相交
    if A == C or A == D or B == C or B == D:
        return False
    
    # 标准线段相交判断
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# 加载文件
filepath = sys.argv[1] if len(sys.argv) > 1 else 'results/07-06-39/225-nodes-NEW-k303.json'
print(f"计算文件: {filepath}")
print(f"Grid granularity: 10000")
print()

with open(filepath) as f:
    data = json.load(f)

# 构建节点字典
nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
edges = [(e['source'], e['target']) for e in data['edges']]

print(f"节点数: {len(nodes)}")
print(f"边数: {len(edges)}")
print()
print("计算每条边的交叉数...")

# 计算每条边与其他边的交叉数
edge_crossings = []
total_crossings = 0

for i, (src1, tgt1) in enumerate(edges):
    p1 = nodes[src1]
    p2 = nodes[tgt1]
    
    count = 0
    for j, (src2, tgt2) in enumerate(edges):
        if i == j:
            continue
        
        p3 = nodes[src2]
        p4 = nodes[tgt2]
        
        if segments_intersect(p1, p2, p3, p4):
            count += 1
    
    edge_crossings.append(count)
    total_crossings += count
    
    if (i + 1) % 100 == 0:
        print(f"  进度: {i+1}/{len(edges)}", end='\r')

print()
print()

# K值是每条边的最大交叉数
k_value = max(edge_crossings)
max_edge_idx = edge_crossings.index(k_value)
max_edge = edges[max_edge_idx]

print("="*70)
print("结果:")
print("="*70)
print(f"K值 (单条边最大交叉数): {k_value}")
print(f"最坏边: 边{max_edge_idx} (节点{max_edge[0]} -> 节点{max_edge[1]})")
print(f"该边与 {k_value} 条其他边相交")
print()
print(f"总交叉对数: {total_crossings // 2} (每对交叉计数2次)")
print()

# 显示交叉分布
from collections import Counter
dist = Counter(edge_crossings)

print("交叉数分布 (前15名):")
for count in sorted(dist.keys(), reverse=True)[:15]:
    print(f"  交叉 {count:3d} 次: {dist[count]:3d} 条边")
