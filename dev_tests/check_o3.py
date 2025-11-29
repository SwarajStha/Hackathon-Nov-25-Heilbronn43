#!/usr/bin/env python3
"""快速检查 o3 文件的 K 值"""

import json
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def analyze(filepath):
    data = load_json(filepath)
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    edge_crossings = defaultdict(int)
    
    for i, (s1, t1) in enumerate(edges):
        A = nodes[s1]
        B = nodes[t1]
        for j, (s2, t2) in enumerate(edges):
            if i == j:
                continue
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            C = nodes[s2]
            D = nodes[t2]
            if segments_intersect(A, B, C, D):
                edge_crossings[i] += 1
    
    k = max(edge_crossings.values()) if edge_crossings else 0
    
    print(f"文件: {filepath}")
    print(f"K-value: {k}")
    
    max_edges = [i for i, cnt in edge_crossings.items() if cnt == k]
    print(f"\n有{k}个交叉的边:")
    for idx in max_edges:
        s, t = edges[idx]
        print(f"  边 #{idx}: {s} -> {t}")
    
    dist = defaultdict(int)
    for cnt in edge_crossings.values():
        dist[cnt] += 1
    dist[0] = len(edges) - len(edge_crossings)
    
    print(f"\n交叉分布:")
    for cnt in sorted(dist.keys()):
        print(f"  {dist[cnt]} 条边有 {cnt} 个交叉")
    
    return k

print("="*60)
k = analyze("results/29-01-22/15-node-o3.json")
print("="*60)
print(f"\n结果: K = {k}")
print(f"官方评分显示: 8")
print(f"匹配: {'✓' if k == 8 else '✗'}")
