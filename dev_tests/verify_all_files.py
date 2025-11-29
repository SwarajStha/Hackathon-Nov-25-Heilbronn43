#!/usr/bin/env python3
"""Quick check of intersection points for o2 file."""

import json

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_intersection(A, B, C, D):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 < t < 1 and 0 < u < 1:
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (round(px, 6), round(py, 6))
    return None

def count_intersection_points(data):
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    points = set()
    for i, (s1, t1) in enumerate(edges):
        A = nodes[s1]
        B = nodes[t1]
        for j, (s2, t2) in enumerate(edges):
            if i >= j:
                continue
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            C = nodes[s2]
            D = nodes[t2]
            pt = line_intersection(A, B, C, D)
            if pt:
                points.add(pt)
    
    return len(points)

def count_k_value(data):
    """Count K-value (max crossings per edge)."""
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    from collections import defaultdict
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
            
            # Check intersection
            if ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D):
                edge_crossings[i] += 1
    
    return max(edge_crossings.values()) if edge_crossings else 0

# Test all three files
files = [
    ("15-node-others.json", 10),
    ("15-node-o2.json", 11),
    ("15-nodes-ours.json", 16)
]

print("="*70)
print("CHECKING ALL FILES")
print("="*70)

for filename, expected in files:
    filepath = f"results/29-01-22/{filename}"
    data = load_json(filepath)
    
    intersection_points = count_intersection_points(data)
    k_value = count_k_value(data)
    
    match = "✓" if intersection_points == expected else "✗"
    
    print(f"\n{filename}:")
    print(f"  Intersection points: {intersection_points} (Expected: {expected}) {match}")
    print(f"  K-value: {k_value}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("官方评分系统计算的是: 独特交叉点的数量 (Unique intersection points)")
print("不是 K-value (最大每条边的交叉数)")
print("="*70)
