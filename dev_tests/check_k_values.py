#!/usr/bin/env python3
"""Detailed K-value analysis."""

import json
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def analyze_k_value(filepath):
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
    
    k_value = max(edge_crossings.values()) if edge_crossings else 0
    
    print(f"\nFile: {filepath}")
    print(f"K-value: {k_value}")
    
    # Show edges with max crossings
    max_edges = [i for i, cnt in edge_crossings.items() if cnt == k_value]
    print(f"\nEdges with {k_value} crossings:")
    for edge_idx in max_edges:
        s, t = edges[edge_idx]
        print(f"  Edge #{edge_idx}: {s} -> {t}")
    
    # Distribution
    dist = defaultdict(int)
    for cnt in edge_crossings.values():
        dist[cnt] += 1
    
    print(f"\nCrossing distribution:")
    for cnt in sorted(dist.keys()):
        print(f"  {dist[cnt]} edges with {cnt} crossing(s)")
    
    return k_value

print("="*70)
print("K-VALUE ANALYSIS (Maximum crossings per edge)")
print("="*70)

k1 = analyze_k_value("results/29-01-22/15-node-others.json")
k2 = analyze_k_value("results/29-01-22/15-node-o2.json")
k3 = analyze_k_value("results/29-01-22/15-nodes-ours.json")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"15-node-others.json: K = {k1} (Expected: 10)")
print(f"15-node-o2.json:     K = {k2} (Expected: 11) {'✓ MATCH' if k2 == 11 else ''}")
print(f"15-nodes-ours.json:  K = {k3} (Expected: 16)")
print("="*70)
