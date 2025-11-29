#!/usr/bin/env python3
"""Alternative crossing calculation methods."""

import json
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def count_method_1_total_pairs(data):
    """Total crossing pairs."""
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    total = 0
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
            if segments_intersect(A, B, C, D):
                total += 1
    return total

def count_method_2_max_per_edge(data):
    """Maximum crossings on any single edge (K-value)."""
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
    
    return max(edge_crossings.values()) if edge_crossings else 0

def count_method_3_sum_all(data):
    """Sum of all edge crossings (NOT divided by 2)."""
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    total = 0
    for i, (s1, t1) in enumerate(edges):
        A = nodes[s1]
        B = nodes[t1]
        count = 0
        for j, (s2, t2) in enumerate(edges):
            if i == j:
                continue
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            C = nodes[s2]
            D = nodes[t2]
            if segments_intersect(A, B, C, D):
                count += 1
        total += count
    return total

def count_method_4_intersection_points(data):
    """Count unique intersection points."""
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
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

def analyze_file(filepath, expected_k):
    """Analyze all counting methods."""
    data = load_json(filepath)
    
    m1 = count_method_1_total_pairs(data)
    m2 = count_method_2_max_per_edge(data)
    m3 = count_method_3_sum_all(data)
    m4 = count_method_4_intersection_points(data)
    
    print(f"\nFile: {filepath}")
    print(f"Expected K-value: {expected_k}")
    print(f"-" * 50)
    print(f"Method 1 (Total pairs):           {m1}")
    print(f"Method 2 (Max per edge - K):      {m2}")
    print(f"Method 3 (Sum all crossings):     {m3}")
    print(f"Method 4 (Intersection points):   {m4}")
    print(f"Method 3 / 2:                     {m3 / 2:.1f}")
    
    if m1 == expected_k:
        print(f"✓ Method 1 MATCHES expected!")
    if m2 == expected_k:
        print(f"✓ Method 2 MATCHES expected!")
    if m3 == expected_k:
        print(f"✓ Method 3 MATCHES expected!")
    if m4 == expected_k:
        print(f"✓ Method 4 MATCHES expected!")
    if m3 / 2 == expected_k:
        print(f"✓ Method 3/2 MATCHES expected!")

if __name__ == "__main__":
    print("="*60)
    print("TESTING DIFFERENT CROSSING CALCULATION METHODS")
    print("="*60)
    
    analyze_file("results/29-01-22/15-node-others.json", expected_k=10)
    analyze_file("results/29-01-22/15-nodes-ours.json", expected_k=16)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Looking for which method matches the expected values...")
