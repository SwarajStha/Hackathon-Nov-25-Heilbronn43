#!/usr/bin/env python3
"""Debug crossing calculation - check if we're counting correctly."""

import json
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def point_on_segment(p, a, b, epsilon=1e-10):
    """Check if point p is on segment ab."""
    # Check if p is collinear with a and b
    cross = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1])
    if abs(cross) > epsilon:
        return False
    
    # Check if p is between a and b
    if p[0] != b[0]:
        return (a[0] <= p[0] <= b[0]) or (b[0] <= p[0] <= a[0])
    else:
        return (a[1] <= p[1] <= b[1]) or (b[1] <= p[1] <= a[1])

def ccw(A, B, C):
    """Counter-clockwise test."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect_proper(A, B, C, D):
    """
    Check if segments AB and CD have a PROPER intersection.
    Proper means they cross in their interiors (not at endpoints).
    """
    # Check if they're on opposite sides
    if ccw(A, C, D) == ccw(B, C, D):
        return False
    if ccw(A, B, C) == ccw(A, B, D):
        return False
    return True

def line_intersection_point(A, B, C, D):
    """Find intersection point of line segments AB and CD."""
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel or coincident
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 < t < 1 and 0 < u < 1:  # Strict inequalities - not at endpoints
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (px, py)
    return None

def count_crossings_strict(data):
    """
    Count crossings strictly according to problem definition:
    1. Only count PROPER crossings (not at endpoints or vertices)
    2. If 3+ edges cross at same point, count pairwise
    3. K = max crossings per edge
    """
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    # First pass: find all intersection points
    intersection_points = defaultdict(list)  # point -> list of edge indices
    
    for i, (s1, t1) in enumerate(edges):
        if s1 == t1:
            continue
        A = nodes[s1]
        B = nodes[t1]
        
        for j, (s2, t2) in enumerate(edges):
            if i >= j:
                continue
            if s2 == t2:
                continue
            
            # Skip if edges share an endpoint
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            
            C = nodes[s2]
            D = nodes[t2]
            
            # Find intersection point
            pt = line_intersection_point(A, B, C, D)
            if pt:
                # Round to avoid floating point issues
                pt_rounded = (round(pt[0], 6), round(pt[1], 6))
                intersection_points[pt_rounded].append((i, j))
    
    # Second pass: count crossings per edge (pairwise)
    edge_crossings = defaultdict(int)
    total_crossings = 0
    
    print(f"\nFound {len(intersection_points)} unique intersection points\n")
    
    for idx, (pt, edge_pairs) in enumerate(intersection_points.items(), 1):
        print(f"Intersection point #{idx} at ({pt[0]:.2f}, {pt[1]:.2f}):")
        
        # Get all unique edges at this point
        edges_at_point = set()
        for i, j in edge_pairs:
            edges_at_point.add(i)
            edges_at_point.add(j)
        
        print(f"  {len(edges_at_point)} edges meet here")
        
        # Count all pairwise crossings
        edges_list = sorted(edges_at_point)
        pairs_at_point = []
        for i in range(len(edges_list)):
            for j in range(i + 1, len(edges_list)):
                ei, ej = edges_list[i], edges_list[j]
                s1, t1 = edges[ei]
                s2, t2 = edges[ej]
                
                # Verify they don't share endpoints
                if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                    continue
                
                pairs_at_point.append((ei, ej))
                edge_crossings[ei] += 1
                edge_crossings[ej] += 1
                total_crossings += 1
                print(f"    Edge{ei} ({s1}->{t1}) × Edge{ej} ({s2}->{t2})")
        
        print(f"  → {len(pairs_at_point)} pairwise crossings at this point\n")
    
    k_value = max(edge_crossings.values()) if edge_crossings else 0
    
    # Distribution
    crossing_dist = defaultdict(int)
    for edge_idx in range(len(edges)):
        count = edge_crossings[edge_idx]
        crossing_dist[count] += 1
    
    print(f"\n{'='*60}")
    print(f"Crossing distribution:")
    for count in sorted(crossing_dist.keys()):
        print(f"  {crossing_dist[count]} edges with {count} crossing(s)")
    
    print(f"\nK-value (max crossings per edge): {k_value}")
    print(f"Total crossings (pairwise): {total_crossings}")
    print(f"{'='*60}\n")
    
    return k_value, total_crossings

if __name__ == "__main__":
    print("="*60)
    print("Testing: 15-node-others.json (Expected: K=10)")
    print("="*60)
    data1 = load_json("results/29-01-22/15-node-others.json")
    k1, total1 = count_crossings_strict(data1)
    
    print("\n" + "="*60)
    print("Testing: 15-nodes-ours.json (Expected: K=16)")
    print("="*60)
    data2 = load_json("results/29-01-22/15-nodes-ours.json")
    k2, total2 = count_crossings_strict(data2)
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"15-node-others.json: K={k1}, Total={total1} (Expected K=10)")
    print(f"15-nodes-ours.json:  K={k2}, Total={total2} (Expected K=16)")
    print("="*60)
