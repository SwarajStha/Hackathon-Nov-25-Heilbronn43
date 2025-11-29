#!/usr/bin/env python3
"""Verify the K-value of the 15-node-others.json file."""

import json
import sys
from collections import defaultdict

def load_json(filepath):
    """Load graph from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def ccw(A, B, C):
    """Check if three points are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    """Check if line segment AB intersects with CD."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def count_crossings_detailed(data):
    """Count crossings with detailed analysis."""
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    # Count crossings per edge
    edge_crossings = defaultdict(int)
    total_crossings = 0
    crossing_pairs = []
    
    for i, (s1, t1) in enumerate(edges):
        if s1 == t1:
            continue
        A = nodes[s1]
        B = nodes[t1]
        
        for j, (s2, t2) in enumerate(edges):
            if i >= j:  # Only count each pair once
                continue
            if s2 == t2:
                continue
            
            # Skip if edges share an endpoint
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            
            C = nodes[s2]
            D = nodes[t2]
            
            if segments_intersect(A, B, C, D):
                edge_crossings[i] += 1
                edge_crossings[j] += 1
                total_crossings += 1
                crossing_pairs.append((i, j, s1, t1, s2, t2))
    
    # Calculate K-value
    k_value = max(edge_crossings.values()) if edge_crossings else 0
    
    # Show distribution
    crossing_dist = defaultdict(int)
    for edge_idx in range(len(edges)):
        count = edge_crossings[edge_idx]
        crossing_dist[count] += 1
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: 15-node-others.json")
    print(f"{'='*60}")
    print(f"\nK-value (max crossings per edge): {k_value}")
    print(f"Total crossings: {total_crossings}")
    print(f"\nCrossing distribution:")
    for count in sorted(crossing_dist.keys()):
        print(f"  {crossing_dist[count]} edges with {count} crossing(s)")
    
    # Find edges with maximum crossings
    max_crossing_edges = [i for i, count in edge_crossings.items() if count == k_value]
    print(f"\nEdges with {k_value} crossings:")
    for edge_idx in max_crossing_edges:
        s, t = edges[edge_idx]
        print(f"  Edge #{edge_idx}: {s} -> {t}")
    
    print(f"\nAll crossing pairs ({total_crossings} total):")
    for idx, (i, j, s1, t1, s2, t2) in enumerate(crossing_pairs, 1):
        print(f"  {idx}. Edge{i} ({s1}->{t1}) × Edge{j} ({s2}->{t2})")
    
    return k_value, total_crossings

if __name__ == "__main__":
    filepath = "results/29-01-22/15-node-others.json"
    data = load_json(filepath)
    k, total = count_crossings_detailed(data)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: K={k}, Total={total}")
    print(f"{'='*60}\n")
