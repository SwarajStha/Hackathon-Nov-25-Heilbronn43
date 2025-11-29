import json
import sys

files = [
    ('15', 'live-2025-example-instances/sol-15-nodes-5-planar.json'),
    ('70', 'live-2025-example-instances/sol-70-nodes-10-planar.json'),
    ('100', 'live-2025-example-instances/sol-100-nodes-10-planar.json'),
]

print("官方解K值目标:")
print("=" * 50)

for name, filepath in files:
    with open(filepath) as f:
        data = json.load(f)
    
    # Calculate crossings per edge
    from collections import defaultdict
    crossings_per_edge = defaultdict(int)
    
    nodes_dict = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects with p3-p4"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        d1 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        d2 = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1)
        d3 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
        d4 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3)
        
        return d1 * d2 < 0 and d3 * d4 < 0
    
    for i, (s1, t1) in enumerate(edges):
        for j, (s2, t2) in enumerate(edges):
            if i >= j:
                continue
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            
            if segments_intersect(nodes_dict[s1], nodes_dict[t1], 
                                 nodes_dict[s2], nodes_dict[t2]):
                crossings_per_edge[i] += 1
                crossings_per_edge[j] += 1
    
    k_value = max(crossings_per_edge.values()) if crossings_per_edge else 0
    total_crossings = sum(crossings_per_edge.values()) // 2
    
    print(f"{name}-nodes: K={k_value}, 总交叉={total_crossings}")

print("=" * 50)
print("\n当前CUDA结果:")
print("=" * 50)
print("15-nodes: K=5, 总交叉=45")
print("70-nodes: K=25, 总交叉=1327")
print("100-nodes: K=32, 总交叉=1166")
