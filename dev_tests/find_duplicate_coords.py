"""查找所有重复坐标的节点"""
import json
import os
from collections import defaultdict

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = [
    os.path.join(base_dir, 'results/11-29-04/15-nodes-cu-k4.json'),
    os.path.join(base_dir, 'results/11-29-04/70-nodes-cu-k24.json'),
    os.path.join(base_dir, 'results/11-29-04/100-nodes-cu-k28.json'),
]

for filepath in files:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    coord_map = defaultdict(list)
    for node in data['nodes']:
        coord = (node['x'], node['y'])
        coord_map[coord].append(node['id'])
    
    duplicates = {coord: ids for coord, ids in coord_map.items() if len(ids) > 1}
    
    print(f"\n{'='*70}")
    print(f"File: {os.path.basename(filepath)}")
    print(f"{'='*70}")
    
    if duplicates:
        print(f"VIOLATION: {len(duplicates)} coordinates with multiple nodes!")
        for coord, node_ids in sorted(duplicates.items()):
            print(f"  Coordinate {coord}: nodes {node_ids}")
            
        # 找出涉及这些节点的边
        all_dup_nodes = set()
        for ids in duplicates.values():
            all_dup_nodes.update(ids)
        
        affected_edges = []
        for edge in data['edges']:
            if edge['source'] in all_dup_nodes or edge['target'] in all_dup_nodes:
                affected_edges.append(edge)
        
        print(f"\n  Total {len(affected_edges)} edges involve duplicate-coord nodes")
    else:
        print("OK: All nodes have unique coordinates")
