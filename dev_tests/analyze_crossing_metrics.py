"""
分析各种crossing指标
"""

import json
from src.LCNv1.core.graph import GridState, GraphData, Point
from src.LCNv1.core.geometry import GeometryCore

def analyze_crossings(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    edges = [(e['source'], e['target']) for e in data['edges']]
    num_nodes = len(data['nodes'])
    graph = GraphData(num_nodes, edges)
    
    positions = {}
    for node in data['nodes']:
        positions[node['id']] = Point(node['x'], node['y'])
    
    state = GridState(positions, data['width'], data['height'],
                     graph_data=graph, enable_constraints=False)
    
    print(f"\n{'='*70}")
    print(f"分析文件: {filepath}")
    print(f"{'='*70}")
    
    # 计算每条边的交叉数
    edge_crossings = [0] * graph.num_edges
    crossing_pairs = []
    
    for i in range(graph.num_edges):
        src_i, tgt_i = graph.get_edge_endpoints(i)
        p1 = state.get_position(src_i)
        p2 = state.get_position(tgt_i)
        
        for j in range(i + 1, graph.num_edges):
            src_j, tgt_j = graph.get_edge_endpoints(j)
            
            # 跳过共享端点的边
            if src_i in (src_j, tgt_j) or tgt_i in (src_j, tgt_j):
                continue
            
            q1 = state.get_position(src_j)
            q2 = state.get_position(tgt_j)
            
            # 检查是否相交
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                edge_crossings[i] += 1
                edge_crossings[j] += 1
                crossing_pairs.append((i, j))
    
    # 指标1: K-value（单边最大交叉数）
    k_value = max(edge_crossings) if edge_crossings else 0
    
    # 指标2: Total crossings（总交叉对数）
    total_pairs = len(crossing_pairs)
    
    # 指标3: Total crossing count（总交叉次数，每条边计数）
    total_count = sum(edge_crossings)
    
    print(f"\n📊 Crossing Metrics:")
    print(f"  1. K-value (max crossings per edge): {k_value}")
    print(f"  2. Total crossing pairs: {total_pairs}")
    print(f"  3. Total crossing count (sum): {total_count}")
    
    # 显示交叉最多的前5条边
    print(f"\n🔝 Top 5 edges by crossings:")
    edge_cross_list = [(i, c) for i, c in enumerate(edge_crossings)]
    edge_cross_list.sort(key=lambda x: x[1], reverse=True)
    
    for i, (edge_idx, cross_count) in enumerate(edge_cross_list[:5]):
        src, tgt = graph.get_edge_endpoints(edge_idx)
        print(f"  {i+1}. Edge {edge_idx} ({src}->{tgt}): {cross_count} crossings")
    
    # 显示交叉分布
    print(f"\n📈 Crossing distribution:")
    cross_dist = {}
    for c in edge_crossings:
        cross_dist[c] = cross_dist.get(c, 0) + 1
    
    for k in sorted(cross_dist.keys(), reverse=True):
        print(f"  {k} crossings: {cross_dist[k]} edges")
    
    return k_value, total_pairs, total_count

# 分析我们的结果
print("\n" + "="*70)
print("CROSSING METRICS ANALYSIS")
print("="*70)

k1, p1, c1 = analyze_crossings('results/11-29-02/15-nodes-ours-v2.json')

# 分析官方答案
k2, p2, c2 = analyze_crossings('live-2025-example-instances/sol-15-nodes-5-planar.json')

# 对比
print(f"\n{'='*70}")
print(f"COMPARISON")
print(f"{'='*70}")
print(f"\nOur result (15-nodes-ours-v2.json):")
print(f"  K-value: {k1}")
print(f"  Total pairs: {p1}")
print(f"  Total count: {c1}")
print(f"\nOfficial answer (sol-15-nodes-5-planar.json):")
print(f"  K-value: {k2}")
print(f"  Total pairs: {p2}")
print(f"  Total count: {c2}")

print(f"\n你说上传后显示 crossing=8")
print(f"可能的对应关系:")
if k1 == 8:
    print(f"  ✓ K-value = 8")
elif p1 == 8:
    print(f"  ✓ Total pairs = 8")
elif c1 == 8:
    print(f"  ✓ Total count = 8")
else:
    print(f"  ✗ 没有指标等于8")
    print(f"  需要检查是否有其他计数方式")
