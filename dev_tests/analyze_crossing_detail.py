"""
详细分析crossing，排除违规的影响
"""

import json
from src.LCNv1.core.graph import GridState, GraphData, Point
from src.LCNv1.core.geometry import GeometryCore
from src.LCNv1.core.violation_repair import ViolationRepair

filepath = 'results/11-29-02/15-nodes-ours-v2.json'

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

print(f"{'='*70}")
print(f"详细分析: {filepath}")
print(f"{'='*70}")

# 1. 检测违规
checker = ViolationRepair(graph, state, max_attempts=0)
violations = checker.detect_violations()

print(f"\n1. 违规检测:")
print(f"  Nodes on edges: {len(violations['nodes_on_edges'])}")
for node_id, edge_idx in violations['nodes_on_edges']:
    src, tgt = graph.get_edge_endpoints(edge_idx)
    print(f"    - Node {node_id} on edge {edge_idx} ({src}->{tgt})")

# 2. 找出涉及违规的边
violated_edges = set()
violated_nodes = set()

for node_id, edge_idx in violations['nodes_on_edges']:
    violated_edges.add(edge_idx)
    violated_nodes.add(node_id)

print(f"\n2. 涉及违规的边: {violated_edges}")
print(f"  涉及违规的节点: {violated_nodes}")

# 3. 分别计算：全部交叉 vs. 排除违规边的交叉
all_crossings = []
valid_crossings = []  # 排除涉及违规边的交叉

for i in range(graph.num_edges):
    src_i, tgt_i = graph.get_edge_endpoints(i)
    p1 = state.get_position(src_i)
    p2 = state.get_position(tgt_i)
    
    for j in range(i + 1, graph.num_edges):
        src_j, tgt_j = graph.get_edge_endpoints(j)
        
        # 跳过共享端点
        if src_i in (src_j, tgt_j) or tgt_i in (src_j, tgt_j):
            continue
        
        q1 = state.get_position(src_j)
        q2 = state.get_position(tgt_j)
        
        # 检查是否相交
        if GeometryCore.segments_intersect(p1, p2, q1, q2):
            all_crossings.append((i, j))
            
            # 如果不涉及违规边，加入valid列表
            if i not in violated_edges and j not in violated_edges:
                valid_crossings.append((i, j))

print(f"\n3. Crossing统计:")
print(f"  全部crossings: {len(all_crossings)}")
print(f"  排除违规边后: {len(valid_crossings)}")
print(f"  被排除的: {len(all_crossings) - len(valid_crossings)}")

# 4. 重新计算排除违规边后的K-value
valid_edge_crossings = [0] * graph.num_edges

for i, j in valid_crossings:
    valid_edge_crossings[i] += 1
    valid_edge_crossings[j] += 1

valid_k = max(valid_edge_crossings) if valid_edge_crossings else 0

print(f"\n4. K-value对比:")
print(f"  全部边: K = {max([0] + [sum(1 for x,y in all_crossings if x==i or y==i) for i in range(graph.num_edges)])}")
print(f"  排除违规边: K = {valid_k}")

# 5. 检查是否有特殊计数方式能得到8
print(f"\n5. 寻找 crossing=8 的可能来源:")

# 可能1: 违规边的数量 + 某个指标
print(f"  违规边数量: {len(violated_edges)}")
print(f"  违规边数量 × 2: {len(violated_edges) * 2}")

# 可能2: K-value + 违规数
k_all = max([0] + [sum(1 for x,y in all_crossings if x==i or y==i) for i in range(graph.num_edges)])
print(f"  K-value + 违规数: {k_all} + {len(violations['nodes_on_edges'])} = {k_all + len(violations['nodes_on_edges'])}")

# 可能3: 涉及违规节点的边的交叉数
violated_node_edge_crossings = 0
for i in range(graph.num_edges):
    src, tgt = graph.get_edge_endpoints(i)
    if src in violated_nodes or tgt in violated_nodes:
        # 计算这条边的交叉数
        for i2, j2 in all_crossings:
            if i == i2 or i == j2:
                violated_node_edge_crossings += 1
                break

print(f"  涉及违规节点的边数: {violated_node_edge_crossings}")

# 可能4: 双倍计数某些违规
print(f"  违规数 × 2: {len(violations['nodes_on_edges']) * 2}")

# 可能5: 检查上传文件时的实际情况
print(f"\n6. 你说上传后显示 crossing=8")
print(f"  如果官方系统的算法是:")
print(f"    - 只计算'有效'的交叉（排除违规边）: {len(valid_crossings)}")
print(f"    - 如果K=8实际意味着有8条边至少有1个crossing:")

edges_with_crossings = sum(1 for c in valid_edge_crossings if c > 0)
print(f"      有crossing的边数量: {edges_with_crossings}")
