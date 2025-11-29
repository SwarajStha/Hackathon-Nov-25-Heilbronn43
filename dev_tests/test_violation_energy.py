"""
测试为什么违规没有被calculate检测到
"""

import json
from src.LCNv1.core.graph import GridState, GraphData, Point
from src.LCNv1.core.k_plane_cost import KPlaneCost
from src.LCNv1.core.violation_repair import ViolationRepair

# 加载最终结果（有7个违规的）
with open('results/11-29-02/15-nodes-ours-v2.json', 'r') as f:
    data = json.load(f)

edges = [(e['source'], e['target']) for e in data['edges']]
num_nodes = len(data['nodes'])
graph = GraphData(num_nodes, edges)

positions = {}
for node in data['nodes']:
    positions[node['id']] = Point(node['x'], node['y'])

state = GridState(positions, data['width'], data['height'],
                 graph_data=graph, enable_constraints=False)

print("State created")
print(f"  Nodes: {graph.num_nodes}")
print(f"  Edges: {graph.num_edges}")

# 1. 检测违规
print("\n1. Violation Detection:")
checker = ViolationRepair(graph, state, max_attempts=0)
violations = checker.detect_violations()
total_v = len(violations['duplicate_coords']) + len(violations['nodes_on_edges']) + len(violations['overlapping_edges'])
print(f"  Total violations: {total_v}")
print(f"    - Duplicates: {len(violations['duplicate_coords'])}")
print(f"    - Nodes on edges: {len(violations['nodes_on_edges'])}")
print(f"    - Overlaps: {len(violations['overlapping_edges'])}")

# 2. 计算能量
print("\n2. Energy Calculation:")
cost_func = KPlaneCost(w_k=10000, w_cross=100, w_len=1)
energy = cost_func.calculate(graph, state)
print(f"  Energy: {energy}")
print(f"  Expected (with violations): ~{1e8 * total_v}")

# 3. 检查第一个违规
if violations['nodes_on_edges']:
    node_id, edge_idx = violations['nodes_on_edges'][0]
    src, tgt = graph.get_edge_endpoints(edge_idx)
    print(f"\n3. First Violation Details:")
    print(f"  Node {node_id} on edge {edge_idx} ({src}->{tgt})")
    print(f"  Node pos: {state.get_position(node_id)}")
    print(f"  Edge: {state.get_position(src)} -> {state.get_position(tgt)}")
    
    from src.LCNv1.core.geometry import GeometryCore
    is_on = GeometryCore.point_on_segment_interior(
        state.get_position(node_id),
        state.get_position(src),
        state.get_position(tgt)
    )
    print(f"  Is on interior? {is_on}")
