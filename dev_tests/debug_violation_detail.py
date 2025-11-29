"""
检查为什么最终验证报告了违规
"""

import json
from src.LCNv1.core.graph import GridState, GraphData, Point
from src.LCNv1.core.violation_repair import ViolationRepair
from src.LCNv1.core.geometry import GeometryCore

def main():
    # 加载15-nodes问题
    with open('live-2025-example-instances/15-nodes.json', 'r') as f:
        input_data = json.load(f)
    
    edges = [(e['source'], e['target']) for e in input_data['edges']]
    num_nodes = len(input_data['nodes'])
    graph = GraphData(num_nodes, edges)
    
    # 加载我们的结果（有违规的那个）
    with open('results/11-29-02/15-nodes-ours-v2.json', 'r') as f:
        result_data = json.load(f)
    
    # 构建positions
    positions = {}
    for node in result_data['nodes']:
        positions[node['id']] = Point(node['x'], node['y'])
    
    state = GridState(positions, result_data['width'], result_data['height'], 
                     graph_data=graph, enable_constraints=False)
    
    # 检测违规
    checker = ViolationRepair(graph, state, max_attempts=0)
    violations = checker.detect_violations()
    
    print(f"检测到的违规:")
    print(f"  nodes_on_edges: {len(violations['nodes_on_edges'])}")
    print(f"  duplicates: {len(violations['duplicate_coords'])}")
    print(f"  overlaps: {len(violations['overlapping_edges'])}")
    
    # 检查第一个违规
    if violations['nodes_on_edges']:
        node_id, edge_idx = violations['nodes_on_edges'][0]
        src, tgt = graph.get_edge_endpoints(edge_idx)
        
        node_pos = state.get_position(node_id)
        src_pos = state.get_position(src)
        tgt_pos = state.get_position(tgt)
        
        print(f"\n第一个违规:")
        print(f"  Node {node_id} at {node_pos}")
        print(f"  Edge {edge_idx} ({src}->{tgt}): {src_pos} -> {tgt_pos}")
        
        # 手动检查
        cross = GeometryCore.cross_product(src_pos, tgt_pos, node_pos)
        print(f"  Cross product: {cross} (0=collinear)")
        
        is_interior = GeometryCore.point_on_segment_interior(node_pos, src_pos, tgt_pos)
        print(f"  On interior? {is_interior}")
        
        # 检查是否是端点
        is_endpoint = (node_id == src or node_id == tgt)
        print(f"  Is endpoint? {is_endpoint}")

if __name__ == '__main__':
    main()
