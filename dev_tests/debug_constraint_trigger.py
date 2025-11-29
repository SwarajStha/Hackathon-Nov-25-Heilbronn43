"""
检查为什么约束没有被触发
"""

import json
from src.LCNv1.core.graph import GridState, GraphData, Point

def main():
    # 加载15-nodes问题
    with open('live-2025-example-instances/15-nodes.json', 'r') as f:
        input_data = json.load(f)
    
    edges = [(e['source'], e['target']) for e in input_data['edges']]
    num_nodes = len(input_data['nodes'])
    graph = GraphData(num_nodes, edges)
    
    # 创建初始positions
    positions = {}
    for node in input_data['nodes']:
        positions[node['id']] = Point(node['x'], node['y'])
    
    state = GridState(positions, input_data['width'], input_data['height'], 
                     graph_data=graph, enable_constraints=True)
    
    print("初始状态:")
    print(f"  enable_constraints: {state._enable_constraints}")
    print(f"  graph_data: {state._graph_data is not None}")
    print(f"  num edges in graph_data: {len(state._graph_data.edges) if state._graph_data else 0}")
    
    # 找到违规的情况并尝试复制它
    # Node 1 on edge (6, 8)
    # 让我们尝试移动node 1到edge (6,8)上
    
    node_6_pos = state.get_position(6)
    node_8_pos = state.get_position(8)
    
    print(f"\nEdge (6, 8): {node_6_pos} -> {node_8_pos}")
    
    # 计算一个在这条边上的点（如果有整数点的话）
    # 简单测试：移动node 1到node 6的位置（应该被duplicate检测阻止）
    print(f"\nNode 1 current: {state.get_position(1)}")
    print(f"Attempting to move node 1 to node 6's position: {node_6_pos}")
    
    try:
        state.move_node(1, node_6_pos)
        print("✗ Move succeeded (should have been blocked by duplicate check!)")
    except ValueError as e:
        print(f"✓ Move blocked: {e}")
    
    # 现在测试一个真实的几何违规
    # 创建一个新的state来测试
    state2 = GridState(positions.copy(), input_data['width'], input_data['height'], 
                      graph_data=graph, enable_constraints=True)
    
    # 移动 node 1 到一个会落在某条边上的位置
    # 使用简单的对角线情况
    test_pos = Point(50, 50)
    print(f"\nAttempting to move node 1 to {test_pos}")
    
    # 先检查这个位置是否在任何边上
    from src.LCNv1.core.geometry import GeometryCore
    for edge_idx in range(len(graph.edges)):
        src, tgt = graph.get_edge_endpoints(edge_idx)
        if src == 1 or tgt == 1:
            continue
        src_pos = state2.get_position(src)
        tgt_pos = state2.get_position(tgt)
        if GeometryCore.point_on_segment_interior(test_pos, src_pos, tgt_pos):
            print(f"  Point {test_pos} is on edge {edge_idx} ({src}->{tgt})")
    
    try:
        state2.move_node(1, test_pos)
        print("✓ Move succeeded (no geometric violation detected)")
    except ValueError as e:
        print(f"✗ Move blocked: {e}")

if __name__ == '__main__':
    main()
