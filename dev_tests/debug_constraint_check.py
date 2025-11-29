"""
调试约束检查 - 为什么5个节点在边内部没有被检测到？
"""

import json
from src.LCNv1.core.graph import GridState, GraphData, Point
from src.LCNv1.core.violation_repair import ViolationRepair

def main():
    # 加载15-nodes问题
    with open('live-2025-example-instances/15-nodes.json', 'r') as f:
        data = json.load(f)
    
    # 正确构造GraphData
    edges = [(e['source'], e['target']) for e in data['edges']]
    num_nodes = len(data['nodes'])
    graph = GraphData(num_nodes, edges)
    print(f"Loaded graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # 创建一个有违规的状态（来自test输出）
    # Node 1 on edge (8, 12)
    positions = {}
    for node in data['nodes']:
        positions[node['id']] = Point(node['x'], node['y'])
    
    # 测试1：不启用约束
    print("\n=== TEST 1: Constraints DISABLED ===")
    state1 = GridState(positions, data['width'], data['height'], graph_data=graph, enable_constraints=False)
    print(f"Created state with enable_constraints={state1._enable_constraints}")
    
    checker1 = ViolationRepair(graph, state1, max_attempts=0)
    viol1 = checker1.detect_violations()
    print(f"Violations: nodes_on_edges={len(viol1['nodes_on_edges'])}")
    
    # 尝试移动节点1到一个会违规的位置
    node_1_pos = state1.get_position(1)
    node_8_pos = state1.get_position(8)
    node_12_pos = state1.get_position(12)
    
    print(f"\nNode 1 current: {node_1_pos}")
    print(f"Edge (8, 12): {node_8_pos} -> {node_12_pos}")
    
    # 计算边(8,12)的中点
    mid_x = (node_8_pos.x + node_12_pos.x) // 2
    mid_y = (node_8_pos.y + node_12_pos.y) // 2
    mid_point = Point(mid_x, mid_y)
    print(f"Midpoint of edge (8,12): {mid_point}")
    
    try:
        state1.move_node(1, mid_point)
        print("✓ Move succeeded (constraints disabled, expected)")
    except ValueError as e:
        print(f"✗ Move failed: {e}")
    
    # 测试2：启用约束
    print("\n=== TEST 2: Constraints ENABLED ===")
    positions2 = positions.copy()
    state2 = GridState(positions2, data['width'], data['height'], graph_data=graph, enable_constraints=True)
    print(f"Created state with enable_constraints={state2._enable_constraints}")
    print(f"Graph data present: {state2._graph_data is not None}")
    
    try:
        state2.move_node(1, mid_point)
        print("✗ Move succeeded (should have been blocked!)")
    except ValueError as e:
        print(f"✓ Move correctly blocked: {e}")
    
    # 测试3：克隆后的约束状态
    print("\n=== TEST 3: After clone() ===")
    state3 = state2.clone()
    print(f"Cloned state enable_constraints={state3._enable_constraints}")
    print(f"Cloned state graph_data present: {state3._graph_data is not None}")
    
    try:
        state3.move_node(1, Point(mid_x + 1, mid_y + 1))  # 稍微偏移避免重复
        print("✗ Move succeeded on cloned state (should have been blocked!)")
    except ValueError as e:
        print(f"✓ Move correctly blocked on clone: {e}")
    
    # 测试4：检查calculate_delta的行为
    print("\n=== TEST 4: calculate_delta behavior ===")
    from src.LCNv1.core.k_plane_cost import KPlaneCost
    
    cost_func = KPlaneCost(w_k=10000, w_cross=100, w_len=1)
    
    # 测试违规移动
    delta = cost_func.calculate_delta(graph, state2, 1, mid_point)
    print(f"Delta for violating move: {delta}")
    print(f"Is inf? {delta == float('inf')}")
    
    # 测试合法移动
    safe_point = Point(node_1_pos.x + 5, node_1_pos.y + 5)
    delta2 = cost_func.calculate_delta(graph, state2, 1, safe_point)
    print(f"Delta for safe move: {delta2}")
    print(f"Is inf? {delta2 == float('inf')}")

if __name__ == '__main__':
    main()
