#!/usr/bin/env python3
"""
验证工具：计算 crossings 并检查 overlaps

模拟官方评分系统的逻辑：
- Crossings: 边的交叉数量
- Overlaps: 节点在边内部、边重叠等违规
"""

import sys
sys.path.insert(0, 'src')

import json
from LCNv1.core.geometry import Point, GeometryCore
from LCNv1.core.graph import GraphData, GridState
from LCNv1.core.violation_repair import ViolationRepair


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_crossings(graph: GraphData, state: GridState) -> int:
    """
    计算边的交叉数量
    
    与官方系统一致：
    - 不包括共享端点的边
    - 不包括端点接触
    - 只计算proper intersections
    """
    crossings = 0
    edges = graph.edges
    
    for i in range(len(edges)):
        src1, tgt1 = edges[i]
        p1 = state.get_position(src1)
        p2 = state.get_position(tgt1)
        
        for j in range(i + 1, len(edges)):
            src2, tgt2 = edges[j]
            
            # 跳过共享端点的边
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = state.get_position(src2)
            q2 = state.get_position(tgt2)
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                crossings += 1
    
    return crossings


def check_upward_drawing(graph: GraphData, state: GridState) -> bool:
    """
    检查是否为upward drawing
    
    Upward: 每条边的target的y坐标 >= source的y坐标
    """
    for src, tgt in graph.edges:
        src_pos = state.get_position(src)
        tgt_pos = state.get_position(tgt)
        
        # 如果有向下的边，不是upward
        if tgt_pos.y < src_pos.y:
            return False
    
    return True


def validate_drawing(filepath: str, name: str):
    """
    验证一个图形绘制
    
    模拟官方系统显示：
    - Crossings数量（绿色=减少，红色=增加）
    - 是否有overlaps（红色高亮）
    - 是否为upward drawing（红色高亮）
    """
    print(f"\n{'='*70}")
    print(f"验证: {name}")
    print(f"文件: {filepath}")
    print(f"{'='*70}")
    
    data = load_json(filepath)
    
    # 构建图
    nodes = {n['id']: Point(n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    graph = GraphData(len(nodes), edges)
    state = GridState(
        nodes,
        width=data.get('width', 1000),
        height=data.get('height', 1000),
        graph_data=graph,
        enable_constraints=False  # 不启用，只检测
    )
    
    # 检测违规
    repairer = ViolationRepair(graph, state)
    violations = repairer.detect_violations()
    
    # 计算crossings
    crossings = count_crossings(graph, state)
    
    # 检查upward
    is_upward = check_upward_drawing(graph, state)
    
    # 总违规数
    total_violations = (len(violations['duplicate_coords']) +
                       len(violations['nodes_on_edges']) +
                       len(violations['overlapping_edges']))
    
    # 输出结果（模拟官方UI）
    print(f"\n📊 Statistics:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    
    # Crossings box
    crossing_color = "🟢" if crossings == 0 else "🔴" if total_violations > 0 else "⚪"
    print(f"\n{crossing_color} Crossings: {crossings}")
    
    if total_violations > 0:
        print(f"  ⚠️ Box highlighted RED (有 overlap)")
    
    # Overlaps详情
    if violations['duplicate_coords']:
        print(f"\n❌ Overlaps detected:")
        print(f"  - Duplicate coordinates: {len(violations['duplicate_coords'])} groups")
        for pos, node_ids in violations['duplicate_coords'][:3]:
            print(f"    位置{pos}: 节点{node_ids}")
    
    if violations['nodes_on_edges']:
        print(f"\n❌ Overlaps detected:")
        print(f"  - Nodes on edge interior: {len(violations['nodes_on_edges'])} cases")
        for node_id, edge_idx in violations['nodes_on_edges'][:3]:
            src, tgt = graph.get_edge_endpoints(edge_idx)
            print(f"    节点{node_id} 在边{edge_idx}({src}->{tgt})内部")
        if len(violations['nodes_on_edges']) > 3:
            print(f"    ... 还有{len(violations['nodes_on_edges'])-3}处")
    
    if violations['overlapping_edges']:
        print(f"\n❌ Overlaps detected:")
        print(f"  - Overlapping edges: {len(violations['overlapping_edges'])} pairs")
        for edge1, edge2 in violations['overlapping_edges'][:3]:
            src1, tgt1 = graph.get_edge_endpoints(edge1)
            src2, tgt2 = graph.get_edge_endpoints(edge2)
            print(f"    边{edge1}({src1}->{tgt1}) × 边{edge2}({src2}->{tgt2})")
    
    if total_violations == 0:
        print(f"\n✅ No overlaps detected")
    
    # Upward drawing
    if not is_upward:
        print(f"\n🔴 NOT upward drawing (有向下的边)")
        print(f"  Box highlighted RED")
    else:
        print(f"\n✅ Upward drawing")
    
    # 最终判定
    print(f"\n{'='*70}")
    if total_violations > 0 or not is_upward:
        print("❌ INVALID: Box would be highlighted RED")
        print(f"   Crossings={crossings} (不可信，因为有违规)")
    else:
        print("✅ VALID: All constraints satisfied")
        print(f"   Crossings={crossings} (可信)")
    
    return {
        'crossings': crossings,
        'violations': total_violations,
        'is_upward': is_upward,
        'valid': total_violations == 0 and is_upward
    }


if __name__ == "__main__":
    # 测试文件
    files = [
        ("live-2025-example-instances/sol-15-nodes-5-planar.json", "官方标准答案"),
        ("results/29-02-18/15-nodes-ours.json", "我们的旧结果（无约束）"),
        ("results/11-29-02/15-nodes-ours-v2.json", "我们的新结果（带修复）"),
    ]
    
    print("="*70)
    print("官方评分系统验证工具")
    print("="*70)
    print("\n根据官方文档:")
    print("  - Crossings: 边的交叉数量")
    print("  - 如果有 overlaps，box显示为RED")
    print("  - 如果不是 upward drawing，box显示为RED")
    print("  - 拖动节点时，crossings减少显示为GREEN，增加显示为RED")
    
    results = {}
    for filepath, name in files:
        try:
            results[name] = validate_drawing(filepath, name)
        except FileNotFoundError:
            print(f"\n⚠️ 文件不存在: {filepath}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
    
    # 比较结果
    print(f"\n{'='*70}")
    print("结果对比")
    print(f"{'='*70}")
    
    for name, result in results.items():
        status = "✅ VALID" if result['valid'] else "❌ INVALID"
        print(f"\n{name}:")
        print(f"  {status}")
        print(f"  Crossings: {result['crossings']}")
        print(f"  Violations: {result['violations']}")
        print(f"  Upward: {'Yes' if result['is_upward'] else 'No'}")
