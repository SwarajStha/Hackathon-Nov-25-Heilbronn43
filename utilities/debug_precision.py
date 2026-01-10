"""
调试精度问题 - 检查Edge 124的交叉计算
CUDA报257个交叉，但实际只有1个
"""
import json

def ccw(A, B, C):
    """Counter-clockwise test using cross product"""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    """Check if segment AB intersects segment CD"""
    # Shared endpoints don't count as crossings
    if A == C or A == D or B == C or B == D:
        return False
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def cross_product_value(ox, oy, ax, ay, bx, by):
    """计算cross product的实际数值 - 模拟CUDA计算"""
    dx1 = ax - ox
    dy1 = ay - oy
    dx2 = bx - ox
    dy2 = by - oy
    result = dx1 * dy2 - dy1 * dx2
    print(f"  CP({ox},{oy})->({ax},{ay})x({bx},{by}): "
          f"dx1={dx1}, dy1={dy1}, dx2={dx2}, dy2={dy2} => {result}")
    return result

# 加载数据
with open('results/07-06-39/225-nodes-NEW-k303.json', 'r') as f:
    data = json.load(f)

nodes = [(n['x'], n['y']) for n in data['nodes']]
edges = [(e['source'], e['target']) for e in data['edges']]

# Edge 124
edge_124 = edges[124]
p1 = nodes[edge_124[0]]
p2 = nodes[edge_124[1]]

print(f"Edge 124: Node {edge_124[0]} {p1} -> Node {edge_124[1]} {p2}")
print(f"坐标数值范围: {max(p1[0], p1[1], p2[0], p2[1])}")
print()

# 计算与所有其他边的交叉
crossings = []
for i, (s, t) in enumerate(edges):
    if i == 124:
        continue
    
    q1 = nodes[s]
    q2 = nodes[t]
    
    if segments_intersect(p1, p2, q1, q2):
        crossings.append(i)

print(f"实际交叉数: {len(crossings)}")
print(f"交叉的边: {crossings}")
print()

# 如果只有1个交叉，详细分析这个交叉的cross product计算
if len(crossings) == 1:
    crossing_edge_idx = crossings[0]
    q1 = nodes[edges[crossing_edge_idx][0]]
    q2 = nodes[edges[crossing_edge_idx][1]]
    
    print(f"唯一交叉: Edge {crossing_edge_idx}")
    print(f"  Edge {crossing_edge_idx}: {q1} -> {q2}")
    print()
    print("Cross Product计算:")
    print(f"  d1 = CP(p1, p2, q1) = CP({p1}, {p2}, {q1})")
    d1 = cross_product_value(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1])
    
    print(f"  d2 = CP(p1, p2, q2) = CP({p1}, {p2}, {q2})")
    d2 = cross_product_value(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1])
    
    print(f"  d3 = CP(q1, q2, p1) = CP({q1}, {q2}, {p1})")
    d3 = cross_product_value(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1])
    
    print(f"  d4 = CP(q1, q2, p2) = CP({q1}, {q2}, {p2})")
    d4 = cross_product_value(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1])
    
    print()
    print(f"d1 * d2 = {d1} * {d2} = {d1 * d2} (< 0? {d1 * d2 < 0})")
    print(f"d3 * d4 = {d3} * {d4} = {d3 * d4} (< 0? {d3 * d4 < 0})")
    print()
    print(f"交叉判定: d1*d2 < 0 AND d3*d4 < 0 = {d1 * d2 < 0 and d3 * d4 < 0}")
    print()
    print(f"最大中间值: {max(abs(d1), abs(d2), abs(d3), abs(d4))}")
    print(f"最大乘积值: {max(abs(d1 * d2), abs(d3 * d4))}")
