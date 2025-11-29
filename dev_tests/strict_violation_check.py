"""
严格检查所有违规：
1. 重复坐标（共点）
2. 共线边（共享线段）
3. 边穿过非端点的节点
"""
import json
import sys

def check_point_on_segment(px, py, x1, y1, x2, y2):
    """检查点(px,py)是否在线段(x1,y1)-(x2,y2)上（不包括端点）"""
    # 检查是否是端点
    if (px == x1 and py == y1) or (px == x2 and py == y2):
        return False
    
    # 检查是否共线
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross != 0:
        return False
    
    # 检查是否在线段范围内
    if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
        return True
    
    return False

def are_collinear(p1, p2, p3):
    """检查三个点是否共线"""
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return cross == 0

def segments_overlap(p1, p2, p3, p4):
    """检查两条线段是否共线且重叠"""
    if not (are_collinear(p1, p2, p3) and are_collinear(p1, p2, p4)):
        return False
    
    # 投影到主轴
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    
    if dx >= dy:
        seg1 = (min(p1[0], p2[0]), max(p1[0], p2[0]))
        seg2 = (min(p3[0], p4[0]), max(p3[0], p4[0]))
    else:
        seg1 = (min(p1[1], p2[1]), max(p1[1], p2[1]))
        seg2 = (min(p3[1], p4[1]), max(p3[1], p4[1]))
    
    overlap_start = max(seg1[0], seg2[0])
    overlap_end = min(seg1[1], seg2[1])
    
    return overlap_start < overlap_end

def check_file(filepath):
    """详细检查文件的所有违规"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    print(f"\n{'='*70}")
    print(f"文件: {filepath}")
    print(f"{'='*70}\n")
    
    violations = []
    
    # 1. 检查重复坐标
    from collections import defaultdict
    coord_map = defaultdict(list)
    for nid, coord in nodes.items():
        coord_map[coord].append(nid)
    
    duplicates = {coord: ids for coord, ids in coord_map.items() if len(ids) > 1}
    if duplicates:
        violations.append("重复坐标")
        print(f"❌ 违规1: 重复坐标 ({len(duplicates)}个)")
        for coord, node_ids in sorted(duplicates.items()):
            print(f"   坐标 {coord}: 节点 {node_ids}")
    else:
        print(f"✅ 检查1: 无重复坐标")
    
    # 2. 检查边是否穿过非端点节点
    edge_through_node = []
    for i, (s, t) in enumerate(edges):
        x1, y1 = nodes[s]
        x2, y2 = nodes[t]
        
        for nid, (px, py) in nodes.items():
            if nid == s or nid == t:
                continue
            
            if check_point_on_segment(px, py, x1, y1, x2, y2):
                edge_through_node.append({
                    'edge': (s, t),
                    'edge_coords': ((x1, y1), (x2, y2)),
                    'node': nid,
                    'node_coord': (px, py)
                })
    
    if edge_through_node:
        violations.append("边穿过节点")
        print(f"\n❌ 违规2: 边穿过非端点节点 ({len(edge_through_node)}个)")
        for v in edge_through_node[:10]:
            print(f"   边 {v['edge']}: {v['edge_coords'][0]} -> {v['edge_coords'][1]}")
            print(f"   穿过节点 {v['node']} 在 {v['node_coord']}")
    else:
        print(f"✅ 检查2: 无边穿过节点")
    
    # 3. 检查共线重叠边
    collinear_edges = []
    for i, (s1, t1) in enumerate(edges):
        p1 = nodes[s1]
        p2 = nodes[t1]
        
        for j, (s2, t2) in enumerate(edges[i+1:], i+1):
            # 跳过共享端点的边
            if s1 == s2 or s1 == t2 or t1 == s2 or t1 == t2:
                continue
            
            p3 = nodes[s2]
            p4 = nodes[t2]
            
            if segments_overlap(p1, p2, p3, p4):
                collinear_edges.append({
                    'edge1': (s1, t1),
                    'edge2': (s2, t2),
                    'coords1': (p1, p2),
                    'coords2': (p3, p4)
                })
    
    if collinear_edges:
        violations.append("共线重叠")
        print(f"\n❌ 违规3: 共线重叠边 ({len(collinear_edges)}对)")
        for v in collinear_edges[:10]:
            print(f"   边 {v['edge1']}: {v['coords1'][0]} -> {v['coords1'][1]}")
            print(f"   边 {v['edge2']}: {v['coords2'][0]} -> {v['coords2'][1]}")
    else:
        print(f"✅ 检查3: 无共线重叠边")
    
    print(f"\n{'='*70}")
    if violations:
        print(f"❌ 总结: 发现 {len(violations)} 类违规: {', '.join(violations)}")
        return False
    else:
        print(f"✅ 总结: 完全合法，无违规")
        return True

if __name__ == '__main__':
    import os
    
    files = [
        'results/11-29-04/100-nodes-cu-k20.json',
    ]
    
    all_valid = True
    for filepath in files:
        if not os.path.exists(filepath):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, filepath)
        
        is_valid = check_file(filepath)
        if not is_valid:
            all_valid = False
    
    sys.exit(0 if all_valid else 1)
