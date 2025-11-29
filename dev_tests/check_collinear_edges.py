"""
检查是否有边共享线段（共线且有重叠部分）
即使不完全重叠，只要有任何共用的线段就算违规
"""
import json
import sys

def are_collinear(p1, p2, p3):
    """检查三个点是否共线"""
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return cross == 0

def point_on_segment(p, seg_start, seg_end):
    """检查点p是否在线段上（假设已知共线）"""
    return (min(seg_start[0], seg_end[0]) <= p[0] <= max(seg_start[0], seg_end[0]) and
            min(seg_start[1], seg_end[1]) <= p[1] <= max(seg_start[1], seg_end[1]))

def segments_share_line(p1, p2, p3, p4):
    """
    检查两条线段是否共享线段
    返回: (是否违规, 详细信息)
    """
    # 首先检查是否所有点都共线
    if not (are_collinear(p1, p2, p3) and are_collinear(p1, p2, p4)):
        return False, None
    
    # 如果共线，检查是否有重叠部分
    # 将线段投影到主轴（X或Y，取变化较大的那个）
    dx1 = abs(p2[0] - p1[0])
    dy1 = abs(p2[1] - p1[1])
    
    if dx1 >= dy1:
        # 使用X轴
        seg1 = (min(p1[0], p2[0]), max(p1[0], p2[0]))
        seg2 = (min(p3[0], p4[0]), max(p3[0], p4[0]))
    else:
        # 使用Y轴
        seg1 = (min(p1[1], p2[1]), max(p1[1], p2[1]))
        seg2 = (min(p3[1], p4[1]), max(p3[1], p4[1]))
    
    # 检查区间是否有交集（不只是端点接触）
    overlap_start = max(seg1[0], seg2[0])
    overlap_end = min(seg1[1], seg2[1])
    
    if overlap_start < overlap_end:
        # 有真正的重叠（不只是端点接触）
        return True, f"overlap range: [{overlap_start}, {overlap_end}]"
    elif overlap_start == overlap_end:
        # 只有一个点接触，检查是否是端点
        # 如果接触点不是两条边的共同端点，则违规
        touch_point_coords = None
        if dx1 >= dy1:
            # 在seg1中找对应的点
            if overlap_start == p1[0]:
                touch_point_coords = p1
            elif overlap_start == p2[0]:
                touch_point_coords = p2
            # 在seg2中验证
            if touch_point_coords and touch_point_coords not in [p3, p4]:
                return True, f"single point overlap at {touch_point_coords} (not a shared endpoint)"
        else:
            if overlap_start == p1[1]:
                touch_point_coords = p1
            elif overlap_start == p2[1]:
                touch_point_coords = p2
            if touch_point_coords and touch_point_coords not in [p3, p4]:
                return True, f"single point overlap at {touch_point_coords} (not a shared endpoint)"
    
    return False, None

def check_file(filepath):
    """检查一个文件中的所有边"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    nodes = {n['id']: (n['x'], n['y']) for n in data['nodes']}
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    violations = []
    
    for i, (s1, t1) in enumerate(edges):
        p1 = nodes[s1]
        p2 = nodes[t1]
        
        for j, (s2, t2) in enumerate(edges[i+1:], i+1):
            # 跳过共享端点的边（这是允许的）
            shared_endpoints = sum([s1 == s2, s1 == t2, t1 == s2, t1 == t2])
            if shared_endpoints > 0:
                continue
            
            p3 = nodes[s2]
            p4 = nodes[t2]
            
            is_violation, detail = segments_share_line(p1, p2, p3, p4)
            if is_violation:
                violations.append({
                    'edge1': (s1, t1),
                    'edge2': (s2, t2),
                    'coords1': (p1, p2),
                    'coords2': (p3, p4),
                    'detail': detail
                })
    
    return violations

if __name__ == '__main__':
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = [
        os.path.join(base_dir, 'results/11-29-04/15-nodes-cu-k4.json'),
        os.path.join(base_dir, 'results/11-29-04/70-nodes-cu-k24.json'),
        os.path.join(base_dir, 'results/11-29-04/100-nodes-cu-k28.json'),
    ]
    
    for filepath in files:
        try:
            violations = check_file(filepath)
            print(f"\n{'='*70}")
            print(f"File: {filepath}")
            print(f"{'='*70}")
            
            if violations:
                print(f"VIOLATION FOUND: {len(violations)} pairs of edges share line segments!")
                print("\nDetails:")
                for i, v in enumerate(violations[:10], 1):
                    print(f"\n{i}. Edge {v['edge1']} and Edge {v['edge2']}")
                    print(f"   Edge1: {v['coords1'][0]} -> {v['coords1'][1]}")
                    print(f"   Edge2: {v['coords2'][0]} -> {v['coords2'][1]}")
                    print(f"   Detail: {v['detail']}")
                
                if len(violations) > 10:
                    print(f"\n... and {len(violations) - 10} more violations")
            else:
                print("OK: No edges share line segments")
                
        except FileNotFoundError:
            print(f"\nFile not found: {filepath}")
        except Exception as e:
            print(f"\nError processing {filepath}: {e}")
