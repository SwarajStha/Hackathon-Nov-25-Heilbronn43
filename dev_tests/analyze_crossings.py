"""
详细分析15-nodes-ours.json的所有交叉
找出所有交叉的边对
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from LCNv1.core.geometry import Point, GeometryCore


def analyze_all_crossings(result_file):
    """分析所有交叉"""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    nodes_dict = {n['id']: n for n in data['nodes']}
    edges = data['edges']
    
    print(f"=== 分析 {result_file.name} ===\n")
    print(f"节点数: {len(data['nodes'])}, 边数: {len(edges)}\n")
    
    # 记录所有交叉
    all_crossings = []
    edge_crossing_counts = [0] * len(edges)
    
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            
            # 跳过共享端点的边
            if (e1['source'] == e2['source'] or e1['source'] == e2['target'] or
                e1['target'] == e2['source'] or e1['target'] == e2['target']):
                continue
            
            p1 = Point(nodes_dict[e1['source']]['x'], nodes_dict[e1['source']]['y'])
            p2 = Point(nodes_dict[e1['target']]['x'], nodes_dict[e1['target']]['y'])
            p3 = Point(nodes_dict[e2['source']]['x'], nodes_dict[e2['source']]['y'])
            p4 = Point(nodes_dict[e2['target']]['x'], nodes_dict[e2['target']]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, p3, p4):
                all_crossings.append((i, j, e1, e2))
                edge_crossing_counts[i] += 1
                edge_crossing_counts[j] += 1
    
    # 打印所有交叉
    print(f"总共找到 {len(all_crossings)} 个交叉:\n")
    for idx, (i, j, e1, e2) in enumerate(all_crossings, 1):
        print(f"{idx}. 边{i} ({e1['source']}->{e1['target']}) × 边{j} ({e2['source']}->{e2['target']})")
    
    # K值
    k = max(edge_crossing_counts) if edge_crossing_counts else 0
    
    print(f"\n" + "="*70)
    print(f"K值 (最大交叉数): {k}")
    print(f"总交叉数: {len(all_crossings)}")
    print(f"="*70)
    
    # 找出交叉最多的边
    print(f"\n交叉数最多的边:")
    max_edges = [(i, edge_crossing_counts[i], edges[i]) for i in range(len(edges)) 
                 if edge_crossing_counts[i] == k]
    for i, count, e in max_edges:
        print(f"  边{i}: ({e['source']}->{e['target']}) 有 {count} 个交叉")


def main():
    # 检查最新的结果
    result_file = Path(__file__).parent / 'results' / '29-01-22' / '15-nodes-ours.json'
    
    if result_file.exists():
        analyze_all_crossings(result_file)
    else:
        print(f"找不到文件: {result_file}")


if __name__ == '__main__':
    main()
