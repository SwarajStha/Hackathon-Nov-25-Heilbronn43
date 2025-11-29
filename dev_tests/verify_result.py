"""
验证我们生成的结果文件的交叉数
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from LCNv1.core.geometry import Point, GeometryCore


def count_crossings_correct(nodes_dict, edges):
    """
    正确计算交叉数（按题目定义）
    
    Returns:
        (k, total_crossings)
        - k: 最大交叉数（任一边的最大交叉数）
        - total_crossings: 总交叉数（所有交叉对的总数）
    """
    # 为每条边计数交叉数
    edge_list = list(edges)
    edge_crossing_counts = [0] * len(edge_list)
    
    print(f"检查 {len(edge_list)} 条边...")
    
    # 检查所有边对
    for i in range(len(edge_list)):
        for j in range(i + 1, len(edge_list)):
            e1 = edge_list[i]
            e2 = edge_list[j]
            
            # 跳过共享端点的边
            if (e1['source'] == e2['source'] or e1['source'] == e2['target'] or
                e1['target'] == e2['source'] or e1['target'] == e2['target']):
                continue
            
            p1 = Point(nodes_dict[e1['source']]['x'], nodes_dict[e1['source']]['y'])
            p2 = Point(nodes_dict[e1['target']]['x'], nodes_dict[e1['target']]['y'])
            p3 = Point(nodes_dict[e2['source']]['x'], nodes_dict[e2['source']]['y'])
            p4 = Point(nodes_dict[e2['target']]['x'], nodes_dict[e2['target']]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, p3, p4):
                # 成对计数：两条边都增加交叉数
                edge_crossing_counts[i] += 1
                edge_crossing_counts[j] += 1
    
    # 计算总交叉数（每个交叉被计数两次，所以除以2）
    total_crossings = sum(edge_crossing_counts) // 2
    
    # 计算K值（任一边的最大交叉数）
    k = max(edge_crossing_counts) if edge_crossing_counts else 0
    
    # 找出交叉数最多的边
    max_crossing_edges = [(i, edge_crossing_counts[i]) for i in range(len(edge_list)) 
                          if edge_crossing_counts[i] == k]
    
    print(f"\n每条边的交叉数分布:")
    crossing_dist = {}
    for count in edge_crossing_counts:
        crossing_dist[count] = crossing_dist.get(count, 0) + 1
    
    for cross_count in sorted(crossing_dist.keys()):
        print(f"  {cross_count} 个交叉: {crossing_dist[cross_count]} 条边")
    
    print(f"\n交叉数最多的边 (K={k}):")
    for i, count in max_crossing_edges[:5]:  # 只显示前5条
        e = edge_list[i]
        print(f"  边 {i}: ({e['source']} -> {e['target']}) 有 {count} 个交叉")
    
    return k, total_crossings


def main():
    # 读取我们生成的结果
    result_file = Path(__file__).parent / 'results' / '29-01-22' / '15-nodes-ours.json'
    
    if not result_file.exists():
        print(f"错误: 找不到文件 {result_file}")
        return
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    print(f"=== 验证 15-nodes-ours.json ===\n")
    print(f"节点数: {len(data['nodes'])}")
    print(f"边数: {len(data['edges'])}")
    print(f"画布大小: {data.get('width', '?')} x {data.get('height', '?')}")
    
    # 构建节点字典
    nodes_dict = {n['id']: n for n in data['nodes']}
    
    # 计算交叉数
    k, total = count_crossings_correct(nodes_dict, data['edges'])
    
    print(f"\n" + "="*70)
    print(f"验证结果:")
    print(f"  K值 (最大交叉数): {k}")
    print(f"  总交叉数: {total}")
    print(f"="*70)
    
    # 对比标准答案
    print(f"\n对比标准答案:")
    print(f"  标准答案: K=5, 总交叉数=62")
    print(f"  我们的结果: K={k}, 总交叉数={total}")
    print(f"  K值改进: {k - 5:+d}")
    print(f"  总交叉数改进: {total - 62:+d}")


if __name__ == '__main__':
    main()
