#!/usr/bin/env python3
"""
修復坐標違規 - 裁剪到有效範圍
將所有坐標限制在 [0, width] x [0, height] 範圍內
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, 'src')
from LCNv1.core.geometry import Point, GeometryCore


def clip_coordinates(data):
    """
    裁剪坐標到有效範圍
    
    Args:
        data: 圖數據字典
    
    Returns:
        修復後的數據,裁剪統計
    """
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    clipped_count = 0
    violations = {
        'x_negative': 0,
        'y_negative': 0,
        'x_overflow': 0,
        'y_overflow': 0
    }
    
    for node in data['nodes']:
        original_x = node['x']
        original_y = node['y']
        
        # 記錄違規
        if original_x < 0:
            violations['x_negative'] += 1
        if original_y < 0:
            violations['y_negative'] += 1
        if original_x > width:
            violations['x_overflow'] += 1
        if original_y > height:
            violations['y_overflow'] += 1
        
        # 裁剪
        node['x'] = max(0, min(node['x'], width))
        node['y'] = max(0, min(node['y'], height))
        
        if node['x'] != original_x or node['y'] != original_y:
            clipped_count += 1
    
    return data, clipped_count, violations


def calculate_k_value(data):
    """計算K值和總交叉數"""
    nodes = {n['id']: n for n in data['nodes']}
    edges = data['edges']
    
    edge_crossings = {}
    
    for i in range(len(edges)):
        e1 = edges[i]
        src1, tgt1 = e1['source'], e1['target']
        p1 = Point(nodes[src1]['x'], nodes[src1]['y'])
        p2 = Point(nodes[tgt1]['x'], nodes[tgt1]['y'])
        
        crossings = 0
        for j in range(len(edges)):
            if i == j:
                continue
            
            e2 = edges[j]
            src2, tgt2 = e2['source'], e2['target']
            
            # 跳過共享端點
            if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                continue
            
            q1 = Point(nodes[src2]['x'], nodes[src2]['y'])
            q2 = Point(nodes[tgt2]['x'], nodes[tgt2]['y'])
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                crossings += 1
        
        edge_crossings[i] = crossings
    
    k = max(edge_crossings.values()) if edge_crossings else 0
    total = sum(edge_crossings.values()) // 2
    
    return k, total


def fix_result_file(filepath, backup=True):
    """
    修復單個結果文件
    
    Args:
        filepath: 文件路徑
        backup: 是否備份原文件
    
    Returns:
        修復統計
    """
    filepath = Path(filepath)
    
    print(f"\n{'='*80}")
    print(f"修復: {filepath.name}")
    print(f"{'='*80}")
    
    # 加載數據
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 檢查原始狀態
    x_coords = [n['x'] for n in data['nodes']]
    y_coords = [n['y'] for n in data['nodes']]
    width = data.get('width', 1000000)
    height = data.get('height', 1000000)
    
    print(f"\n原始狀態:")
    print(f"  X範圍: [{min(x_coords)}, {max(x_coords)}] (限制: [0, {width}])")
    print(f"  Y範圍: [{min(y_coords)}, {max(y_coords)}] (限制: [0, {height}])")
    
    # 計算原始K值
    k_before, total_before = calculate_k_value(data)
    print(f"  K值: {k_before}, 總交叉數: {total_before}")
    
    # 備份
    if backup:
        backup_path = filepath.parent / f"{filepath.stem}_backup{filepath.suffix}"
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n備份已保存: {backup_path.name}")
    
    # 裁剪坐標
    data, clipped_count, violations = clip_coordinates(data)
    
    # 檢查修復後狀態
    x_coords = [n['x'] for n in data['nodes']]
    y_coords = [n['y'] for n in data['nodes']]
    
    print(f"\n修復後狀態:")
    print(f"  X範圍: [{min(x_coords)}, {max(x_coords)}]")
    print(f"  Y範圍: [{min(y_coords)}, {max(y_coords)}]")
    print(f"  裁剪節點數: {clipped_count}/{len(data['nodes'])}")
    
    if violations['x_negative'] > 0:
        print(f"  - X負值: {violations['x_negative']} 個")
    if violations['y_negative'] > 0:
        print(f"  - Y負值: {violations['y_negative']} 個")
    if violations['x_overflow'] > 0:
        print(f"  - X超出: {violations['x_overflow']} 個")
    if violations['y_overflow'] > 0:
        print(f"  - Y超出: {violations['y_overflow']} 個")
    
    # 計算修復後K值
    k_after, total_after = calculate_k_value(data)
    print(f"  K值: {k_after}, 總交叉數: {total_after}")
    
    # 顯示變化
    if k_after != k_before or total_after != total_before:
        print(f"\n變化:")
        print(f"  K值: {k_before} -> {k_after} ({k_after - k_before:+d})")
        print(f"  總交叉數: {total_before} -> {total_after} ({total_after - total_before:+d})")
    
    # 保存修復後的文件
    # 如果K值改變,更新文件名
    if k_after != k_before:
        new_filename = filepath.stem.rsplit('-k', 1)[0] + f"-k{k_after}{filepath.suffix}"
        new_filepath = filepath.parent / new_filename
        
        # 刪除舊文件
        filepath.unlink()
        
        with open(new_filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n保存為: {new_filepath.name}")
        return {
            'file': new_filepath.name,
            'k_before': k_before,
            'k_after': k_after,
            'total_before': total_before,
            'total_after': total_after,
            'clipped': clipped_count
        }
    else:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n已更新: {filepath.name}")
        return {
            'file': filepath.name,
            'k_before': k_before,
            'k_after': k_after,
            'total_before': total_before,
            'total_after': total_after,
            'clipped': clipped_count
        }


def main():
    print("="*80)
    print("坐標約束修復工具")
    print("="*80)
    print("\n功能: 將所有坐標裁剪到 [0, width] x [0, height] 範圍內")
    
    # 修復所有結果文件
    results_dir = Path('results/11-29-03')
    
    result_files = list(results_dir.glob('*-nodes-cu-k*.json'))
    
    if not result_files:
        print(f"\n沒有找到結果文件在 {results_dir}")
        return
    
    print(f"\n找到 {len(result_files)} 個文件需要檢查")
    
    all_stats = []
    
    for filepath in sorted(result_files):
        stats = fix_result_file(filepath, backup=True)
        all_stats.append(stats)
    
    # 總結
    print(f"\n{'='*80}")
    print("總結")
    print(f"{'='*80}")
    print(f"\n{'文件':<30} {'K值變化':<12} {'交叉數變化':<15} {'裁剪節點':<10}")
    print("-"*70)
    for stats in all_stats:
        k_change = f"{stats['k_before']}->{stats['k_after']}"
        total_change = f"{stats['total_before']}->{stats['total_after']}"
        print(f"{stats['file']:<30} {k_change:<12} {total_change:<15} {stats['clipped']:<10}")
    
    print(f"\n完成! 所有文件已修復並符合坐標約束")
    print(f"備份文件已保存 (_backup後綴)")


if __name__ == "__main__":
    main()
