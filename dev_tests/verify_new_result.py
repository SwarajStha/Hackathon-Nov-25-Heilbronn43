#!/usr/bin/env python3
"""验证新生成的结果文件"""

import json

def verify_result(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查重复坐标
    coords = {}
    duplicates = []
    for node in data['nodes']:
        coord = (node['x'], node['y'])
        if coord in coords:
            duplicates.append((coord, [coords[coord], node['id']]))
        coords[coord] = node['id']
    
    print(f"文件: {filepath}")
    print(f"节点数: {len(data['nodes'])}")
    
    if duplicates:
        print(f"❌ 重复坐标: {len(duplicates)}组")
        for coord, nodes in duplicates:
            print(f"  {coord}: 节点{nodes}")
        return False
    else:
        print(f"✅ 所有坐标唯一")
        return True

print("="*60)
verify_result("results/29-02-18/15-nodes-ours.json")
print("="*60)
