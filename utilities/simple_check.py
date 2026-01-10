#!/usr/bin/env python3
"""简单测试新的违规修复策略"""
import json
from collections import Counter

# 直接使用内置方式（不导入planar_cuda）
def check_file_violations(filepath):
    """检查文件违规"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    x = [n['x'] for n in data['nodes']]
    y = [n['y'] for n in data['nodes']]
    
    x_viol = len([c for c in Counter(x).values() if c >= 3])
    y_viol = len([c for c in Counter(y).values() if c >= 3])
    
    return x_viol, y_viol, x_viol + y_viol

# 检查原始文件
print("检查原始 225-nodes.json:")
x, y, total = check_file_violations("live-2025-example-instances/225-nodes.json")
print(f"  违规: 垂直={x}, 水平={y}, 总计={total}")

# 检查最新结果
print("\n检查最新结果 225-nodes-cu-k482.json:")
x2, y2, total2 = check_file_violations("results/06-55-34/225-nodes-cu-k482.json")
print(f"  违规: 垂直={x2}, 水平={y2}, 总计={total2}")
print(f"  改善: {total - total2} 个违规被修复 ({(total - total2)/total*100:.1f}%)")

print("\n结论:")
if total2 == 0:
    print("✅ 完全成功！所有违规已修复！")
elif total2 < total:
    print(f"⚠️ 部分成功：违规从 {total} 减少到 {total2}")
    print(f"   但仍有 {total2} 个违规存在")
    print(f"   说明：当前策略在减少违规，但未完全消除")
else:
    print(f"❌ 无改善：违规数量未减少")
