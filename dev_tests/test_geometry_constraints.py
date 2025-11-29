#!/usr/bin/env python3
"""
测试几何约束检查函数
"""

import sys
sys.path.insert(0, 'src')

from LCNv1.core.geometry import Point, GeometryCore

# 测试 point_on_segment_interior
print("测试 point_on_segment_interior:")
print("="*60)

# 情况1: 点在线段内部
a = Point(0, 0)
b = Point(10, 10)
p1 = Point(5, 5)  # 在内部
p2 = Point(0, 0)  # 在端点
p3 = Point(15, 15)  # 在外部
p4 = Point(5, 6)  # 不共线

print(f"线段: {a} -> {b}")
print(f"  {p1} 在内部? {GeometryCore.point_on_segment_interior(p1, a, b)} (应该 True)")
print(f"  {p2} 在内部? {GeometryCore.point_on_segment_interior(p2, a, b)} (应该 False - 端点)")
print(f"  {p3} 在内部? {GeometryCore.point_on_segment_interior(p3, a, b)} (应该 False - 外部)")
print(f"  {p4} 在内部? {GeometryCore.point_on_segment_interior(p4, a, b)} (应该 False - 不共线)")

# 情况2: 水平线段
a2 = Point(0, 5)
b2 = Point(10, 5)
p5 = Point(5, 5)  # 在内部
p6 = Point(0, 5)  # 端点
p7 = Point(5, 6)  # 不共线

print(f"\n线段: {a2} -> {b2}")
print(f"  {p5} 在内部? {GeometryCore.point_on_segment_interior(p5, a2, b2)} (应该 True)")
print(f"  {p6} 在内部? {GeometryCore.point_on_segment_interior(p6, a2, b2)} (应该 False - 端点)")
print(f"  {p7} 在内部? {GeometryCore.point_on_segment_interior(p7, a2, b2)} (应该 False - 不共线)")

# 情况3: 垂直线段
a3 = Point(5, 0)
b3 = Point(5, 10)
p8 = Point(5, 5)  # 在内部
p9 = Point(5, 0)  # 端点
p10 = Point(6, 5)  # 不共线

print(f"\n线段: {a3} -> {b3}")
print(f"  {p8} 在内部? {GeometryCore.point_on_segment_interior(p8, a3, b3)} (应该 True)")
print(f"  {p9} 在内部? {GeometryCore.point_on_segment_interior(p9, a3, b3)} (应该 False - 端点)")
print(f"  {p10} 在内部? {GeometryCore.point_on_segment_interior(p10, a3, b3)} (应该 False - 不共线)")
