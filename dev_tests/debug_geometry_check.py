"""
深入调试几何检查
"""

from src.LCNv1.core.graph import Point
from src.LCNv1.core.geometry import GeometryCore

def main():
    # 测试点是否在线段内部
    # Edge (8, 12): Point(131, 34) -> Point(100, 42)
    a = Point(131, 34)
    b = Point(100, 42)
    
    print(f"Segment: {a} -> {b}")
    print(f"Direction: dx={b.x-a.x}, dy={b.y-a.y}")
    
    # 寻找真正在线段上的整数点
    # 参数化: P(t) = a + t*(b-a)
    # (x,y) = (131, 34) + t*(-31, 8)
    # 对于整数点，需要gcd(-31, 8) = 1，所以每步都生成新的整数点
    
    print("\n找到线段上的所有整数点：")
    points_on_segment = []
    for t_num in range(0, 32):  # t从0到1，步长1/31
        x = 131 - t_num
        y = 34 + (t_num * 8) // 31  # 整数近似
        p = Point(x, y)
        cross = GeometryCore.cross_product(a, b, p)
        if cross == 0:  # 真正在线上
            points_on_segment.append((p, t_num))
            print(f"  t={t_num}/31: {p}, cross={cross}")
    
    if len(points_on_segment) > 2:  # 除了端点还有其他点
        # 选择一个非端点的点来测试
        test_point = points_on_segment[len(points_on_segment)//2][0]
        print(f"\n选择测试点: {test_point}")
        
        result = GeometryCore.point_on_segment_interior(test_point, a, b)
        print(f"point_on_segment_interior result: {result} (should be True)")
    else:
        print("\n只有端点是整数点！创建一个非常接近的点来测试逻辑...")
        # 使用更简单的线段
        a2 = Point(0, 0)
        b2 = Point(10, 10)
        mid2 = Point(5, 5)
        cross = GeometryCore.cross_product(a2, b2, mid2)
        result = GeometryCore.point_on_segment_interior(mid2, a2, b2)
        print(f"Simple segment: {a2} -> {b2}")
        print(f"Midpoint: {mid2}, cross={cross}, on_interior={result} (should be True)")

if __name__ == '__main__':
    main()
