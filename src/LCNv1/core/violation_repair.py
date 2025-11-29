"""
违规修复算法

修复三种违规：
1. 重复坐标
2. 节点在边内部
3. 边重叠
"""

import random
from typing import Set, Tuple, List
from .geometry import Point, GeometryCore
from .graph import GraphData, GridState


class ViolationRepair:
    """违规修复器"""
    
    def __init__(self, graph: GraphData, state: GridState, max_attempts: int = 1000):
        self.graph = graph
        self.state = state
        self.max_attempts = max_attempts
    
    def detect_violations(self) -> dict:
        """
        检测所有违规
        
        Returns:
            {
                'duplicate_coords': [(pos, [node_ids])],
                'nodes_on_edges': [(node_id, edge_idx)],
                'overlapping_edges': [(edge1_idx, edge2_idx)]
            }
        """
        violations = {
            'duplicate_coords': [],
            'nodes_on_edges': [],
            'overlapping_edges': []
        }
        
        # 1. 检查重复坐标
        coord_to_nodes = {}
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            if pos not in coord_to_nodes:
                coord_to_nodes[pos] = []
            coord_to_nodes[pos].append(node_id)
        
        for pos, nodes in coord_to_nodes.items():
            if len(nodes) > 1:
                violations['duplicate_coords'].append((pos, nodes))
        
        # 2. 检查节点在边内部
        for node_id in range(self.graph.num_nodes):
            node_pos = self.state.get_position(node_id)
            
            for edge_idx in range(len(self.graph.edges)):
                src, tgt = self.graph.get_edge_endpoints(edge_idx)
                
                # 跳过端点
                if node_id == src or node_id == tgt:
                    continue
                
                src_pos = self.state.get_position(src)
                tgt_pos = self.state.get_position(tgt)
                
                if GeometryCore.point_on_segment_interior(node_pos, src_pos, tgt_pos):
                    violations['nodes_on_edges'].append((node_id, edge_idx))
        
        # 3. 检查边重叠
        for i in range(len(self.graph.edges)):
            src1, tgt1 = self.graph.get_edge_endpoints(i)
            a1 = self.state.get_position(src1)
            b1 = self.state.get_position(tgt1)
            
            for j in range(i + 1, len(self.graph.edges)):
                src2, tgt2 = self.graph.get_edge_endpoints(j)
                
                # 跳过共享端点
                if src1 in (src2, tgt2) or tgt1 in (src2, tgt2):
                    continue
                
                a2 = self.state.get_position(src2)
                b2 = self.state.get_position(tgt2)
                
                if GeometryCore.segments_overlap(a1, b1, a2, b2):
                    violations['overlapping_edges'].append((i, j))
        
        return violations
    
    def repair_all(self, verbose: bool = False) -> bool:
        """
        修复所有违规
        
        Returns:
            True if all violations fixed, False if failed
        """
        iteration = 0
        
        while iteration < self.max_attempts:
            violations = self.detect_violations()
            
            total_violations = (len(violations['duplicate_coords']) +
                              len(violations['nodes_on_edges']) +
                              len(violations['overlapping_edges']))
            
            if total_violations == 0:
                if verbose:
                    print(f"✅ 所有违规已修复 (迭代 {iteration})")
                return True
            
            if verbose and iteration % 100 == 0:
                print(f"  [{iteration:>4}] 违规: {total_violations} "
                      f"(重复:{len(violations['duplicate_coords'])}, "
                      f"节点在边:{len(violations['nodes_on_edges'])}, "
                      f"边重叠:{len(violations['overlapping_edges'])})")
            
            # 优先修复最严重的违规
            if violations['duplicate_coords']:
                self._fix_duplicate_coords(violations['duplicate_coords'][0])
            elif violations['nodes_on_edges']:
                self._fix_node_on_edge(violations['nodes_on_edges'][0])
            elif violations['overlapping_edges']:
                self._fix_overlapping_edges(violations['overlapping_edges'][0])
            
            iteration += 1
        
        if verbose:
            print(f"❌ 修复失败: 达到最大迭代次数 {self.max_attempts}")
        return False
    
    def _fix_duplicate_coords(self, violation: Tuple[Point, List[int]]):
        """修复重复坐标：随机移动一个节点"""
        pos, nodes = violation
        
        # 移动第二个节点（保留第一个）
        node_to_move = nodes[1]
        
        # 尝试在附近找空位
        for _ in range(100):
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_pos = Point(
                max(0, min(self.state.width, pos.x + dx)),
                max(0, min(self.state.height, pos.y + dy))
            )
            
            if not self.state.is_occupied(new_pos):
                # 临时禁用约束检查
                old_enable = self.state._enable_constraints
                self.state._enable_constraints = False
                try:
                    self.state.move_node(node_to_move, new_pos)
                    return
                finally:
                    self.state._enable_constraints = old_enable
    
    def _fix_node_on_edge(self, violation: Tuple[int, int]):
        """修复节点在边内部：移动节点到垂直方向，使用更大范围"""
        node_id, edge_idx = violation
        
        src, tgt = self.graph.get_edge_endpoints(edge_idx)
        src_pos = self.state.get_position(src)
        tgt_pos = self.state.get_position(tgt)
        node_pos = self.state.get_position(node_id)
        
        # 计算边的方向向量
        dx = tgt_pos.x - src_pos.x
        dy = tgt_pos.y - src_pos.y
        
        # 垂直向量 (顺时针90度)
        perp_dx = dy
        perp_dy = -dx
        
        # 归一化（整数近似）
        length = max(abs(perp_dx), abs(perp_dy), 1)
        
        # 尝试多个距离：从5到30
        for distance in [5, 10, 15, 20, 30]:
            perp_dx_scaled = (perp_dx * distance) // length
            perp_dy_scaled = (perp_dy * distance) // length
            
            # 尝试两个方向
            for sign in [1, -1]:
                new_x = node_pos.x + sign * perp_dx_scaled
                new_y = node_pos.y + sign * perp_dy_scaled
                
                # 边界约束
                new_x = max(0, min(self.state.width, new_x))
                new_y = max(0, min(self.state.height, new_y))
                new_pos = Point(new_x, new_y)
                
                if not self.state.is_occupied(new_pos):
                    # 临时禁用约束检查
                    old_enable = self.state._enable_constraints
                    self.state._enable_constraints = False
                    try:
                        self.state.move_node(node_id, new_pos)
                        return
                    finally:
                        self.state._enable_constraints = old_enable
        
        # 如果垂直方向都不行，使用更大范围随机移动
        self._random_move_node(node_id, max_range=50)
    
    def _fix_overlapping_edges(self, violation: Tuple[int, int]):
        """修复边重叠：移动其中一条边的一个端点，使用更大范围"""
        edge1_idx, edge2_idx = violation
        
        # 移动第二条边的一个端点
        src, tgt = self.graph.get_edge_endpoints(edge2_idx)
        
        # 随机选择移动哪个端点
        node_to_move = random.choice([src, tgt])
        self._random_move_node(node_to_move, max_range=50)
    
    def _random_move_node(self, node_id: int, max_range: int = 30):
        """随机移动节点到附近空位，使用可配置的范围"""
        old_pos = self.state.get_position(node_id)
        
        # 尝试不同的距离范围
        for attempt_range in [max_range, max_range * 2, max_range * 3]:
            for _ in range(100):
                dx = random.randint(-attempt_range, attempt_range)
                dy = random.randint(-attempt_range, attempt_range)
                new_x = max(0, min(self.state.width, old_pos.x + dx))
                new_y = max(0, min(self.state.height, old_pos.y + dy))
                new_pos = Point(new_x, new_y)
                
                if not self.state.is_occupied(new_pos):
                    # 临时禁用约束检查
                    old_enable = self.state._enable_constraints
                    self.state._enable_constraints = False
                    try:
                        self.state.move_node(node_id, new_pos)
                        return
                    finally:
                        self.state._enable_constraints = old_enable
