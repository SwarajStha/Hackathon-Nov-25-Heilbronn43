"""
智能移動生成器 - 從源頭避免違規
Smart Move Generator - Prevent Violations at Source

核心思想：
1. 不生成違規的移動，而不是生成後再檢測拒絕
2. 使用幾何約束保證合法性
3. 大幅減少計算浪費

Key Ideas:
1. Generate only valid moves instead of detect-and-reject
2. Use geometric constraints to ensure legality
3. Significantly reduce computation waste
"""
from typing import Tuple, List, Set, Optional
import random
import math

from .geometry import Point, GeometryCore
from .graph import GraphData, GridState
from .spatial_index import SpatialHash


class SmartMoveGenerator:
    """
    智能移動生成器
    
    策略：
    1. 避免重複坐標：不選擇已被佔用的位置
    2. 避免共線：檢查新位置是否在相鄰邊上
    3. 自適應搜索：如果連續失敗，擴大搜索半徑
    
    Strategy:
    1. Avoid duplicates: Don't select occupied positions
    2. Avoid collinearity: Check if new_pos is on incident edges
    3. Adaptive search: Expand radius if consecutive failures
    """
    
    def __init__(self, max_retries: int = 10):
        """
        Args:
            max_retries: 最大重試次數 (maximum retry attempts)
        """
        self.max_retries = max_retries
        self.geometry = GeometryCore()
        
    def generate_valid_move(
        self,
        graph: GraphData,
        state: GridState,
        node_id: int,
        step_size: int
    ) -> Optional[Point]:
        """
        生成一個保證合法的移動
        Generate a guaranteed valid move
        
        Args:
            graph: 圖拓撲 (graph topology)
            state: 當前狀態 (current state)
            node_id: 要移動的節點 (node to move)
            step_size: 移動半徑 (move radius)
            
        Returns:
            合法的新位置，或 None（如果嘗試多次都失敗）
            Valid new position, or None if all retries fail
        """
        old_pos = state.get_position(node_id)
        
        # 自適應搜索半徑
        # Adaptive search radius
        current_radius = step_size
        
        for attempt in range(self.max_retries):
            # 生成候選位置
            # Generate candidate position
            new_x = old_pos.x + random.randint(-current_radius, current_radius)
            new_y = old_pos.y + random.randint(-current_radius, current_radius)
            
            # 邊界裁剪
            # Clip to bounds
            new_x = max(0, min(state.width, new_x))
            new_y = max(0, min(state.height, new_y))
            new_pos = Point(new_x, new_y)
            
            # 檢查1：實時檢查是否與其他節點重複（不使用緩存）
            # Check 1: Real-time duplicate check (no caching)
            is_duplicate = False
            for nid in range(graph.num_nodes):
                if nid != node_id:
                    other_pos = state.get_position(nid)
                    if other_pos.x == new_pos.x and other_pos.y == new_pos.y:
                        is_duplicate = True
                        break
            
            if is_duplicate:
                current_radius = int(current_radius * 1.2)  # 擴大搜索範圍
                continue
            
            if is_duplicate:
                current_radius = int(current_radius * 1.2)  # 擴大搜索範圍
                continue
            
            # 檢查2：獲取與 node_id 相連的邊（實時獲取）
            # Check 2: Get incident edges (real-time)
            incident_edges = graph.get_incident_edges(node_id)
            
            # 檢查2：新位置是否在任何相鄰邊的內部
            # Check 2: Is new_pos on any incident edge interior?
            if self._is_on_incident_edge_interior(
                graph, state, node_id, new_pos, incident_edges
            ):
                current_radius = int(current_radius * 1.2)
                continue
            
            # 通過所有檢查！
            # Passed all checks!
            return new_pos
        
        # 所有嘗試都失敗，返回原位置（保持不動）
        # All retries failed, return original position (stay put)
        return old_pos
    
    def generate_fallback_move(
        self,
        graph: GraphData,
        state: GridState,
        node_id: int,
        step_size: int
    ) -> Point:
        """
        後備方案：快速生成移動（不保證合法，由 cost.py 檢查）
        Fallback: Fast move generation (not guaranteed valid, checked by cost.py)
        
        用於高溫階段，接受率高，不需要嚴格檢查
        For high temperature phase where acceptance rate is high
        """
        old_pos = state.get_position(node_id)
        
        new_x = old_pos.x + random.randint(-step_size, step_size)
        new_y = old_pos.y + random.randint(-step_size, step_size)
        
        # Clip to bounds
        new_x = max(0, min(state.width, new_x))
        new_y = max(0, min(state.height, new_y))
        
        return Point(new_x, new_y)
    
    def _get_occupied_positions(
        self,
        graph: GraphData,
        state: GridState,
        exclude_node: int
    ) -> Set[Point]:
        """
        獲取所有已佔用的位置（排除特定節點）
        Get all occupied positions (excluding specific node)
        """
        occupied = set()
        for nid in range(graph.num_nodes):
            if nid != exclude_node:
                pos = state.get_position(nid)
                occupied.add(pos)
        return occupied
    
    def _is_on_incident_edge_interior(
        self,
        graph: GraphData,
        state: GridState,
        node_id: int,
        new_pos: Point,
        incident_edges: List[Tuple[int, int]]
    ) -> bool:
        """
        檢查 new_pos 是否在任何相鄰邊的內部
        Check if new_pos is on interior of any incident edge
        
        相鄰邊：與 node_id 相連的邊
        Incident edges: Edges connected to node_id
        
        當 node_id 移動到 new_pos 後，這條邊的另一個端點與 new_pos 形成新的邊
        我們需要確保沒有其他節點落在這條新邊的內部
        
        When node_id moves to new_pos, the edge is reformed between
        new_pos and the other endpoint. We need to ensure no other
        node lies on the interior of this new edge.
        """
        for edge_idx in incident_edges:
            # Get edge endpoints
            src, tgt = graph.get_edge_endpoints(edge_idx)
            
            # 找到邊的另一個端點
            # Find the other endpoint
            other_node = tgt if src == node_id else src
            other_pos = state.get_position(other_node)
            
            # 檢查是否有任何其他節點在 [new_pos, other_pos] 線段的內部
            # Check if any other node is on interior of segment [new_pos, other_pos]
            for nid in range(graph.num_nodes):
                if nid == node_id or nid == other_node:
                    continue
                
                node_pos = state.get_position(nid)
                
                # 使用與 CUDA 相同的 point_on_segment_strict 邏輯
                # Use same point_on_segment_strict logic as CUDA
                if self._point_on_segment_strict(node_pos, new_pos, other_pos):
                    return True  # 違規！有節點在邊內部
        
        return False  # 合法
    
    def _point_on_segment_strict(
        self,
        point: Point,
        seg_a: Point,
        seg_b: Point
    ) -> bool:
        """
        檢查點是否在線段內部（排除端點）
        Check if point is on segment interior (excluding endpoints)
        
        匹配 CUDA 的 point_on_segment_interior 邏輯
        Matches CUDA's point_on_segment_interior logic
        """
        # 共線性檢查
        # Collinearity check
        cross = self.geometry.cross_product(point, seg_a, seg_b)
        if cross != 0:
            return False  # 不共線
        
        # 檢查是否在線段範圍內（排除端點）
        # Check if in segment range (excluding endpoints)
        min_x = min(seg_a.x, seg_b.x)
        max_x = max(seg_a.x, seg_b.x)
        min_y = min(seg_a.y, seg_b.y)
        max_y = max(seg_a.y, seg_b.y)
        
        # 嚴格不等式：排除端點
        # Strict inequality: exclude endpoints
        in_x_range = (min_x < point.x < max_x) if min_x != max_x else True
        in_y_range = (min_y < point.y < max_y) if min_y != max_y else True
        
        # 排除端點
        # Exclude endpoints
        if point == seg_a or point == seg_b:
            return False
        
        return in_x_range and in_y_range


class HybridMoveGenerator(SmartMoveGenerator):
    """
    混合移動生成器
    Hybrid Move Generator
    
    策略：根據溫度自適應選擇生成策略
    Strategy: Adaptively choose generation strategy based on temperature
    
    - 高溫（temp > threshold）：使用快速生成（後備方案）
    - 低溫（temp <= threshold）：使用智能生成（保證合法）
    
    - High temp (temp > threshold): Use fast generation (fallback)
    - Low temp (temp <= threshold): Use smart generation (guaranteed valid)
    
    原理：
    - 高溫階段：接受率高，即使生成違規移動，違規檢查會拒絕，浪費不大
    - 低溫階段：接受率低，生成合法移動避免浪費計算
    
    Rationale:
    - High temp: High acceptance rate, violation check rejects invalid moves anyway
    - Low temp: Low acceptance rate, generate valid moves to avoid waste
    """
    
    def __init__(
        self,
        max_retries: int = 10,
        smart_threshold: float = 10.0
    ):
        """
        Args:
            max_retries: 智能生成的最大重試次數
            smart_threshold: 溫度閾值（低於此值使用智能生成）
        """
        super().__init__(max_retries)
        self.smart_threshold = smart_threshold
        
    def generate_move(
        self,
        graph: GraphData,
        state: GridState,
        node_id: int,
        step_size: int,
        temperature: float
    ) -> Point:
        """
        根據溫度選擇生成策略
        Choose generation strategy based on temperature
        """
        if temperature <= self.smart_threshold:
            # 低溫：使用智能生成
            # Low temp: Use smart generation
            result = self.generate_valid_move(graph, state, node_id, step_size)
            if result is None:
                # 後備方案
                # Fallback
                return self.generate_fallback_move(graph, state, node_id, step_size)
            return result
        else:
            # 高溫：使用快速生成
            # High temp: Use fast generation
            return self.generate_fallback_move(graph, state, node_id, step_size)
