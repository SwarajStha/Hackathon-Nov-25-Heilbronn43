"""
Sprint 1: Geometry Core - Implementation
Pure integer arithmetic for all geometric computations.
Follows the principle: Math is Truth - no floating point comparisons.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Point:
    """
    Immutable 2D point on an integer grid.
    Value object - can be used as dictionary keys and in sets.
    """
    x: int
    y: int
    
    def __post_init__(self):
        """Validate that coordinates are integers."""
        if not isinstance(self.x, int):
            raise TypeError(f"x must be int, got {type(self.x)}")
        if not isinstance(self.y, int):
            raise TypeError(f"y must be int, got {type(self.y)}")
    
    def __hash__(self):
        """Make Point hashable for use in sets and as dict keys."""
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        """Check equality based on coordinates."""
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y
    
    def __repr__(self):
        """String representation for debugging."""
        return f"Point({self.x}, {self.y})"


class GeometryCore:
    """
    Static utility class for core geometric primitives.
    All operations use pure integer arithmetic.
    
    Mathematical Foundation:
    - Cross product: (B - O) × (C - O) = (B.x - O.x)(C.y - O.y) - (B.y - O.y)(C.x - O.x)
    - Intersection test: Two segments intersect if endpoints are on opposite sides
    """
    
    @staticmethod
    def cross_product(o: Point, a: Point, b: Point) -> int:
        """
        Calculate the cross product of vectors OA and OB.
        
        Returns:
            > 0: Counter-clockwise turn (B is left of OA)
            < 0: Clockwise turn (B is right of OA)
            = 0: Collinear (O, A, B are on the same line)
        
        Formula: (A.x - O.x)(B.y - O.y) - (A.y - O.y)(B.x - O.x)
        
        This is the signed area of the parallelogram formed by OA and OB,
        multiplied by 2.
        """
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    
    @staticmethod
    def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
        """
        Determine if two line segments (p1-p2) and (q1-q2) intersect.
        
        Uses the cross product method:
        - Two segments intersect if and only if:
          1. q1 and q2 are on opposite sides of line p1-p2, AND
          2. p1 and p2 are on opposite sides of line q1-q2
        
        Special cases (all return False):
        - Segments share an endpoint (not a crossing)
        - Endpoints touch (not a crossing)
        - Collinear segments (even if overlapping)
        
        Returns:
            True if segments properly intersect (cross each other)
            False otherwise
        """
        # Check for shared endpoints first (fast path)
        if p1 == q1 or p1 == q2 or p2 == q1 or p2 == q2:
            return False
        
        # Calculate cross products to determine orientation
        # For segment p1-p2 with points q1, q2:
        d1 = GeometryCore.cross_product(p1, p2, q1)
        d2 = GeometryCore.cross_product(p1, p2, q2)
        
        # For segment q1-q2 with points p1, p2:
        d3 = GeometryCore.cross_product(q1, q2, p1)
        d4 = GeometryCore.cross_product(q1, q2, p2)
        
        # Segments intersect if:
        # - q1 and q2 are on opposite sides of p1-p2 (d1 and d2 have opposite signs)
        # - p1 and p2 are on opposite sides of q1-q2 (d3 and d4 have opposite signs)
        #
        # Opposite signs means: d1 * d2 < 0 (strictly negative)
        # We use strictly less than to exclude:
        # - Collinear cases (where one product is 0)
        # - Endpoint touching (where one product is 0)
        
        if d1 * d2 < 0 and d3 * d4 < 0:
            return True
        
        # Handle special collinear cases
        # If all four cross products are 0, segments are collinear
        if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
            # Collinear segments - check for overlap using bounding box
            # Even if they overlap, we don't count this as a crossing
            return False
        
        # If only some cross products are 0, we have endpoint touching
        # or one segment touching the line of the other
        # These are NOT crossings in our definition
        return False
    
    @staticmethod
    def orientation(p: Point, q: Point, r: Point) -> int:
        """
        Find orientation of ordered triplet (p, q, r).
        
        Returns:
            0: Collinear
            1: Clockwise
            2: Counter-clockwise
        
        This is a convenience wrapper around cross_product.
        """
        val = GeometryCore.cross_product(p, q, r)
        if val == 0:
            return 0  # Collinear
        return 1 if val < 0 else 2  # Clockwise or Counter-clockwise
    
    @staticmethod
    def on_segment(p: Point, q: Point, r: Point) -> bool:
        """
        Check if point q lies on segment pr (given p, q, r are collinear).
        
        Returns:
            True if q is on segment pr, False otherwise
        """
        return (
            min(p.x, r.x) <= q.x <= max(p.x, r.x) and
            min(p.y, r.y) <= q.y <= max(p.y, r.y)
        )
    
    @staticmethod
    def point_on_segment_interior(p: Point, a: Point, b: Point) -> bool:
        """
        Check if point p lies strictly in the interior of segment a-b.
        (NOT including endpoints)
        
        Returns:
            True if p is strictly between a and b on the line segment
        """
        # Check if p is one of the endpoints
        if p == a or p == b:
            return False
        
        # Point must be collinear with segment
        if GeometryCore.cross_product(a, b, p) != 0:
            return False
        
        # Point must be within the bounding box of the segment
        # For strictly interior points, BOTH coordinates must be within bounds
        # (or equal for degenerate cases like vertical/horizontal lines)
        x_in_range = min(a.x, b.x) <= p.x <= max(a.x, b.x)
        y_in_range = min(a.y, b.y) <= p.y <= max(a.y, b.y)
        
        return x_in_range and y_in_range
    
    @staticmethod
    def segments_overlap(a1: Point, b1: Point, a2: Point, b2: Point) -> bool:
        """
        Check if two line segments overlap (share more than just an endpoint).
        
        Returns:
            True if segments share a non-trivial portion (overlap or one contains the other)
        """
        # Check if segments are collinear
        if GeometryCore.cross_product(a1, b1, a2) != 0:
            return False
        if GeometryCore.cross_product(a1, b1, b2) != 0:
            return False
        
        # Segments are collinear, check for overlap
        # Project onto dominant axis
        if abs(b1.x - a1.x) > abs(b1.y - a1.y):
            # Use x-axis
            seg1_min, seg1_max = min(a1.x, b1.x), max(a1.x, b1.x)
            seg2_min, seg2_max = min(a2.x, b2.x), max(a2.x, b2.x)
        else:
            # Use y-axis
            seg1_min, seg1_max = min(a1.y, b1.y), max(a1.y, b1.y)
            seg2_min, seg2_max = min(a2.y, b2.y), max(a2.y, b2.y)
        
        # Check for overlap (more than just endpoint touching)
        # Overlap exists if intervals intersect with more than a single point
        overlap_start = max(seg1_min, seg2_min)
        overlap_end = min(seg1_max, seg2_max)
        return overlap_start < overlap_end
    
    @staticmethod
    def point_to_segment_distance_squared(p: Point, a: Point, b: Point) -> int:
        """
        Calculate squared distance from point p to segment a-b.
        
        Returns integer squared distance (avoids floating point).
        Uses projection formula but keeps everything in integer arithmetic.
        """
        # Vector from a to b
        ab_x = b.x - a.x
        ab_y = b.y - a.y
        
        # Vector from a to p
        ap_x = p.x - a.x
        ap_y = p.y - a.y
        
        # Squared length of ab
        ab_sq = ab_x * ab_x + ab_y * ab_y
        
        # If a == b, return distance to point a
        if ab_sq == 0:
            return ap_x * ap_x + ap_y * ap_y
        
        # Dot product ap · ab
        dot = ap_x * ab_x + ap_y * ab_y
        
        # Parameter t (where projection lands on line)
        # We multiply by ab_sq to keep integer arithmetic
        # t_scaled = dot, compare with 0 and ab_sq
        
        if dot <= 0:
            # Projection before point a
            return ap_x * ap_x + ap_y * ap_y
        elif dot >= ab_sq:
            # Projection after point b
            bp_x = p.x - b.x
            bp_y = p.y - b.y
            return bp_x * bp_x + bp_y * bp_y
        else:
            # Projection on segment
            # Distance squared = |ap|² - (ap · ab)² / |ab|²
            # To avoid division, we compute: |ap|² * |ab|² - (ap · ab)²
            # Then divide result at the end (but we want squared distance)
            # Actually, for exact integer distance squared, this gets complex
            # Let's use the perpendicular distance formula
            
            # Cross product gives twice the signed area
            cross = ap_x * ab_y - ap_y * ab_x
            # Distance squared = (cross)² / ab_sq
            # But we want integer result, so we return (cross)² / ab_sq
            # However, this requires division. For comparisons, we can return (cross)²
            # and compare with threshold * ab_sq
            
            # For now, return a scaled value that preserves ordering
            return (cross * cross) // ab_sq if ab_sq > 0 else 0


class BoundingBox:
    """
    Axis-aligned bounding box for spatial queries.
    All coordinates are integers.
    """
    
    def __init__(self, min_x: int, min_y: int, max_x: int, max_y: int):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
    
    def contains(self, p: Point) -> bool:
        """Check if point is inside bounding box."""
        return (self.min_x <= p.x <= self.max_x and 
                self.min_y <= p.y <= self.max_y)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box intersects another."""
        return not (self.max_x < other.min_x or 
                   self.min_x > other.max_x or
                   self.max_y < other.min_y or 
                   self.min_y > other.max_y)
    
    @staticmethod
    def from_segment(p1: Point, p2: Point) -> 'BoundingBox':
        """Create bounding box from a line segment."""
        return BoundingBox(
            min(p1.x, p2.x),
            min(p1.y, p2.y),
            max(p1.x, p2.x),
            max(p1.y, p2.y)
        )
    
    def __repr__(self):
        return f"BBox({self.min_x}, {self.min_y}, {self.max_x}, {self.max_y})"
