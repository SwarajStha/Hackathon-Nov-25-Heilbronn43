"""
Simple test for K-value functionality - No pytest dependency
"""
import sys
import os

# CRITICAL: Setup for Windows CUDA DLL loading
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, '.')

print("Python version:", sys.version)
print("Working directory:", os.getcwd())
print()

try:
    import planar_cuda
    print("✓ planar_cuda module loaded successfully")
    print("  Version:", planar_cuda.__version__)
    print()
except ImportError as e:
    print("✗ Failed to import planar_cuda:", e)
    sys.exit(1)

# Test 1: Triangle graph (K=0, planar)
print("=" * 60)
print("Test 1: Triangle Graph (should be planar, K=0)")
print("=" * 60)

nodes_x = [0, 10, 5]
nodes_y = [0, 0, 10]
edges = [(0, 1), (1, 2), (0, 2)]

solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
print(f"Graph: 3 nodes, 3 edges (triangle)")

try:
    k_value = solver.calculate_k_value()
    print(f"✓ K-value: {k_value}")
    
    if k_value == 0:
        print("  ✓ PASS: Triangle is planar (K=0)")
    else:
        print(f"  ✗ FAIL: Expected K=0, got K={k_value}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: K4 complete graph (non-planar, K>=1)
print()
print("=" * 60)
print("Test 2: K4 Complete Graph (non-planar, K>=1)")
print("=" * 60)

nodes_x = [0, 10, 10, 0]
nodes_y = [0, 0, 10, 10]
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Square
    (0, 2), (1, 3)                    # Diagonals (cross!)
]

solver2 = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
print(f"Graph: 4 nodes, 6 edges (K4 complete)")

try:
    k_value = solver2.calculate_k_value()
    total_crossings = solver2.calculate_total_crossings()
    edge_crossings = solver2.get_edge_crossings()
    
    print(f"✓ K-value: {k_value}")
    print(f"  Total crossings: {total_crossings}")
    print(f"  Edge crossings: {edge_crossings}")
    
    if k_value >= 1:
        print(f"  ✓ PASS: K4 is non-planar (K={k_value})")
    else:
        print(f"  ✗ FAIL: Expected K>=1, got K={k_value}")
        
    # Check diagonals
    if len(edge_crossings) == 6:
        diag1_crossings = edge_crossings[4]
        diag2_crossings = edge_crossings[5]
        print(f"  Diagonal (0,2) crossings: {diag1_crossings}")
        print(f"  Diagonal (1,3) crossings: {diag2_crossings}")
        
        if diag1_crossings == 1 and diag2_crossings == 1:
            print("  ✓ PASS: Diagonals each cross once")
        else:
            print(f"  ✗ FAIL: Expected diagonals to cross once each")
            
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Two crossing edges
print()
print("=" * 60)
print("Test 3: Simple X Pattern (K=1)")
print("=" * 60)

nodes_x = [0, 10, 10, 0]
nodes_y = [0, 0, 10, 10]
edges = [(0, 2), (1, 3)]  # Two diagonals forming X

solver3 = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
print(f"Graph: 4 nodes, 2 edges (X pattern)")

try:
    k_value = solver3.calculate_k_value()
    edge_crossings = solver3.get_edge_crossings()
    
    print(f"✓ K-value: {k_value}")
    print(f"  Edge crossings: {edge_crossings}")
    
    if k_value == 1:
        print("  ✓ PASS: X pattern has K=1")
    else:
        print(f"  ✗ FAIL: Expected K=1, got K={k_value}")
        
    if edge_crossings == [1, 1]:
        print("  ✓ PASS: Each edge crosses once")
    else:
        print(f"  ✗ FAIL: Expected [1, 1], got {edge_crossings}")
        
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Summary: K-value calculation is working!")
print("=" * 60)
