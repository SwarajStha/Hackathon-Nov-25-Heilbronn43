"""
Test CUDA Smart Move Generation
Tests the GPU implementation of violation-avoiding move generation
"""
import sys
import json
import random
import time

# Import CUDA module
import os
try:
    # Add CUDA DLL path
    os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
    
    build_dir = os.path.join(os.path.dirname(__file__), 'build_artifacts')
    sys.path.insert(0, build_dir)
    import planar_cuda
    print(f"✅ CUDA module loaded: {planar_cuda.__version__}")
except ImportError as e:
    print(f"❌ Failed to import planar_cuda: {e}")
    print(f"Build dir: {build_dir}")
    sys.exit(1)


def load_graph(filepath):
    """Load graph from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    nodes_x = [node['x'] for node in data['nodes']]
    nodes_y = [node['y'] for node in data['nodes']]
    edges = [(edge['source'], edge['target']) for edge in data['edges']]
    
    return nodes_x, nodes_y, edges


def test_smart_move_generation():
    """Test smart move generation directly"""
    print("\n" + "="*60)
    print("Test 1: Smart Move Generation (Direct)")
    print("="*60)
    
    # Load test graph
    filepath = "sample.json"
    nodes_x, nodes_y, edges = load_graph(filepath)
    
    # Create solver
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    # Test smart move generation for first node
    node_id = 0
    step_size = 100
    
    print(f"\n🔍 Testing smart move for node {node_id}")
    print(f"Original position: ({nodes_x[node_id]}, {nodes_y[node_id]})")
    
    # Generate 5 smart moves
    for i in range(5):
        new_pos = solver.generate_smart_move(node_id, step_size, max_retries=10)
        print(f"  Smart move {i+1}: ({new_pos[0]}, {new_pos[1]})")


def test_sa_with_smart_moves(use_smart=False):
    """Test SA with/without smart moves"""
    print("\n" + "="*60)
    print(f"Test 2: SA Optimization (smart_moves={use_smart})")
    print("="*60)
    
    # Load test graph
    filepath = "sample.json"
    nodes_x, nodes_y, edges = load_graph(filepath)
    
    # Create solver
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    # Run SA
    iterations = 500
    start_temp = 100.0
    cooling_rate = 0.995
    
    print(f"\n⚙️ Parameters:")
    print(f"  Iterations: {iterations}")
    print(f"  Start temp: {start_temp}")
    print(f"  Cooling rate: {cooling_rate}")
    print(f"  Smart moves: {use_smart}")
    
    start_time = time.time()
    
    stats = solver.run_sa_optimization(
        iterations=iterations,
        start_temp=start_temp,
        cooling_rate=cooling_rate,
        cost_function="total_crossings",
        use_smart_moves=use_smart,
        smart_threshold=10.0
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n📊 Results:")
    print(f"  Final crossings: {stats['final_crossings']:.0f}")
    print(f"  Final K-value: {stats['final_k']:.0f}")
    print(f"  Accepted moves: {stats['accepted_moves']:.0f}")
    print(f"  Rejected moves: {stats['rejected_moves']:.0f}")
    print(f"  Time: {elapsed:.3f}s")
    
    return stats, elapsed


def compare_strategies():
    """Compare regular vs smart SA"""
    print("\n" + "="*60)
    print("Test 3: Strategy Comparison")
    print("="*60)
    
    # Run both strategies
    print("\n🔄 Running REGULAR SA...")
    stats_regular, time_regular = test_sa_with_smart_moves(use_smart=False)
    
    print("\n🔄 Running SMART SA...")
    stats_smart, time_smart = test_sa_with_smart_moves(use_smart=True)
    
    # Compare
    print("\n" + "="*60)
    print("📈 Comparison")
    print("="*60)
    
    speedup = time_regular / time_smart
    k_diff = stats_smart['final_k'] - stats_regular['final_k']
    
    print(f"\n⏱️ Speed:")
    print(f"  Regular: {time_regular:.3f}s")
    print(f"  Smart:   {time_smart:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    print(f"\n🎯 K-Value:")
    print(f"  Regular: {stats_regular['final_k']:.0f}")
    print(f"  Smart:   {stats_smart['final_k']:.0f}")
    print(f"  Diff:    {k_diff:+.0f} ({'+better' if k_diff < 0 else 'worse'})")


if __name__ == "__main__":
    random.seed(42)
    
    # Test 1: Direct smart move generation
    test_smart_move_generation()
    
    # Test 2: SA comparison
    compare_strategies()
    
    print("\n✅ All tests completed!")
