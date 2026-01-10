"""
Test CUDA Smart Moves on Real Instance (15 nodes)
"""
import sys
import json
import os
import time

# Import CUDA module
try:
    # Add CUDA DLL path
    os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
    
    build_dir = os.path.join(os.path.dirname(__file__), 'build_artifacts')
    sys.path.insert(0, build_dir)
    import planar_cuda
    print(f"✅ CUDA module loaded: {planar_cuda.__version__}\n")
except ImportError as e:
    print(f"❌ Failed to import planar_cuda: {e}")
    sys.exit(1)


def load_graph(filepath):
    """Load graph from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    nodes_x = [node['x'] for node in data['nodes']]
    nodes_y = [node['y'] for node in data['nodes']]
    edges = [(edge['source'], edge['target']) for edge in data['edges']]
    
    return nodes_x, nodes_y, edges, data.get('width', 1000000), data.get('height', 1000000)


def run_test(filepath, iterations=2000, runs=3):
    """Compare regular vs smart SA on real instance"""
    print("="*70)
    print(f"Testing: {filepath}")
    print("="*70)
    
    nodes_x, nodes_y, edges, width, height = load_graph(filepath)
    num_nodes = len(nodes_x)
    num_edges = len(edges)
    
    print(f"\n📊 Graph: {num_nodes} nodes, {num_edges} edges")
    print(f"   Grid: {width} x {height}")
    print(f"   Iterations: {iterations}, Runs: {runs}\n")
    
    # Test Regular SA
    print("🔄 Running REGULAR SA...")
    regular_times = []
    regular_k_values = []
    
    for run in range(runs):
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        start = time.time()
        stats = solver.run_sa_optimization(
            iterations=iterations,
            start_temp=100.0,
            cooling_rate=0.995,
            cost_function="total_crossings",
            use_smart_moves=False
        )
        elapsed = time.time() - start
        
        regular_times.append(elapsed)
        regular_k_values.append(stats['final_k'])
        
        print(f"  Run {run+1}: K={stats['final_k']:.0f}, time={elapsed:.3f}s")
    
    # Test Smart SA
    print("\n🔄 Running SMART SA...")
    smart_times = []
    smart_k_values = []
    
    for run in range(runs):
        solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
        
        start = time.time()
        stats = solver.run_sa_optimization(
            iterations=iterations,
            start_temp=100.0,
            cooling_rate=0.995,
            cost_function="total_crossings",
            use_smart_moves=True,
            smart_threshold=10.0
        )
        elapsed = time.time() - start
        
        smart_times.append(elapsed)
        smart_k_values.append(stats['final_k'])
        
        print(f"  Run {run+1}: K={stats['final_k']:.0f}, time={elapsed:.3f}s")
    
    # Summary
    print("\n" + "="*70)
    print("📈 SUMMARY")
    print("="*70)
    
    avg_regular_time = sum(regular_times) / len(regular_times)
    avg_smart_time = sum(smart_times) / len(smart_times)
    best_regular_k = min(regular_k_values)
    best_smart_k = min(smart_k_values)
    
    speedup = avg_regular_time / avg_smart_time
    k_improvement = best_regular_k - best_smart_k
    
    print(f"\n⏱️ Average Time:")
    print(f"  Regular: {avg_regular_time:.3f}s")
    print(f"  Smart:   {avg_smart_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    print(f"\n🎯 Best K-Value:")
    print(f"  Regular: {best_regular_k:.0f}")
    print(f"  Smart:   {best_smart_k:.0f}")
    print(f"  Improvement: {k_improvement:+.0f}")
    
    print(f"\n{'✅ SMART WINS!' if speedup > 1.0 and k_improvement >= 0 else '⚠️ Mixed results'}")
    print()


if __name__ == "__main__":
    # Find a 15-node instance
    import glob
    
    results_files = glob.glob("results/**/*15-nodes*.json", recursive=True)
    if results_files:
        test_file = results_files[0]
        print(f"Using: {test_file}\n")
        run_test(test_file, iterations=2000, runs=3)
    else:
        print("No 15-node files found, using sample.json")
        run_test("sample.json", iterations=500, runs=3)
