"""
Benchmark K-value Optimization on Real Instances

Tests the new K-value calculation and optimization functionality
on the actual benchmark instances.
"""
import os
import sys
import json
import time

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# Setup for CUDA DLL loading
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, '.')

import planar_cuda

def load_instance(filename):
    """Load a benchmark instance"""
    with open(filename) as f:
        data = json.load(f)
    
    nodes_x = [n['x'] for n in data['nodes']]
    nodes_y = [n['y'] for n in data['nodes']]
    edges = [(e['source'], e['target']) for e in data['edges']]
    
    return nodes_x, nodes_y, edges, data

def test_instance(name, filename, target_k, iterations=20000):
    """Test K-value optimization on a single instance"""
    print("=" * 70)
    print(f"Testing: {name}")
    print("=" * 70)
    
    # Load instance
    nodes_x, nodes_y, edges, data = load_instance(filename)
    
    print(f"Graph: {len(nodes_x)} nodes, {len(edges)} edges")
    print(f"Target: K ≤ {target_k}")
    print()
    
    # Create solver
    solver = planar_cuda.PlanarSolver(
        nodes_x, nodes_y, edges,
        cell_size=100,
        width=data.get('width', 1000000),
        height=data.get('height', 1000000)
    )
    
    # Initial state
    initial_k = solver.calculate_k_value()
    initial_total = solver.calculate_total_crossings()
    initial_edge_crossings = solver.get_edge_crossings()
    
    print(f"Initial state:")
    print(f"  K-value: {initial_k}")
    print(f"  Total crossings: {initial_total}")
    print(f"  Max edge crossings (K): {max(initial_edge_crossings) if initial_edge_crossings else 0}")
    print()
    
    # Run optimization (currently uses total crossing cost)
    print(f"Running SA optimization ({iterations:,} iterations)...")
    start_time = time.time()
    
    result = solver.run_sa_optimization(
        iterations=iterations,
        start_temp=100.0,
        cooling_rate=0.95
    )
    
    elapsed = time.time() - start_time
    
    # Final state
    final_k = solver.calculate_k_value()
    final_total = solver.calculate_total_crossings()
    final_edge_crossings = solver.get_edge_crossings()
    
    print()
    print(f"Final state:")
    print(f"  K-value: {final_k} (Δ = {final_k - initial_k:+d})")
    print(f"  Total crossings: {int(final_total)} (Δ = {int(final_total - initial_total):+d})")
    print(f"  Accepted moves: {int(result.get('accepted_moves', 0))}")
    print(f"  Time: {elapsed:.2f}s ({elapsed/iterations*1000:.2f} ms/iter)")
    print()
    
    # Analysis
    if final_k <= target_k:
        print(f"✓ SUCCESS: K={final_k} ≤ {target_k} (target achieved!)")
    else:
        print(f"✗ FAIL: K={final_k} > {target_k} (target not reached)")
        print(f"  Gap: {final_k - target_k} above target")
    
    # Show top-5 worst edges
    if final_edge_crossings:
        sorted_crossings = sorted(enumerate(final_edge_crossings), key=lambda x: x[1], reverse=True)
        print()
        print("Top-5 worst edges:")
        for i, (edge_idx, count) in enumerate(sorted_crossings[:5]):
            if count > 0:
                print(f"  {i+1}. Edge {edge_idx}: {count} crossings")
    
    print()
    return {
        'name': name,
        'nodes': len(nodes_x),
        'edges': len(edges),
        'initial_k': initial_k,
        'final_k': final_k,
        'target_k': target_k,
        'success': final_k <= target_k,
        'time': elapsed,
        'iterations': iterations
    }

if __name__ == "__main__":
    print()
    print("=" * 70)
    print(" K-VALUE OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print()
    print("Testing K-value calculation and optimization")
    print("NOTE: Current SA uses total crossing cost (not K-value cost yet)")
    print()
    
    results = []
    
    # Test instances
    instances = [
        ("15-nodes", "live-2025-example-instances/15-nodes.json", 5, 10000),
        ("70-nodes", "live-2025-example-instances/70.json", 10, 20000),
        ("100-nodes", "live-2025-example-instances/100.json", 10, 20000),
    ]
    
    for name, filename, target, iters in instances:
        try:
            result = test_instance(name, filename, target, iters)
            results.append(result)
        except FileNotFoundError:
            print(f"✗ SKIP: {filename} not found")
            print()
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Summary
    if results:
        print("=" * 70)
        print(" SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Instance':<15} {'Nodes':>6} {'Initial K':>10} {'Final K':>8} {'Target':>7} {'Status':>8} {'Time':>8}")
        print("-" * 70)
        
        for r in results:
            status = "✓ PASS" if r['success'] else "✗ FAIL"
            print(f"{r['name']:<15} {r['nodes']:>6} {r['initial_k']:>10} {r['final_k']:>8} {r['target_k']:>7} {status:>8} {r['time']:>7.2f}s")
        
        print()
        success_count = sum(1 for r in results if r['success'])
        print(f"Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.0f}%)")
        print()
        
        print("NOTE: To improve results, we need to implement K-value cost function")
        print("      (see docs/OOP_REFACTORING_DESIGN.md for next steps)")
