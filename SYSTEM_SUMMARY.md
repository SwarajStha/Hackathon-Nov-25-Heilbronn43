# K-Planar Graph Optimization System - Technical Summary

## Project Overview

This project implements a graph layout optimization system that minimizes edge crossings in planar graphs using Simulated Annealing (SA) with GPU acceleration. The primary objective is to minimize the K-value, defined as the maximum number of crossings on any single edge.

**Core Problem:**
- Input: A graph (nodes + edges) with initial coordinates
- Objective: Rearrange node positions to minimize edge crossings
- K-value: Maximum crossing count among all edges

---

## System Architecture

The system is implemented in two layers:

### 1. Python Layer (High-Level Logic)

#### Geometric Core (geometry.py)

**Mathematical Foundation: Integer Coordinate System**

All geometric computations use pure integer arithmetic to avoid floating-point precision errors.

**Core Algorithm: Cross Product**
```
Formula: (A - O) × (B - O) = (A.x - O.x)(B.y - O.y) - (A.y - O.y)(B.x - O.x)

Return values:
  > 0: Counter-clockwise turn (B is left of OA)
  < 0: Clockwise turn (B is right of OA)
  = 0: Collinear (O, A, B are on the same line)
```

**Line Segment Intersection Test (segments_intersect):**
```
Two segments intersect if and only if:
1. Endpoints of segment 1 are on opposite sides of segment 2, AND
2. Endpoints of segment 2 are on opposite sides of segment 1

Implementation:
- Compute 4 cross products: d1, d2, d3, d4
- Check for opposite signs: (d1 < 0 && d2 > 0) || (d1 > 0 && d2 < 0)
- Special cases: shared endpoints, collinearity → not a crossing
```

---

#### Cost Function (cost.py)

**Energy Function Definition:**
```
E = W_cross · Σ(k_i^p) + W_len · Σ(len_i^2)

Where:
- k_i: Number of crossings for edge i
- p: Penalty exponent (typically 2, quadratic penalty)
- W_cross: Crossing weight (default 100)
- W_len: Edge length weight (default 1)
```

**Key Optimization: Delta Calculation**

Problem: Recalculating the entire graph's cost after moving one node is expensive (O(E²))

Solution: Calculate only the change in cost
```
When moving node_id:
1. Only edges connected to node_id change position
2. Only recalculate crossings for these affected edges
3. ΔE = E_new - E_old (compute only the difference)

Complexity: O(d·k)
  - d = node degree (number of connected edges)
  - k = average edges per spatial hash cell
```

**Spatial Hash Acceleration:**
- Divide coordinate space into grid cells (cell_size = 50)
- Each edge only checks crossings with edges in the same grid cells
- Reduces global O(E²) comparison to O(E·k)

---

#### Simulated Annealing Algorithm (new.py)

**Algorithm Flow:**
```
1. Initialize: T₀ = 50, cooling_rate = 0.995
2. For each iteration:
   a. Randomly select a node
   b. Generate random new position (step size ∝ temperature)
   c. Calculate energy change ΔE (using delta calculation)
   d. Acceptance criterion:
      - If ΔE < 0 (improvement) → accept
      - If ΔE > 0 (worse) → accept with probability exp(-ΔE/T)
   e. Cool down: T = T × 0.995
   f. If stuck (500 iterations without improvement) → reheat
3. Return best solution found
```

**Metropolis Criterion (accepting worse solutions):**
```
P(accept) = exp(-ΔE/T)

- High temperature: more likely to accept worse solutions (exploration)
- Low temperature: only accept improvements (convergence)
```

---

### 2. CUDA Layer (GPU Acceleration)

#### Core Advantage: Parallelization

Python version problem:
- Checking E edges for crossings requires comparing E(E-1)/2 edge pairs
- For 150 edges: 11,175 comparisons needed
- Serial execution is too slow

CUDA solution:
- Each GPU thread processes one edge
- 150 edges → launch 150 parallel threads
- All 11,175 comparisons execute simultaneously

---

#### CUDA Core Functions (planar_cuda.cu)

**Kernel 1: Parallel Crossing Count**
```cpp
__global__ void count_crossings_kernel(...) {
    int tid = thread_ID;  // 0 to num_edges-1
    
    // Each thread handles one edge
    Edge edge_i = edges[tid];
    
    local_count = 0;
    
    // Check against all subsequent edges
    for (j = tid+1; j < num_edges; j++) {
        Edge edge_j = edges[j];
        
        if (segments_intersect(edge_i, edge_j)) {
            local_count++;
        }
    }
    
    // Atomic accumulation to global counter
    atomicAdd(crossings, local_count);
}
```

**Thread Configuration:**
- Block Size: 256 threads
- Grid Size: (num_edges + 255) / 256 blocks
- Total threads = number of edges (one-to-one mapping)

---

**Kernel 2: K-Value Calculation (per-edge crossing counts)**
```cpp
__global__ void count_edge_crossings_kernel(...) {
    int edge_idx = thread_ID;
    
    // Count crossings for this edge against all others
    int count = 0;
    for (j = 0; j < num_edges; j++) {
        if (j != edge_idx && segments_intersect(...)) {
            count++;
        }
    }
    
    // Store crossing count for this edge
    edge_crossings[edge_idx] = count;
}
```

**K-value = max(edge_crossings[])**
- Use parallel reduction to find maximum
- O(log N) complexity

---

**Kernel 3: Delta K-Value Calculation**
```cpp
__global__ void compute_delta_k_kernel(
    node_id,      // Node being moved
    new_x, new_y  // New position
) {
    // Each thread checks one edge
    Edge edge = edges[tid];
    
    // Calculate crossings before move
    count_before = count_crossings(edge, old_position);
    
    // Calculate crossings after move
    count_after = count_crossings(edge, new_position);
    
    // Store changes
    edge_crossings_before[tid] = count_before;
    edge_crossings_after[tid] = count_after;
}

// On CPU:
delta_k = max(edge_crossings_after) - max(edge_crossings_before)
```

---

#### Memory Management

**Strategy: Minimize CPU-GPU Transfers**
```cpp
// Upload once during initialization
cudaMalloc(&d_nodes_x, size);
cudaMemcpy(d_nodes_x, h_nodes_x, size, cudaMemcpyHostToDevice);

// Compute on GPU
calculate_crossings();  // Execute on GPU
update_node();          // Execute on GPU

// Download only final result
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
```

**RAII Principle (Resource Acquisition Is Initialization):**
```cpp
class PlanarSolver {
    // Constructor: allocate GPU memory
    PlanarSolver(...) {
        cudaMalloc(...);
    }
    
    // Destructor: automatic cleanup
    ~PlanarSolver() {
        cudaFree(...);
    }
};
```

---

## Complete System Workflow

```
1. Load graph from JSON file
   ↓
2. Apply FMME initial layout
   ↓
3. Upload nodes/edges to GPU (CUDA)
   ↓
4. Simulated Annealing loop (20,000 iterations):
   a. Randomly select node
   b. Calculate ΔE using CUDA
   c. Accept/reject move (Metropolis criterion)
   d. Update GPU node positions
   e. Cool down: T = T × 0.995
   ↓
5. Download best solution from GPU
   ↓
6. Export optimized layout to JSON
```

---

## Mathematical and Algorithmic Summary

### 1. Geometric Foundation
- Pure integer cross product for line segment intersection
- Avoids floating-point precision errors
- O(1) intersection test per edge pair

### 2. Optimization Algorithm
- Simulated Annealing with Metropolis criterion
- Temperature schedule: T₀=50, α=0.995
- Acceptance probability: P = exp(-ΔE/T)

### 3. Cost Function
- Energy: E = 100·Σk² + 1·Σlen²
- K-value objective: minimize max(k_i)
- Geometric constraint penalties: 10¹⁰ per violation

### 4. Acceleration Techniques
- **Spatial Hashing**: O(E²) → O(E·k) comparison reduction
- **Delta Calculation**: Incremental energy updates (only affected edges)
- **GPU Parallelization**: 3× speedup on large graphs (150+ nodes)

### 5. Constraint Enforcement
- No three collinear nodes
- No edge passing through non-endpoint nodes
- No overlapping edges on same line
- Penalty-based soft constraints (allows SA exploration)

---

## Implementation Highlights

### Python Components
1. **Geometry Core**: Integer-only geometric primitives
2. **Cost Function**: Energy calculation with spatial hash optimization
3. **Graph State**: Immutable point objects, state management
4. **Solver Strategy**: Pluggable architecture with factory pattern

### CUDA Components
1. **Parallel Crossing Detection**: O(E²) brute force parallelized
2. **K-Value Computation**: Per-edge crossing counts with reduction
3. **Delta Calculations**: Incremental updates for SA efficiency
4. **Memory Management**: RAII pattern, minimized transfers

### Key Design Patterns
- **Strategy Pattern**: Multiple solver implementations
- **Factory Pattern**: Dynamic solver instantiation
- **RAII**: Automatic GPU memory management
- **Immutability**: Point objects as value types
- **Separation of Concerns**: Geometry, graph, cost, optimization layers

---

## System Characteristics

### Strengths
- Mathematically rigorous (integer arithmetic)
- GPU-accelerated for large graphs
- Clean architectural separation
- Constraint-aware optimization
- Incremental delta calculations

### Complexity Analysis
- Crossing detection: O(E²) parallelized
- Delta calculation: O(d·k) with spatial hash
- K-value computation: O(E²) + O(E log E) reduction
- Space complexity: O(N + E) on GPU

### Scalability
- Small graphs (<50 nodes): CPU competitive
- Large graphs (>100 nodes): GPU 2-3× faster
- Throughput: ~510 iterations/second (stable)

---

**Version:** 1.0  
**Date:** December 2, 2025
