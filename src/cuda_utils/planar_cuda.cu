/**
 * CUDA Implementation: Cycle 1 - Geometry Verification
 * 
 * OOA/OOD Design:
 * - Entity: PlanarSolver (manages graph state in GPU memory)
 * - Behavior: calculate_total_crossings() (parallel intersection detection)
 * - Device Functions: segments_intersect (pure integer geometry)
 * 
 * Architecture:
 * - Host (C++): Memory management, pybind11 interface
 * - Device (CUDA): Parallel crossing detection kernels
 * 
 * Memory Strategy:
 * - Device Memory: Persistent storage for node coordinates and edges
 * - Minimized Transfers: Upload once, compute on GPU, download result only
 * 
 * @version 0.2.0-cycle1
 * @date November 28, 2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace py = pybind11;

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA Error: ") + cudaGetErrorString(error) + \
                " at " + __FILE__ + ":" + std::to_string(__LINE__) \
            ); \
        } \
    } while(0)

// ============================================================================
// Device Functions: Pure Integer Geometry
// ============================================================================

/**
 * Cross product of vectors OA and OB.
 * 
 * Mathematical Foundation:
 *   cross = (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
 * 
 * Returns:
 *   > 0: Counter-clockwise turn (B is left of OA)
 *   < 0: Clockwise turn (B is right of OA)
 *   = 0: Collinear (O, A, B on same line)
 * 
 * @note __device__ qualifier means this runs on GPU only
 * @note __forceinline__ suggests compiler to inline for performance
 */
__device__ __forceinline__ long long cross_product(
    int ox, int oy,  // Origin point O
    int ax, int ay,  // Point A
    int bx, int by   // Point B
) {
    // Use long long to prevent integer overflow
    // Maximum value: 2^31 * 2^31 = 2^62 (fits in 64-bit)
    long long dx1 = (long long)(ax - ox);
    long long dy1 = (long long)(ay - oy);
    long long dx2 = (long long)(bx - ox);
    long long dy2 = (long long)(by - oy);
    
    return dx1 * dy2 - dy1 * dx2;
}

/**
 * Determine if two line segments properly intersect.
 * 
 * Algorithm:
 *   Two segments (p1-p2) and (q1-q2) intersect if and only if:
 *   1. q1 and q2 are on opposite sides of line p1-p2, AND
 *   2. p1 and p2 are on opposite sides of line q1-q2
 * 
 * Special Cases (all return false):
 *   - Shared endpoints (not a crossing)
 *   - Endpoint touching (not a crossing)
 *   - Collinear segments (even if overlapping)
 * 
 * Implementation:
 *   Use cross product signs to determine "sidedness"
 *   Opposite sides means: d1 * d2 < 0 (strictly negative)
 * 
 * @param p1x, p1y: First endpoint of segment 1
 * @param p2x, p2y: Second endpoint of segment 1
 * @param q1x, q1y: First endpoint of segment 2
 * @param q2x, q2y: Second endpoint of segment 2
 * @return true if segments properly cross, false otherwise
 */
__device__ bool segments_intersect(
    int p1x, int p1y,
    int p2x, int p2y,
    int q1x, int q1y,
    int q2x, int q2y
) {
    // Fast rejection: Check for shared endpoints
    // If segments share a vertex, they don't "cross"
    if ((p1x == q1x && p1y == q1y) || (p1x == q2x && p1y == q2y) ||
        (p2x == q1x && p2y == q1y) || (p2x == q2x && p2y == q2y)) {
        return false;
    }
    
    // Calculate cross products to determine orientations
    // For segment p1-p2 with respect to points q1, q2:
    long long d1 = cross_product(p1x, p1y, p2x, p2y, q1x, q1y);
    long long d2 = cross_product(p1x, p1y, p2x, p2y, q2x, q2y);
    
    // For segment q1-q2 with respect to points p1, p2:
    long long d3 = cross_product(q1x, q1y, q2x, q2y, p1x, p1y);
    long long d4 = cross_product(q1x, q1y, q2x, q2y, p2x, p2y);
    
    // Segments intersect if:
    // - q1 and q2 are on opposite sides of p1-p2: d1 and d2 have opposite signs
    // - p1 and p2 are on opposite sides of q1-q2: d3 and d4 have opposite signs
    //
    // CRITICAL: With 6-digit coordinates (~1M), d1*d2 can exceed long long range!
    // Example: d1 ≈ 4.5×10^10, d2 ≈ 4×10^10 → d1*d2 ≈ 1.8×10^21 > 2^63-1
    // Solution: Check signs separately instead of multiplying
    //
    // "Opposite signs" means: (d1 < 0 && d2 > 0) || (d1 > 0 && d2 < 0)
    // Equivalent to: (d1 ^ d2) < 0 (XOR of sign bits)
    // But we also need STRICT inequality (exclude 0 for collinear/touching)
    bool opposite_12 = (d1 < 0 && d2 > 0) || (d1 > 0 && d2 < 0);
    bool opposite_34 = (d3 < 0 && d4 > 0) || (d3 > 0 && d4 < 0);
    
    if (opposite_12 && opposite_34) {
        return true;
    }
    
    // All other cases (collinear, parallel, touching) are not crossings
    return false;
}

// ============================================================================
// NEW: Smart Move Generation Device Functions
// ============================================================================

/**
 * Check if a position is occupied by any node (excluding specific node).
 * Device function - runs on GPU.
 * 
 * @param nodes_x: All node x-coordinates
 * @param nodes_y: All node y-coordinates
 * @param num_nodes: Total number of nodes
 * @param exclude_node: Node ID to exclude from check
 * @param check_x: X-coordinate to check
 * @param check_y: Y-coordinate to check
 * @return true if position is occupied
 */
__device__ bool is_position_occupied(
    const int* nodes_x,
    const int* nodes_y,
    int num_nodes,
    int exclude_node,
    int check_x,
    int check_y
) {
    for (int i = 0; i < num_nodes; i++) {
        if (i != exclude_node) {
            if (nodes_x[i] == check_x && nodes_y[i] == check_y) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Check if point is on segment interior (excluding endpoints).
 * Device function - matches Python implementation.
 * 
 * @param px, py: Point to check
 * @param x1, y1: Segment start
 * @param x2, y2: Segment end
 * @return true if point is on segment interior
 */
__device__ bool point_on_segment_interior_dev(
    int px, int py,
    int x1, int y1,
    int x2, int y2
) {
    // Check if endpoint
    if ((px == x1 && py == y1) || (px == x2 && py == y2)) {
        return false;
    }
    
    // Check collinearity
    long long cross = cross_product(x1, y1, x2, y2, px, py);
    if (cross != 0) {
        return false;
    }
    
    // Check if within bounding box
    int min_x = (x1 < x2) ? x1 : x2;
    int max_x = (x1 > x2) ? x1 : x2;
    int min_y = (y1 < y2) ? y1 : y2;
    int max_y = (y1 > y2) ? y1 : y2;
    
    if (px < min_x || px > max_x) return false;
    if (py < min_y || py > max_y) return false;
    
    return true;
}

/**
 * Check if new position would create edge-through-node violation.
 * Device function - checks if any incident edge would pass through nodes.
 * 
 * @param nodes_x, nodes_y: All node coordinates
 * @param edges: All edges
 * @param num_nodes, num_edges: Counts
 * @param node_id: Node being moved
 * @param new_x, new_y: Proposed new position
 * @return true if violation detected
 */
__device__ bool check_edge_violations(
    const int* nodes_x,
    const int* nodes_y,
    const int2* edges,
    int num_nodes,
    int num_edges,
    int node_id,
    int new_x,
    int new_y
) {
    // Check incident edges of node_id
    for (int i = 0; i < num_edges; i++) {
        int src = edges[i].x;
        int tgt = edges[i].y;
        
        // Only check edges connected to node_id
        if (src != node_id && tgt != node_id) {
            continue;
        }
        
        // Get the other endpoint
        int other_node = (src == node_id) ? tgt : src;
        int other_x = nodes_x[other_node];
        int other_y = nodes_y[other_node];
        
        // Check if any other node is on this edge's new position
        for (int nid = 0; nid < num_nodes; nid++) {
            if (nid == node_id || nid == other_node) {
                continue;
            }
            
            int node_x = nodes_x[nid];
            int node_y = nodes_y[nid];
            
            if (point_on_segment_interior_dev(
                node_x, node_y,
                new_x, new_y,
                other_x, other_y
            )) {
                return true;  // Violation!
            }
        }
    }
    
    return false;  // No violation
}

/**
 * Smart move generation kernel - generates valid moves on GPU.
 * 
 * Strategy:
 * - Each thread generates a valid move for one node
 * - Checks: duplicate coordinates, edge violations
 * - Returns valid position or original position if failed
 * 
 * @param nodes_x, nodes_y: Current node coordinates
 * @param edges: Edge pairs
 * @param num_nodes, num_edges: Counts
 * @param node_id: Node to move (single node for SA)
 * @param step_size: Maximum move distance
 * @param max_retries: Maximum retry attempts
 * @param seed: Random seed
 * @param out_x, out_y: Output position
 * @param max_width, max_height: Canvas bounds
 */
__global__ void generate_smart_move_kernel(
    const int* nodes_x,
    const int* nodes_y,
    const int2* edges,
    int num_nodes,
    int num_edges,
    int node_id,
    int step_size,
    int max_retries,
    unsigned int seed,
    int* out_x,
    int* out_y,
    int max_width,
    int max_height
) {
    // Only one thread does the work (SA moves one node at a time)
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    
    // Initialize random state
    curandState state;
    curand_init(seed, 0, 0, &state);
    
    int old_x = nodes_x[node_id];
    int old_y = nodes_y[node_id];
    
    int current_radius = step_size;
    
    for (int attempt = 0; attempt < max_retries; attempt++) {
        // Generate candidate position
        int offset_x = curand(&state) % (2 * current_radius + 1) - current_radius;
        int offset_y = curand(&state) % (2 * current_radius + 1) - current_radius;
        
        int new_x = old_x + offset_x;
        int new_y = old_y + offset_y;
        
        // Clip to bounds
        if (new_x < 0) new_x = 0;
        if (new_x > max_width) new_x = max_width;
        if (new_y < 0) new_y = 0;
        if (new_y > max_height) new_y = max_height;
        
        // Check 1: Duplicate position
        if (is_position_occupied(nodes_x, nodes_y, num_nodes, node_id, new_x, new_y)) {
            current_radius = (int)(current_radius * 1.2f);
            continue;
        }
        
        // Check 2: Edge violations
        if (check_edge_violations(nodes_x, nodes_y, edges, num_nodes, num_edges,
                                 node_id, new_x, new_y)) {
            current_radius = (int)(current_radius * 1.2f);
            continue;
        }
        
        // Valid move found!
        *out_x = new_x;
        *out_y = new_y;
        return;
    }
    
    // All retries failed - return original position
    *out_x = old_x;
    *out_y = old_y;
}

// ============================================================================
// CUDA Kernel: Parallel Crossing Detection
// ============================================================================

/**
 * Kernel to count edge crossings in parallel.
 * 
 * Strategy: O(E^2) brute force (optimized with Spatial Hash in Phase 3)
 * Each thread is responsible for checking one edge against all subsequent edges.
 * 
 * Parallelization:
 *   - Thread ID (tid) maps to edge index
 *   - Each thread checks edges[tid] against edges[tid+1 ... num_edges-1]
 *   - Use atomicAdd to accumulate crossing count (thread-safe)
 * 
 * @param nodes_x: Array of node x-coordinates on device
 * @param nodes_y: Array of node y-coordinates on device
 * @param edges: Array of edge pairs (u, v) as int2 on device
 * @param num_edges: Total number of edges
 * @param crossings: Output counter (single integer on device)
 */
__global__ void count_crossings_kernel(
    const int* nodes_x,
    const int* nodes_y,
    const int2* edges,
    int num_edges,
    unsigned long long* crossings
) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: ensure thread ID is within valid range
    if (tid >= num_edges) {
        return;
    }
    
    // Get edge i (the edge this thread is responsible for)
    int2 edge_i = edges[tid];
    int u1 = edge_i.x;
    int v1 = edge_i.y;
    
    // Get coordinates of edge i's endpoints
    int p1x = nodes_x[u1];
    int p1y = nodes_y[u1];
    int p2x = nodes_x[v1];
    int p2y = nodes_y[v1];
    
    // Local counter for this thread (reduces atomic contention)
    unsigned long long local_count = 0;
    
    // Check edge i against all subsequent edges j (j > i)
    // This avoids double-counting (edge pair checked only once)
    for (int j = tid + 1; j < num_edges; j++) {
        int2 edge_j = edges[j];
        int u2 = edge_j.x;
        int v2 = edge_j.y;
        
        // Skip edges that share an endpoint (no crossing possible)
        if (u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2) {
            continue;
        }
        
        // Get coordinates of edge j's endpoints
        int q1x = nodes_x[u2];
        int q1y = nodes_y[u2];
        int q2x = nodes_x[v2];
        int q2y = nodes_y[v2];
        
        // Check if segments intersect
        if (segments_intersect(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y)) {
            local_count++;
        }
    }
    
    // Atomically add local count to global counter
    // atomicAdd is thread-safe but has performance overhead
    // Future optimization: use shared memory reduction
    if (local_count > 0) {
        atomicAdd(crossings, local_count);
    }
}

// ============================================================================
// Cycle 3.5: Spatial Hash GPU Kernel
// ============================================================================

/**
 * Cycle 3.5: Spatial hash crossing detection kernel (O(E·k) instead of O(E²))
 * 
 * Strategy:
 * - Use pre-computed cell bounds for each edge (avoid redundant computation)
 * - Only check edges with overlapping cells
 * - Dramatically reduce comparisons for sparse graphs
 * 
 * @param nodes_x: Node x-coordinates
 * @param nodes_y: Node y-coordinates
 * @param edges: Edge pairs
 * @param num_edges: Total edges
 * @param edge_cell_min_x: Pre-computed min cell x for each edge
 * @param edge_cell_max_x: Pre-computed max cell x for each edge
 * @param edge_cell_min_y: Pre-computed min cell y for each edge
 * @param edge_cell_max_y: Pre-computed max cell y for each edge
 * @param crossings: Output counter
 */
__global__ void count_crossings_spatial_kernel(
    const int* nodes_x,
    const int* nodes_y,
    const int2* edges,
    int num_edges,
    const int* edge_cell_min_x,
    const int* edge_cell_max_x,
    const int* edge_cell_min_y,
    const int* edge_cell_max_y,
    unsigned long long* crossings
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_edges) {
        return;
    }
    
    // Get edge i
    int2 edge_i = edges[tid];
    int u1 = edge_i.x;
    int v1 = edge_i.y;
    
    int p1x = nodes_x[u1];
    int p1y = nodes_y[u1];
    int p2x = nodes_x[v1];
    int p2y = nodes_y[v1];
    
    // Load pre-computed cell bounds for edge i
    int cell_min_x = edge_cell_min_x[tid];
    int cell_max_x = edge_cell_max_x[tid];
    int cell_min_y = edge_cell_min_y[tid];
    int cell_max_y = edge_cell_max_y[tid];
    
    unsigned long long local_count = 0;
    
    // Only check edges that could potentially intersect
    for (int j = tid + 1; j < num_edges; j++) {
        // Load pre-computed cell bounds for edge j
        int j_cell_min_x = edge_cell_min_x[j];
        int j_cell_max_x = edge_cell_max_x[j];
        int j_cell_min_y = edge_cell_min_y[j];
        int j_cell_max_y = edge_cell_max_y[j];
        
        // Check if bounding boxes overlap (spatial filtering)
        bool cells_overlap = !(j_cell_max_x < cell_min_x || j_cell_min_x > cell_max_x ||
                               j_cell_max_y < cell_min_y || j_cell_min_y > cell_max_y);
        
        if (cells_overlap) {
            // Only load coordinates if cells overlap
            int2 edge_j = edges[j];
            int u2 = edge_j.x;
            int v2 = edge_j.y;
            
            // Skip edges that share an endpoint (no crossing possible)
            if (u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2) {
                continue;
            }
            
            int q1x = nodes_x[u2];
            int q1y = nodes_y[u2];
            int q2x = nodes_x[v2];
            int q2y = nodes_y[v2];
            
            // Check intersection
            if (segments_intersect(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y)) {
                local_count++;
            }
        }
    }
    
    if (local_count > 0) {
        atomicAdd(crossings, local_count);
    }
}

// ============================================================================
// NEW KERNEL: K-Value Calculation (Per-Edge Crossing Counts)
// ============================================================================

/**
 * Count crossings for each edge individually (K-value computation).
 * 
 * Purpose:
 *   - Calculate how many edges each edge crosses with
 *   - Used to find K-value (maximum crossings for any single edge)
 *   - Different from total crossing count (which sums all crossings)
 * 
 * Algorithm:
 *   Each thread processes one edge:
 *   1. Compare this edge with ALL other edges
 *   2. Count intersections
 *   3. Store count in edge_crossings[thread_id]
 * 
 * Complexity: O(E²) but parallelized across E threads
 * 
 * @param nodes_x: Node x-coordinates
 * @param nodes_y: Node y-coordinates  
 * @param edges: Edge pairs
 * @param num_edges: Total number of edges
 * @param edge_crossings: Output array [num_edges] - crossings per edge
 */
__global__ void count_edge_crossings_kernel(
    const int* nodes_x,
    const int* nodes_y,
    const int2* edges,
    int num_edges,
    int* edge_crossings  // Output: crossing count for each edge
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge_idx >= num_edges) {
        return;
    }
    
    // Get endpoints of current edge
    int2 edge_i = edges[edge_idx];
    int u1 = edge_i.x;
    int v1 = edge_i.y;
    
    int p1x = nodes_x[u1];
    int p1y = nodes_y[u1];
    int p2x = nodes_x[v1];
    int p2y = nodes_y[v1];
    
    int count = 0;
    
    // Compare with ALL other edges (including both i<j and i>j)
    for (int j = 0; j < num_edges; j++) {
        if (j == edge_idx) continue;  // Skip self
        
        int2 edge_j = edges[j];
        int u2 = edge_j.x;
        int v2 = edge_j.y;
        
        // Skip edges that share an endpoint (no crossing possible)
        if (u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2) {
            continue;
        }
        
        int q1x = nodes_x[u2];
        int q1y = nodes_y[u2];
        int q2x = nodes_x[v2];
        int q2y = nodes_y[v2];
        
        // Check intersection
        if (segments_intersect(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y)) {
            count++;
        }
    }
    
    // Store crossing count for this edge
    edge_crossings[edge_idx] = count;
}

/**
 * Find maximum value in array (parallel reduction).
 * 
 * Purpose:
 *   - Find K-value = max(edge_crossings[])
 *   - Parallel reduction for O(log N) complexity
 * 
 * Algorithm:
 *   1. Load data into shared memory
 *   2. Binary reduction tree (each step halves active threads)
 *   3. Final result in result[0]
 * 
 * @param data: Input array
 * @param n: Array size
 * @param result: Output (max value)
 */
__global__ void find_max_kernel(
    const int* data,
    int n,
    int* result
) {
    extern __shared__ int sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();
    
    // Parallel reduction (find max)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicMax(result, sdata[0]);
    }
}

/**
 * Compute delta K-value for moving one node (incremental calculation).
 * 
 * Purpose:
 *   - Calculate K-value change if we move node_id to (new_x, new_y)
 *   - More efficient than recalculating full K-value
 *   - Used in SA optimization with K-value cost function
 * 
 * Algorithm:
 *   1. For each edge connected to node_id:
 *      - Calculate crossings with current position
 *      - Calculate crossings with new position
 *      - Store delta in temp array
 *   2. Find max crossing count before move
 *   3. Find max crossing count after move
 *   4. Return delta_k = k_after - k_before
 * 
 * @param nodes_x: Current node x-coordinates
 * @param nodes_y: Current node y-coordinates
 * @param edges: Edge pairs
 * @param num_edges: Total number of edges
 * @param node_id: Node being moved
 * @param new_x: New x-coordinate for node_id
 * @param new_y: New y-coordinate for node_id
 * @param edge_crossings_before: Output - crossings per edge (current state)
 * @param edge_crossings_after: Output - crossings per edge (after move)
 */
__global__ void compute_delta_k_kernel(
    const int* nodes_x,
    const int* nodes_y,
    const int2* edges,
    int num_edges,
    int node_id,
    int new_x,
    int new_y,
    int* edge_crossings_before,  // Output: current crossings per edge
    int* edge_crossings_after    // Output: crossings after move
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge_idx >= num_edges) {
        return;
    }
    
    int2 edge_i = edges[edge_idx];
    int u1 = edge_i.x;
    int v1 = edge_i.y;
    
    // Check if this edge involves the moving node
    bool edge_involves_node = (u1 == node_id || v1 == node_id);
    
    // Get current positions
    int p1x_before = nodes_x[u1];
    int p1y_before = nodes_y[u1];
    int p2x_before = nodes_x[v1];
    int p2y_before = nodes_y[v1];
    
    // Get positions after move
    int p1x_after = (u1 == node_id) ? new_x : nodes_x[u1];
    int p1y_after = (u1 == node_id) ? new_y : nodes_y[u1];
    int p2x_after = (v1 == node_id) ? new_x : nodes_x[v1];
    int p2y_after = (v1 == node_id) ? new_y : nodes_y[v1];
    
    int count_before = 0;
    int count_after = 0;
    
    // Compare with all other edges
    for (int j = 0; j < num_edges; j++) {
        if (j == edge_idx) continue;
        
        int2 edge_j = edges[j];
        int u2 = edge_j.x;
        int v2 = edge_j.y;
        
        // Skip edges that share an endpoint (no crossing possible)
        if (u1 == u2 || u1 == v2 || v1 == u2 || v1 == v2) {
            continue;
        }
        
        bool other_edge_involves_node = (u2 == node_id || v2 == node_id);
        
        // Get other edge positions (before)
        int q1x_before = nodes_x[u2];
        int q1y_before = nodes_y[u2];
        int q2x_before = nodes_x[v2];
        int q2y_before = nodes_y[v2];
        
        // Get other edge positions (after)
        int q1x_after = (u2 == node_id) ? new_x : nodes_x[u2];
        int q1y_after = (u2 == node_id) ? new_y : nodes_y[u2];
        int q2x_after = (v2 == node_id) ? new_x : nodes_x[v2];
        int q2y_after = (v2 == node_id) ? new_y : nodes_y[v2];
        
        // Only recompute if at least one edge involves the moving node
        if (edge_involves_node || other_edge_involves_node) {
            // Check intersection BEFORE move
            if (segments_intersect(p1x_before, p1y_before, p2x_before, p2y_before,
                                 q1x_before, q1y_before, q2x_before, q2y_before)) {
                count_before++;
            }
            
            // Check intersection AFTER move
            if (segments_intersect(p1x_after, p1y_after, p2x_after, p2y_after,
                                 q1x_after, q1y_after, q2x_after, q2y_after)) {
                count_after++;
            }
        } else {
            // Neither edge involves moving node - crossing status unchanged
            if (segments_intersect(p1x_before, p1y_before, p2x_before, p2y_before,
                                 q1x_before, q1y_before, q2x_before, q2y_before)) {
                count_before++;
                count_after++;  // Same as before
            }
        }
    }
    
    edge_crossings_before[edge_idx] = count_before;
    edge_crossings_after[edge_idx] = count_after;
}

// ============================================================================
// C++ Class: PlanarSolver (OOP Interface)
// ============================================================================

/**
 * Host-side class managing CUDA resources for graph optimization.
 * 
 * Design Principles (OOD):
 * - Encapsulation: Hide CUDA memory management details
 * - RAII: Resource Acquisition Is Initialization (constructor allocates, destructor frees)
 * - Single Responsibility: Manages GPU state for one graph instance
 * 
 * Memory Management:
 * - Constructor: cudaMalloc + cudaMemcpy (Host → Device)
 * - Destructor: cudaFree (automatic cleanup)
 * - Copy/Move: Disabled (prevent double-free)
 * 
 * Usage Pattern (Python):
 *   solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
 *   crossings = solver.calculate_total_crossings()
 *   del solver  # Automatic GPU memory cleanup
 */
class PlanarSolver {
private:
    // Device memory pointers (GPU VRAM)
    int* d_nodes_x;       ///< Node x-coordinates on device
    int* d_nodes_y;       ///< Node y-coordinates on device
    int2* d_edges;        ///< Edge pairs (u, v) on device
    
    // Cycle 2: Initial state backup for reset functionality
    int* d_initial_x;     ///< Initial x-coordinates (for reset)
    int* d_initial_y;     ///< Initial y-coordinates (for reset)
    
    // Cycle 3: Spatial hash for optimization
    int cell_size;        ///< Grid cell size (0 = disabled, use brute force)
    bool use_spatial_hash; ///< Whether spatial hash is enabled
    
    // Cycle 3.5: Cached bounding box and edge cell indices for performance
    int bbox_min_x, bbox_max_x;
    int bbox_min_y, bbox_max_y;
    int grid_width;
    bool bbox_cached;
    
    // Pre-computed cell indices for each edge (avoid redundant computation)
    int* d_edge_cell_min_x;
    int* d_edge_cell_max_x;
    int* d_edge_cell_min_y;
    int* d_edge_cell_max_y;
    bool edge_cells_cached;
    
    // NEW: K-value calculation (per-edge crossing counts)
    int* d_edge_crossings;  ///< Device array: crossing count for each edge
    int* d_k_value_result;  ///< Device memory for K-value result
    bool k_value_dirty;     ///< True if K-value needs recalculation
    
    // Smart move optimization control
    bool enable_violation_check;  ///< Enable duplicate/collinear checks in delta_e (disable when using smart moves)
    
    // Graph dimensions
    int num_nodes;        ///< Number of nodes in graph
    int num_edges;        ///< Number of edges in graph
    
    // Coordinate constraints
    int max_width;        ///< Maximum x-coordinate (default: 1000000)
    int max_height;       ///< Maximum y-coordinate (default: 1000000)
    
    /**
     * Compute automatic cell size based on graph bounds.
     * 
     * Strategy: Divide space into ~sqrt(E) cells
     * - For 100 edges: ~10x10 grid
     * - For 400 edges: ~20x20 grid
     * 
     * @param nodes_x: Node x-coordinates
     * @param nodes_y: Node y-coordinates
     * @return Optimal cell size
     */
    int compute_auto_cell_size(const std::vector<int>& nodes_x, const std::vector<int>& nodes_y) {
        if (nodes_x.empty()) return 100;  // Default fallback
        
        // Find bounding box
        int min_x = *std::min_element(nodes_x.begin(), nodes_x.end());
        int max_x = *std::max_element(nodes_x.begin(), nodes_x.end());
        int min_y = *std::min_element(nodes_y.begin(), nodes_y.end());
        int max_y = *std::max_element(nodes_y.begin(), nodes_y.end());
        
        int width = max_x - min_x;
        int height = max_y - min_y;
        int max_dim = std::max(width, height);
        
        // Target: sqrt(num_edges) cells per dimension
        int target_cells = static_cast<int>(std::sqrt(static_cast<double>(num_edges))) + 1;
        target_cells = std::max(target_cells, 1);
        
        int auto_cell_size = max_dim / target_cells;
        auto_cell_size = std::max(auto_cell_size, 1);  // At least 1
        
        return auto_cell_size;
    }
    
public:
    /**
     * Constructor: Initialize GPU memory with graph data.
     * 
     * OOA Entity Construction:
     * - Allocates device memory for node coordinates and edges
     * - Copies data from host (Python) to device (GPU)
     * - Cycle 3: Optionally enables spatial hash for acceleration
     * 
     * @param nodes_x: Vector of node x-coordinates
     * @param nodes_y: Vector of node y-coordinates
     * @param edges: Vector of edge pairs (source, target)
     * @param cell_size: Spatial hash cell size (0 = auto, negative = disable)
     * @throws std::runtime_error if CUDA operations fail
     */
    PlanarSolver(
        const std::vector<int>& nodes_x,
        const std::vector<int>& nodes_y,
        const std::vector<std::pair<int, int>>& edges,
        int cell_size = -1,  // Default: auto-compute or disable
        int width = 1000000,  // Maximum x-coordinate
        int height = 1000000  // Maximum y-coordinate
    ) : d_nodes_x(nullptr), d_nodes_y(nullptr), d_edges(nullptr),
        d_initial_x(nullptr), d_initial_y(nullptr),
        d_edge_cell_min_x(nullptr), d_edge_cell_max_x(nullptr),
        d_edge_cell_min_y(nullptr), d_edge_cell_max_y(nullptr),
        d_edge_crossings(nullptr), d_k_value_result(nullptr),  // NEW
        num_nodes(nodes_x.size()), num_edges(edges.size()),
        bbox_cached(false), edge_cells_cached(false), k_value_dirty(true),  // NEW
        enable_violation_check(true),  // Default: enable violation checks
        max_width(width), max_height(height)
    {
        // Cycle 3: Configure spatial hash
        if (cell_size == 0) {
            // Auto-compute cell size based on graph bounds
            this->use_spatial_hash = true;
            this->cell_size = compute_auto_cell_size(nodes_x, nodes_y);
        } else if (cell_size > 0) {
            // User-specified cell size
            this->use_spatial_hash = true;
            this->cell_size = cell_size;
        } else {
            // Disabled (brute force)
            this->use_spatial_hash = false;
            this->cell_size = 0;
        }
        
        // Validate input dimensions
        if (nodes_x.size() != nodes_y.size()) {
            throw std::invalid_argument("nodes_x and nodes_y must have same size");
        }
        
        // Handle empty graph case (no allocation needed)
        if (num_nodes == 0 || num_edges == 0) {
            return;
        }
        
        // Cycle 3.5: Compute and cache bounding box if using spatial hash
        if (use_spatial_hash && cell_size > 0) {
            bbox_min_x = *std::min_element(nodes_x.begin(), nodes_x.end());
            bbox_max_x = *std::max_element(nodes_x.begin(), nodes_x.end());
            bbox_min_y = *std::min_element(nodes_y.begin(), nodes_y.end());
            bbox_max_y = *std::max_element(nodes_y.begin(), nodes_y.end());
            grid_width = (bbox_max_x - bbox_min_x) / cell_size + 1;
            bbox_cached = true;
        }
        
        // ====================================================================
        // Step 1: Allocate device memory
        // ====================================================================
        
        CUDA_CHECK(cudaMalloc(&d_nodes_x, num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nodes_y, num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_edges, num_edges * sizeof(int2)));
        
        // Cycle 2: Allocate backup memory for reset functionality
        CUDA_CHECK(cudaMalloc(&d_initial_x, num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_initial_y, num_nodes * sizeof(int)));
        
        // ====================================================================
        // Step 2: Copy node coordinates to device
        // ====================================================================
        
        CUDA_CHECK(cudaMemcpy(
            d_nodes_x,
            nodes_x.data(),
            num_nodes * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        
        CUDA_CHECK(cudaMemcpy(
            d_nodes_y,
            nodes_y.data(),
            num_nodes * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        
        // Cycle 2: Backup initial state
        CUDA_CHECK(cudaMemcpy(
            d_initial_x,
            nodes_x.data(),
            num_nodes * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        
        CUDA_CHECK(cudaMemcpy(
            d_initial_y,
            nodes_y.data(),
            num_nodes * sizeof(int),
            cudaMemcpyHostToDevice
        ));
        
        // ====================================================================
        // Step 3: Convert edge pairs to int2 and copy to device
        // ====================================================================
        
        std::vector<int2> edge_pairs(num_edges);
        for (int i = 0; i < num_edges; i++) {
            edge_pairs[i].x = edges[i].first;   // source node
            edge_pairs[i].y = edges[i].second;  // target node
        }
        
        CUDA_CHECK(cudaMemcpy(
            d_edges,
            edge_pairs.data(),
            num_edges * sizeof(int2),
            cudaMemcpyHostToDevice
        ));
        
        // NEW: Allocate K-value calculation buffers
        CUDA_CHECK(cudaMalloc(&d_edge_crossings, num_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_k_value_result, sizeof(int)));
        k_value_dirty = true;  // Needs initial calculation
        
        // Cycle 3.5: Pre-compute edge cell indices if using spatial hash
        if (use_spatial_hash && bbox_cached) {
            // Allocate device memory for cell indices
            CUDA_CHECK(cudaMalloc(&d_edge_cell_min_x, num_edges * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_edge_cell_max_x, num_edges * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_edge_cell_min_y, num_edges * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_edge_cell_max_y, num_edges * sizeof(int)));
            
            // Compute on host (avoid GPU kernel overhead for one-time computation)
            std::vector<int> cell_min_x(num_edges);
            std::vector<int> cell_max_x(num_edges);
            std::vector<int> cell_min_y(num_edges);
            std::vector<int> cell_max_y(num_edges);
            
            for (int i = 0; i < num_edges; i++) {
                int u = edge_pairs[i].x;
                int v = edge_pairs[i].y;
                
                int p1x = nodes_x[u];
                int p1y = nodes_y[u];
                int p2x = nodes_x[v];
                int p2y = nodes_y[v];
                
                // Compute bounding box
                int min_px = std::min(p1x, p2x);
                int max_px = std::max(p1x, p2x);
                int min_py = std::min(p1y, p2y);
                int max_py = std::max(p1y, p2y);
                
                // Convert to cell indices
                int cmin_x = (min_px - bbox_min_x) / cell_size;
                int cmax_x = (max_px - bbox_min_x) / cell_size;
                int cmin_y = (min_py - bbox_min_y) / cell_size;
                int cmax_y = (max_py - bbox_min_y) / cell_size;
                
                // Expand by 1 cell to catch neighboring edges
                cell_min_x[i] = std::max(0, cmin_x - 1);
                cell_max_x[i] = cmax_x + 1;
                cell_min_y[i] = std::max(0, cmin_y - 1);
                cell_max_y[i] = cmax_y + 1;
            }
            
            // Copy to device
            CUDA_CHECK(cudaMemcpy(d_edge_cell_min_x, cell_min_x.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_edge_cell_max_x, cell_max_x.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_edge_cell_min_y, cell_min_y.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_edge_cell_max_y, cell_max_y.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
            
            edge_cells_cached = true;
        }
    }
    
    /**
     * Destructor: Free GPU memory (RAII principle).
     * 
     * Automatically called when Python object is deleted.
     * Prevents memory leaks by ensuring cudaFree is always called.
     */
    ~PlanarSolver() {
        if (d_nodes_x) cudaFree(d_nodes_x);
        if (d_nodes_y) cudaFree(d_nodes_y);
        if (d_edges) cudaFree(d_edges);
        if (d_initial_x) cudaFree(d_initial_x);
        if (d_initial_y) cudaFree(d_initial_y);
        if (d_edge_cell_min_x) cudaFree(d_edge_cell_min_x);
        if (d_edge_cell_max_x) cudaFree(d_edge_cell_max_x);
        if (d_edge_cell_min_y) cudaFree(d_edge_cell_min_y);
        if (d_edge_cell_max_y) cudaFree(d_edge_cell_max_y);
        // NEW: Free K-value buffers
        if (d_edge_crossings) cudaFree(d_edge_crossings);
        if (d_k_value_result) cudaFree(d_k_value_result);
    }
    
    // Disable copy and move (prevent double-free)
    PlanarSolver(const PlanarSolver&) = delete;
    PlanarSolver& operator=(const PlanarSolver&) = delete;
    PlanarSolver(PlanarSolver&&) = delete;
    PlanarSolver& operator=(PlanarSolver&&) = delete;
    
    /**
     * Calculate total number of edge crossings.
     * 
     * Cycle 3.5: Uses spatial hash kernel if enabled, otherwise brute force
     * 
     * Algorithm:
     *   1. Allocate device counter (initialized to 0)
     *   2. Launch appropriate kernel (spatial hash or brute force)
     *   3. Each thread checks its edge against relevant edges
     *   4. Atomic accumulation of crossing count
     *   5. Copy result back to host
     * 
     * @return Total number of edge crossings
     * @throws std::runtime_error if CUDA operations fail
     */
    long long calculate_total_crossings() {
        // Handle empty graph
        if (num_edges == 0) {
            return 0;
        }
        
        // ====================================================================
        // Step 1: Allocate and initialize device counter
        // ====================================================================
        
        unsigned long long* d_crossings;
        CUDA_CHECK(cudaMalloc(&d_crossings, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_crossings, 0, sizeof(unsigned long long)));
        
        // ====================================================================
        // Step 2: Configure kernel launch parameters
        // ====================================================================
        
        // Use 256 threads per block (common choice, multiple of warp size 32)
        int threads_per_block = 256;
        
        // Calculate number of blocks needed to cover all edges
        // Use ceiling division: (num_edges + 255) / 256
        int num_blocks = (num_edges + threads_per_block - 1) / threads_per_block;
        
        // ====================================================================
        // Step 3: Launch kernel (spatial hash or brute force)
        // ====================================================================
        
        if (use_spatial_hash && cell_size > 0 && edge_cells_cached) {
            // Cycle 3.5: Use spatial hash kernel with pre-computed cell indices
            count_crossings_spatial_kernel<<<num_blocks, threads_per_block>>>(
                d_nodes_x,
                d_nodes_y,
                d_edges,
                num_edges,
                d_edge_cell_min_x,
                d_edge_cell_max_x,
                d_edge_cell_min_y,
                d_edge_cell_max_y,
                d_crossings
            );
        } else {
            // Original brute force kernel
            count_crossings_kernel<<<num_blocks, threads_per_block>>>(
                d_nodes_x,
                d_nodes_y,
                d_edges,
                num_edges,
                d_crossings
            );
        }
        
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        
        // Wait for kernel to complete
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // ====================================================================
        // Step 4: Copy result back to host
        // ====================================================================
        
        unsigned long long h_crossings;
        CUDA_CHECK(cudaMemcpy(
            &h_crossings,
            d_crossings,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost
        ));
        
        // ====================================================================
        // Step 5: Cleanup temporary device memory
        // ====================================================================
        
        CUDA_CHECK(cudaFree(d_crossings));
        
        return h_crossings;
    }
    
    /**
     * Get current node coordinates (for future use in optimization cycles).
     * 
     * @return Pair of vectors (nodes_x, nodes_y)
     */
    std::pair<std::vector<int>, std::vector<int>> get_coordinates() {
        std::vector<int> nodes_x(num_nodes);
        std::vector<int> nodes_y(num_nodes);
        
        if (num_nodes > 0) {
            CUDA_CHECK(cudaMemcpy(
                nodes_x.data(),
                d_nodes_x,
                num_nodes * sizeof(int),
                cudaMemcpyDeviceToHost
            ));
            
            CUDA_CHECK(cudaMemcpy(
                nodes_y.data(),
                d_nodes_y,
                num_nodes * sizeof(int),
                cudaMemcpyDeviceToHost
            ));
        }
        
        return {nodes_x, nodes_y};
    }
    
    // ========================================================================
    // Cycle 2: State Management Methods
    // ========================================================================
    
    /**
     * Check if two line segments are collinear and overlap (share a line segment).
     * This is a VIOLATION - edges cannot share line segments.
     * 
     * @param p1x, p1y: Start of segment 1
     * @param p2x, p2y: End of segment 1
     * @param p3x, p3y: Start of segment 2
     * @param p4x, p4y: End of segment 2
     * @return true if segments are collinear and overlap
     */
    bool segments_share_line(int p1x, int p1y, int p2x, int p2y,
                             int p3x, int p3y, int p4x, int p4y) {
        // Check if all points are collinear using cross product
        long long cross1 = (long long)(p2x - p1x) * (p3y - p1y) - (long long)(p2y - p1y) * (p3x - p1x);
        long long cross2 = (long long)(p2x - p1x) * (p4y - p1y) - (long long)(p2y - p1y) * (p4x - p1x);
        
        if (cross1 != 0 || cross2 != 0) {
            return false; // Not collinear
        }
        
        // All points are collinear, check for overlap
        // Project onto the dominant axis
        int dx = std::abs(p2x - p1x);
        int dy = std::abs(p2y - p1y);
        
        int seg1_min, seg1_max, seg2_min, seg2_max;
        
        if (dx >= dy) {
            // Use X axis
            seg1_min = std::min(p1x, p2x);
            seg1_max = std::max(p1x, p2x);
            seg2_min = std::min(p3x, p4x);
            seg2_max = std::max(p3x, p4x);
        } else {
            // Use Y axis
            seg1_min = std::min(p1y, p2y);
            seg1_max = std::max(p1y, p2y);
            seg2_min = std::min(p3y, p4y);
            seg2_max = std::max(p3y, p4y);
        }
        
        // Check if ranges overlap (not just touch at endpoints)
        int overlap_start = std::max(seg1_min, seg2_min);
        int overlap_end = std::min(seg1_max, seg2_max);
        
        // True overlap if overlap_start < overlap_end
        return overlap_start < overlap_end;
    }
    
    /**
     * Check if a point lies on a line segment (excluding endpoints).
     * This is a VIOLATION - edges cannot pass through non-endpoint nodes.
     * 
     * @param px, py: Point to check
     * @param x1, y1: Segment start
     * @param x2, y2: Segment end
     * @return true if point is on segment (not at endpoints)
     */
    bool point_on_segment_interior(int px, int py, int x1, int y1, int x2, int y2) {
        // Check if it's an endpoint
        if ((px == x1 && py == y1) || (px == x2 && py == y2)) {
            return false;
        }
        
        // Check if collinear using cross product
        long long cross = (long long)(x2 - x1) * (py - y1) - (long long)(y2 - y1) * (px - x1);
        if (cross != 0) {
            return false;
        }
        
        // Check if within bounding box
        if (px < std::min(x1, x2) || px > std::max(x1, x2)) return false;
        if (py < std::min(y1, y2) || py > std::max(y1, y2)) return false;
        
        return true;
    }
    
    /**
     * Count violations where edges pass through non-endpoint nodes.
     * Used to add penalty in compute_delta_e.
     * 
     * This checks TWO scenarios:
     * 1. Edges connected to node_id passing through other nodes
     * 2. Other edges passing through node_id's NEW position
     * 
     * @param node_id: The node that will be moved
     * @param all_x, all_y: All node coordinates (with node_id at NEW position)
     * @return Number of violations
     */
    int count_edge_through_node_violations(int node_id,
                                          const std::vector<int>& all_x,
                                          const std::vector<int>& all_y) {
        if (num_edges == 0) return 0;
        
        // Copy edges from device to host
        std::vector<int2> edges_data(num_edges);
        CUDA_CHECK(cudaMemcpy(
            edges_data.data(),
            d_edges,
            num_edges * sizeof(int2),
            cudaMemcpyDeviceToHost
        ));
        
        int violations = 0;
        
        // Scenario 1: Check edges connected to node_id passing through other nodes
        for (int i = 0; i < num_edges; i++) {
            int src = edges_data[i].x;
            int tgt = edges_data[i].y;
            
            // Only check edges connected to node_id
            if (src != node_id && tgt != node_id) {
                continue;
            }
            
            int x1 = all_x[src], y1 = all_y[src];
            int x2 = all_x[tgt], y2 = all_y[tgt];
            
            // Check if this edge passes through any other node
            for (int nid = 0; nid < num_nodes; nid++) {
                if (nid == src || nid == tgt) continue;
                
                int px = all_x[nid], py = all_y[nid];
                
                if (point_on_segment_interior(px, py, x1, y1, x2, y2)) {
                    violations++;
                }
            }
        }
        
        // Scenario 2: Check if OTHER edges pass through node_id's NEW position
        int new_x = all_x[node_id];
        int new_y = all_y[node_id];
        
        for (int i = 0; i < num_edges; i++) {
            int src = edges_data[i].x;
            int tgt = edges_data[i].y;
            
            // Skip edges connected to node_id (already handled in Scenario 1)
            if (src == node_id || tgt == node_id) {
                continue;
            }
            
            int x1 = all_x[src], y1 = all_y[src];
            int x2 = all_x[tgt], y2 = all_y[tgt];
            
            // Check if node_id's NEW position is on this edge
            if (point_on_segment_interior(new_x, new_y, x1, y1, x2, y2)) {
                violations++;
            }
        }
        
        return violations;
    }
    
    /**
     * Check if three or more nodes are collinear (on same line).
     * This is a CRITICAL constraint: no three nodes can be on the same line.
     * 
     * IMPORTANT: This function checks if moving node_id would create ANY collinear triple,
     * including with existing nodes that are already placed.
     * 
     * @param node_id: The node that will be moved
     * @param all_x, all_y: All node coordinates (with node_id at NEW position)
     * @return Number of violations (number of collinear triples involving node_id)
     */
    int count_collinear_nodes_violations(int node_id,
                                         const std::vector<int>& all_x,
                                         const std::vector<int>& all_y) {
        if (num_nodes < 3) return 0;
        
        int violations = 0;
        int nx = all_x[node_id];
        int ny = all_y[node_id];
        
        // Strategy 1: Check all pairs of other nodes to see if node_id forms a collinear triple
        for (int i = 0; i < num_nodes; i++) {
            if (i == node_id) continue;
            
            for (int j = i + 1; j < num_nodes; j++) {
                if (j == node_id) continue;
                
                int ix = all_x[i], iy = all_y[i];
                int jx = all_x[j], jy = all_y[j];
                
                // Check if node_id, i, j are collinear using cross product
                // Vector (i -> node_id) × (i -> j) should be non-zero
                long long cross = (long long)(nx - ix) * (jy - iy) - (long long)(ny - iy) * (jx - ix);
                
                if (cross == 0) {
                    // Three points are collinear - VIOLATION!
                    violations++;
                }
            }
        }
        
        // Strategy 2: CRITICAL FIX - Also check if moving node_id to same x or y 
        // as TWO or more existing nodes (creates vertical/horizontal collinearity)
        // This catches cases like: nodes at (100, 10), (100, 20) exist, and we try to move to (100, 30)
        
        int same_x_count = 0;
        int same_y_count = 0;
        
        for (int i = 0; i < num_nodes; i++) {
            if (i == node_id) continue;
            if (all_x[i] == nx) same_x_count++;
            if (all_y[i] == ny) same_y_count++;
        }
        
        // If 2+ nodes already have the same x coordinate, moving here creates vertical collinearity
        if (same_x_count >= 2) {
            violations += same_x_count * (same_x_count - 1) / 2;  // Number of pairs
        }
        
        // If 2+ nodes already have the same y coordinate, moving here creates horizontal collinearity
        if (same_y_count >= 2) {
            violations += same_y_count * (same_y_count - 1) / 2;  // Number of pairs
        }
        
        return violations;
    }
    
    /**
     * Count collinear edge violations for edges connected to a specific node.
     * Used to add penalty in compute_delta_e.
     * 
     * @param node_id: The node that will be moved
     * @param all_x, all_y: All node coordinates (with node_id at NEW position)
     * @return Number of collinear violations involving edges connected to node_id
     */
    int count_collinear_violations(int node_id, 
                                   const std::vector<int>& all_x,
                                   const std::vector<int>& all_y) {
        if (num_edges == 0) return 0;
        
        // Copy edges from device to host
        std::vector<int2> edges_data(num_edges);
        CUDA_CHECK(cudaMemcpy(
            edges_data.data(),
            d_edges,
            num_edges * sizeof(int2),
            cudaMemcpyDeviceToHost
        ));
        
        int violations = 0;
        
        // Get edges connected to node_id
        std::vector<std::pair<int, int>> my_edges;
        for (int i = 0; i < num_edges; i++) {
            int src = edges_data[i].x;
            int tgt = edges_data[i].y;
            if (src == node_id || tgt == node_id) {
                my_edges.push_back({src, tgt});
            }
        }
        
        // Check each of my edges against all other edges
        for (const auto& my_edge : my_edges) {
            int s1 = my_edge.first;
            int t1 = my_edge.second;
            int p1x = all_x[s1], p1y = all_y[s1];
            int p2x = all_x[t1], p2y = all_y[t1];
            
            for (int i = 0; i < num_edges; i++) {
                int s2 = edges_data[i].x;
                int t2 = edges_data[i].y;
                
                // Skip if it's the same edge
                if ((s1 == s2 && t1 == t2) || (s1 == t2 && t1 == s2)) {
                    continue;
                }
                
                // Skip if edges share endpoints (this is allowed)
                int shared = 0;
                if (s1 == s2 || s1 == t2) shared++;
                if (t1 == s2 || t1 == t2) shared++;
                if (shared > 0) {
                    continue;
                }
                
                int p3x = all_x[s2], p3y = all_y[s2];
                int p4x = all_x[t2], p4y = all_y[t2];
                
                if (segments_share_line(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y)) {
                    violations++;
                }
            }
        }
        
        return violations;
    }
    
    /**
     * Update a single node's position in GPU memory.
     * 
     * OOA State Modification:
     * - Directly modifies GPU-resident coordinates
     * - Enables incremental updates without full data transfer
     * 
     * @param node_id: Index of node to update (0-based)
     * @param new_x: New x-coordinate
     * @param new_y: New y-coordinate
     * @throws std::runtime_error if node_id is invalid
     */
    void update_node_position(int node_id, int new_x, int new_y) {
        // Validate node ID
        if (node_id < 0 || node_id >= num_nodes) {
            throw std::out_of_range(
                "Node ID " + std::to_string(node_id) + 
                " out of range [0, " + std::to_string(num_nodes) + ")"
            );
        }
        
        // Update x-coordinate
        CUDA_CHECK(cudaMemcpy(
            d_nodes_x + node_id,
            &new_x,
            sizeof(int),
            cudaMemcpyHostToDevice
        ));
        
        // Update y-coordinate
        CUDA_CHECK(cudaMemcpy(
            d_nodes_y + node_id,
            &new_y,
            sizeof(int),
            cudaMemcpyHostToDevice
        ));
        
        // NEW: Mark K-value as dirty after position change
        k_value_dirty = true;
    }
    
    /**
     * Get current position of a node.
     * 
     * @param node_id: Index of node to query
     * @return Pair (x, y) of node coordinates
     * @throws std::runtime_error if node_id is invalid
     */
    std::pair<int, int> get_node_position(int node_id) {
        if (node_id < 0 || node_id >= num_nodes) {
            throw std::out_of_range(
                "Node ID " + std::to_string(node_id) + 
                " out of range [0, " + std::to_string(num_nodes) + ")"
            );
        }
        
        int x, y;
        
        CUDA_CHECK(cudaMemcpy(
            &x,
            d_nodes_x + node_id,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));
        
        CUDA_CHECK(cudaMemcpy(
            &y,
            d_nodes_y + node_id,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));
        
        return {x, y};
    }
    
    // ========================================================================
    // NEW: K-Value Calculation Methods
    // ========================================================================
    
    /**
     * Calculate K-value: maximum crossing count among all edges.
     * 
     * Purpose:
     *   - The competition metric is K-value, NOT total crossings
     *   - K = max(crossings per edge)
     *   - Lower K is better
     * 
     * Algorithm:
     *   1. Launch GPU kernel to count crossings for each edge
     *   2. Find maximum value using parallel reduction
     *   3. Return K-value
     * 
     * Complexity: O(E²) parallelized + O(E) reduction
     * 
     * @return K-value (maximum crossing count on any single edge)
     */
    int calculate_k_value() {
        if (num_edges == 0) return 0;
        
        // Step 1: Count crossings for each edge
        dim3 block(256);
        dim3 grid((num_edges + 255) / 256);
        
        count_edge_crossings_kernel<<<grid, block>>>(
            d_nodes_x,
            d_nodes_y,
            d_edges,
            num_edges,
            d_edge_crossings
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 2: Find maximum value
        int h_k_value = 0;
        CUDA_CHECK(cudaMemcpy(
            d_k_value_result,
            &h_k_value,
            sizeof(int),
            cudaMemcpyHostToDevice
        ));
        
        int shared_mem_size = block.x * sizeof(int);
        find_max_kernel<<<grid, block, shared_mem_size>>>(
            d_edge_crossings,
            num_edges,
            d_k_value_result
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Step 3: Download result
        CUDA_CHECK(cudaMemcpy(
            &h_k_value,
            d_k_value_result,
            sizeof(int),
            cudaMemcpyDeviceToHost
        ));
        
        k_value_dirty = false;
        return h_k_value;
    }
    
    /**
     * Get crossing counts for all edges.
     * 
     * Purpose:
     *   - Detailed analysis: which edges have high crossings?
     *   - Debugging and visualization
     * 
     * @return Vector of crossing counts (one per edge)
     */
    std::vector<int> get_edge_crossings() {
        if (num_edges == 0) return {};
        
        // Compute if needed
        if (k_value_dirty) {
            calculate_k_value();
        }
        
        // Download edge crossings array
        std::vector<int> h_edge_crossings(num_edges);
        CUDA_CHECK(cudaMemcpy(
            h_edge_crossings.data(),
            d_edge_crossings,
            num_edges * sizeof(int),
            cudaMemcpyDeviceToHost
        ));
        
        return h_edge_crossings;
    }
    
    /**
     * Compute delta K-value for moving one node (incremental calculation).
     * 
     * Purpose:
     *   - Calculate change in K-value if we move node_id to (new_x, new_y)
     *   - More efficient than full K-value recalculation
     *   - Used in SA optimization with K-value cost function
     * 
     * Algorithm:
     *   1. Calculate edge crossings before and after move (GPU parallel)
     *   2. Find K-value before and after
     *   3. Return delta_k = k_after - k_before
     * 
     * Complexity: O(E²) parallelized, but only for affected edges
     * 
     * @param node_id: Node to move
     * @param new_x: New x-coordinate
     * @param new_y: New y-coordinate
     * @return Change in K-value (can be negative, zero, or positive)
     */
    int compute_delta_k(int node_id, int new_x, int new_y) {
        if (num_edges == 0) return 0;
        
        // Allocate temporary buffers for before/after edge crossings
        int* d_crossings_before = nullptr;
        int* d_crossings_after = nullptr;
        CUDA_CHECK(cudaMalloc(&d_crossings_before, num_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_crossings_after, num_edges * sizeof(int)));
        
        // Launch kernel to compute edge crossings before and after move
        dim3 block(256);
        dim3 grid((num_edges + 255) / 256);
        
        compute_delta_k_kernel<<<grid, block>>>(
            d_nodes_x,
            d_nodes_y,
            d_edges,
            num_edges,
            node_id,
            new_x,
            new_y,
            d_crossings_before,
            d_crossings_after
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Find max value before move
        int* d_k_before = nullptr;
        int* d_k_after = nullptr;
        CUDA_CHECK(cudaMalloc(&d_k_before, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_k_after, sizeof(int)));
        
        int h_zero = 0;
        CUDA_CHECK(cudaMemcpy(d_k_before, &h_zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_after, &h_zero, sizeof(int), cudaMemcpyHostToDevice));
        
        int shared_mem_size = block.x * sizeof(int);
        find_max_kernel<<<grid, block, shared_mem_size>>>(
            d_crossings_before,
            num_edges,
            d_k_before
        );
        
        find_max_kernel<<<grid, block, shared_mem_size>>>(
            d_crossings_after,
            num_edges,
            d_k_after
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Download results
        int h_k_before, h_k_after;
        CUDA_CHECK(cudaMemcpy(&h_k_before, d_k_before, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_k_after, d_k_after, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Cleanup temporary buffers
        CUDA_CHECK(cudaFree(d_crossings_before));
        CUDA_CHECK(cudaFree(d_crossings_after));
        CUDA_CHECK(cudaFree(d_k_before));
        CUDA_CHECK(cudaFree(d_k_after));
        
        int delta_k = h_k_after - h_k_before;
        return delta_k;
    }
    
    /**
     * Compute bottleneck penalty cost (sum of X(e)^p).
     * 
     * Purpose:
     *   - Calculate weighted sum of edge crossings
     *   - Higher power p gives more penalty to high-crossing edges
     *   - p=2: quadratic penalty, p=3: cubic penalty
     * 
     * Algorithm:
     *   - Get crossing count for each edge
     *   - Compute sum of (count^p) for all edges
     * 
     * @param power: Exponent p (typically 2 or 3)
     * @return Sum of X(e)^p across all edges
     */
    long long compute_bottleneck_cost(int power) {
        if (num_edges == 0) return 0;
        
        // Get per-edge crossing counts
        auto edge_crossings = get_edge_crossings();
        
        // Compute sum of X(e)^p
        long long total_cost = 0;
        for (int count : edge_crossings) {
            long long powered = 1;
            for (int i = 0; i < power; i++) {
                powered *= count;
            }
            total_cost += powered;
        }
        
        return total_cost;
    }
    
    /**
     * Compute delta bottleneck cost for moving one node.
     * 
     * Purpose:
     *   - Calculate change in sum(X(e)^p) if we move node_id
     *   - More efficient than full recalculation
     * 
     * @param node_id: Node to move
     * @param new_x: New x-coordinate
     * @param new_y: New y-coordinate
     * @param power: Exponent p (2 or 3)
     * @return Change in bottleneck cost
     */
    long long compute_delta_bottleneck(int node_id, int new_x, int new_y, int power) {
        if (num_edges == 0) return 0;
        
        // CRITICAL: Check geometry constraints BEFORE computing cost
        // Temporarily update coordinates for constraint checking
        std::vector<int> all_x(num_nodes);
        std::vector<int> all_y(num_nodes);
        CUDA_CHECK(cudaMemcpy(all_x.data(), d_nodes_x, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(all_y.data(), d_nodes_y, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        
        int orig_x = all_x[node_id];
        int orig_y = all_y[node_id];
        
        // STRATEGY: Penalty-based constraint enforcement
        // - Add penalty for violations (not absolute rejection)
        // - Allows SA to explore through violations at high temperature
        // - Penalty: 10 billion per violation (discourages but doesn't block)
        // - Can be DISABLED when using smart move generator (which already prevents violations)
        
        long long penalty_change = 0;
        
        if (enable_violation_check) {
            // Count violations in ORIGINAL position
            int violations_before = count_collinear_nodes_violations(node_id, all_x, all_y);
            violations_before += count_collinear_violations(node_id, all_x, all_y);
            violations_before += count_edge_through_node_violations(node_id, all_x, all_y);
            
            // Update to NEW position for checking
            all_x[node_id] = new_x;
            all_y[node_id] = new_y;
            
            // Count violations in NEW position
            int violations_after = count_collinear_nodes_violations(node_id, all_x, all_y);
            violations_after += count_collinear_violations(node_id, all_x, all_y);
            violations_after += count_edge_through_node_violations(node_id, all_x, all_y);
            
            // Calculate violation penalty change
            // Penalty: 10 billion per violation (allows exploration but discourages violations)
            penalty_change = static_cast<long long>(violations_after - violations_before) * 10000000000LL;
        }
        
        // Note: penalty_change will be ADDED to crossing cost later
        // - If violations increase: positive penalty (discourage)
        // - If violations decrease: negative penalty (reward)
        // - SA can still accept violations at high temperature
        
        // Use compute_delta_k kernel to get before/after crossings
        int* d_crossings_before = nullptr;
        int* d_crossings_after = nullptr;
        CUDA_CHECK(cudaMalloc(&d_crossings_before, num_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_crossings_after, num_edges * sizeof(int)));
        
        dim3 block(256);
        dim3 grid((num_edges + 255) / 256);
        
        compute_delta_k_kernel<<<grid, block>>>(
            d_nodes_x,
            d_nodes_y,
            d_edges,
            num_edges,
            node_id,
            new_x,
            new_y,
            d_crossings_before,
            d_crossings_after
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Download results
        std::vector<int> h_before(num_edges);
        std::vector<int> h_after(num_edges);
        CUDA_CHECK(cudaMemcpy(h_before.data(), d_crossings_before, num_edges * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_after.data(), d_crossings_after, num_edges * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Compute bottleneck cost before and after
        long long cost_before = 0;
        long long cost_after = 0;
        
        for (int i = 0; i < num_edges; i++) {
            long long powered_before = 1;
            long long powered_after = 1;
            
            for (int p = 0; p < power; p++) {
                powered_before *= h_before[i];
                powered_after *= h_after[i];
            }
            
            cost_before += powered_before;
            cost_after += powered_after;
        }
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_crossings_before));
        CUDA_CHECK(cudaFree(d_crossings_after));
        
        return cost_after - cost_before + penalty_change;
    }
    
    /**
     * Mark K-value as dirty (needs recalculation).
     * 
     * Called internally after node movements.
     */
    void mark_k_value_dirty() {
        k_value_dirty = true;
    }
    
    /**
     * Compute delta_e (change in crossings) for a hypothetical move.
     * 
     * OOA Analysis:
     * - Temporarily modifies state
     * - Computes energy difference
     * - Restores original state
     * 
     * This is critical for Simulated Annealing:
     *   - Accept move if delta_E < 0 (improvement)
     *   - Accept move with probability exp(-delta_E/T) if delta_E > 0
     * 
     * @param node_id: Node to hypothetically move
     * @param new_x: Hypothetical new x-coordinate
     * @param new_y: Hypothetical new y-coordinate
     * @return Change in crossing count (can be negative, zero, or positive)
     */
    long long compute_delta_e(int node_id, int new_x, int new_y) {
        if (node_id < 0 || node_id >= num_nodes) {
            throw std::out_of_range(
                "Node ID " + std::to_string(node_id) + 
                " out of range [0, " + std::to_string(num_nodes) + ")"
            );
        }
        
        // Step 0: Check for duplicate coordinates (CRITICAL CONSTRAINT)
        // If new position overlaps with ANY other node, apply massive penalty
        // This prevents violations where multiple nodes share the same coordinate
        // Can be DISABLED when using smart move generator (which already prevents duplicates)
        
        if (enable_violation_check) {
            std::vector<int> all_x(num_nodes);
            std::vector<int> all_y(num_nodes);
            CUDA_CHECK(cudaMemcpy(all_x.data(), d_nodes_x, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(all_y.data(), d_nodes_y, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
            
            for (int i = 0; i < num_nodes; i++) {
                if (i != node_id && all_x[i] == new_x && all_y[i] == new_y) {
                    // VIOLATION: Duplicate coordinate detected!
                    // Return huge penalty to prevent SA from accepting this move
                    // Penalty = 1 billion crossings (effectively infinite)
                    return 1000000000LL;
                }
            }
        }
        
        // Step 1: Get current crossing count
        long long current_crossings = calculate_total_crossings();
        
        // Step 2: Save current position
        int old_x, old_y;
        CUDA_CHECK(cudaMemcpy(&old_x, d_nodes_x + node_id, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&old_y, d_nodes_y + node_id, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Step 3: Check for collinear edge violations (CRITICAL CONSTRAINT)
        // Can be DISABLED when using smart move generator (which already prevents violations)
        
        long long penalty_change = 0;
        
        if (enable_violation_check) {
            // Get coordinates for violation checking
            std::vector<int> all_x(num_nodes);
            std::vector<int> all_y(num_nodes);
            CUDA_CHECK(cudaMemcpy(all_x.data(), d_nodes_x, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(all_y.data(), d_nodes_y, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
            
            int orig_x = all_x[node_id];
            int orig_y = all_y[node_id];
            
            // STRATEGY: Penalty-based constraint enforcement (same as bottleneck)
            int violations_before = count_collinear_nodes_violations(node_id, all_x, all_y);
            violations_before += count_collinear_violations(node_id, all_x, all_y);
            violations_before += count_edge_through_node_violations(node_id, all_x, all_y);
            
            all_x[node_id] = new_x;
            all_y[node_id] = new_y;
            
            int violations_after = count_collinear_nodes_violations(node_id, all_x, all_y);
            violations_after += count_collinear_violations(node_id, all_x, all_y);
            violations_after += count_edge_through_node_violations(node_id, all_x, all_y);
            
            // Calculate violation penalty change (10B per violation)
            penalty_change = static_cast<long long>(violations_after - violations_before) * 10000000000LL;
        }
        
        // Step 4: Temporarily apply the move
        CUDA_CHECK(cudaMemcpy(d_nodes_x + node_id, &new_x, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes_y + node_id, &new_y, sizeof(int), cudaMemcpyHostToDevice));
        
        // Step 5: Calculate new crossing count
        long long new_crossings = calculate_total_crossings();
        
        // Step 6: Restore original position
        CUDA_CHECK(cudaMemcpy(d_nodes_x + node_id, &old_x, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes_y + node_id, &old_y, sizeof(int), cudaMemcpyHostToDevice));
        
        // Step 7: Return delta + penalty
        return new_crossings - current_crossings + penalty_change;
    }
    
    /**
     * Reset graph to initial state.
     * 
     * OOA State Reset:
     * - Restores GPU memory to constructor values
     * - Enables SA algorithm to restart from known configuration
     */
    void reset_to_initial() {
        if (num_nodes == 0) return;
        
        // Copy backup to current state
        CUDA_CHECK(cudaMemcpy(
            d_nodes_x,
            d_initial_x,
            num_nodes * sizeof(int),
            cudaMemcpyDeviceToDevice
        ));
        
        CUDA_CHECK(cudaMemcpy(
            d_nodes_y,
            d_initial_y,
            num_nodes * sizeof(int),
            cudaMemcpyDeviceToDevice
        ));
    }
    
    // ========================================================================
    // Cycle 3: Spatial Hash Methods
    // ========================================================================
    
    /**
     * Get spatial hash statistics.
     * 
     * Returns dictionary with:
     * - cell_size: Size of grid cells
     * - num_cells: Estimated number of cells used
     * - edges_per_cell_avg: Average edges per cell
     * - enabled: Whether spatial hash is active
     * 
     * @return Dictionary of spatial hash statistics
     */
    std::map<std::string, double> get_spatial_hash_stats() {
        std::map<std::string, double> stats;
        
        stats["enabled"] = use_spatial_hash ? 1.0 : 0.0;
        stats["cell_size"] = static_cast<double>(cell_size);
        
        if (use_spatial_hash && num_nodes > 0) {
            // Get coordinates to compute bounds
            auto coords = get_coordinates();
            const auto& nodes_x = coords.first;
            const auto& nodes_y = coords.second;
            
            // Compute grid dimensions
            int min_x = *std::min_element(nodes_x.begin(), nodes_x.end());
            int max_x = *std::max_element(nodes_x.begin(), nodes_x.end());
            int min_y = *std::min_element(nodes_y.begin(), nodes_y.end());
            int max_y = *std::max_element(nodes_y.begin(), nodes_y.end());
            
            int grid_width = (max_x - min_x) / cell_size + 1;
            int grid_height = (max_y - min_y) / cell_size + 1;
            int num_cells = grid_width * grid_height;
            
            stats["num_cells"] = static_cast<double>(num_cells);
            stats["grid_width"] = static_cast<double>(grid_width);
            stats["grid_height"] = static_cast<double>(grid_height);
            
            // Estimate edges per cell (simple approximation)
            // In reality, edges can span multiple cells
            double edges_per_cell = num_cells > 0 ? 
                static_cast<double>(num_edges) / num_cells : 0.0;
            stats["edges_per_cell_avg"] = edges_per_cell;
        } else {
            stats["num_cells"] = 0.0;
            stats["grid_width"] = 0.0;
            stats["grid_height"] = 0.0;
            stats["edges_per_cell_avg"] = 0.0;
        }
        
        return stats;
    }
    
    /**
     * NEW: Generate smart move on GPU (prevents violations at source).
     * 
     * Purpose:
     *   - Generate valid move directly on GPU
     *   - Avoids violation checking overhead in compute_delta_e
     *   - Matches Python smart move generator logic
     * 
     * Strategy:
     *   - Use GPU kernel to check duplicates and edge violations
     *   - Retry with expanding radius if violations detected
     *   - Return original position if all retries fail
     * 
     * @param node_id: Node to move
     * @param step_size: Maximum move distance
     * @param max_retries: Maximum retry attempts (default: 10)
     * @return Pair (new_x, new_y) - guaranteed valid or original position
     */
    std::pair<int, int> generate_smart_move(
        int node_id,
        int step_size,
        int max_retries = 10
    ) {
        // Allocate device memory for result
        int* d_out_x = nullptr;
        int* d_out_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_out_x, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_out_y, sizeof(int)));
        
        // Generate random seed
        std::random_device rd;
        unsigned int seed = rd();
        
        // Launch smart move generation kernel
        generate_smart_move_kernel<<<1, 1>>>(
            d_nodes_x,
            d_nodes_y,
            d_edges,
            num_nodes,
            num_edges,
            node_id,
            step_size,
            max_retries,
            seed,
            d_out_x,
            d_out_y,
            max_width,
            max_height
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Download result
        int new_x, new_y;
        CUDA_CHECK(cudaMemcpy(&new_x, d_out_x, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&new_y, d_out_y, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_out_x));
        CUDA_CHECK(cudaFree(d_out_y));
        
        return {new_x, new_y};
    }
    
    /**
     * Cycle 4: Run Simulated Annealing optimization on GPU.
     * 
     * This is the core SA loop running entirely in C++/CUDA:
     * - Random node selection
     * - Random position generation (smart or random)
     * - Delta cost computation (GPU) - can be total crossings OR K-value
     * - Metropolis acceptance criterion
     * - Temperature cooling
     * 
     * All operations happen on device - no host-device transfers during loop.
     * 
     * @param iterations: Number of SA iterations
     * @param start_temp: Initial temperature
     * @param cooling_rate: Temperature multiplier per iteration (0.9-0.99)
     * @param cost_function: "total_crossings", "bottleneck_p2", "bottleneck_p3", or "k_value"
     * @param use_smart_moves: Use smart move generation (prevents violations)
     * @param smart_threshold: Temperature threshold for smart moves
     * @return Statistics dictionary (initial/final crossings, accepted moves, etc.)
     */
    std::map<std::string, double> run_sa_optimization(
        int iterations,
        double start_temp,
        double cooling_rate,
        const std::string& cost_function = "total_crossings",
        bool use_smart_moves = false,
        double smart_threshold = 10.0
    ) {
        // Initialize statistics
        std::map<std::string, double> stats;
        
        // Determine which cost function to use
        bool use_k_value = (cost_function == "k_value");
        bool use_bottleneck_p2 = (cost_function == "bottleneck_p2");
        bool use_bottleneck_p3 = (cost_function == "bottleneck_p3");
        
        // Get initial energy
        long long initial_crossings = calculate_total_crossings();
        int initial_k = calculate_k_value();
        stats["initial_crossings"] = static_cast<double>(initial_crossings);
        stats["initial_k"] = static_cast<double>(initial_k);
        
        // SA parameters
        double temperature = start_temp;
        int accepted_moves = 0;
        int rejected_moves = 0;
        
        // Best state tracking (critical for SA!)
        long long best_energy = initial_crossings;
        int best_k = initial_k;
        long long current_energy = initial_crossings;  // Track incrementally
        std::vector<int> best_nodes_x(num_nodes);
        std::vector<int> best_nodes_y(num_nodes);
        CUDA_CHECK(cudaMemcpy(best_nodes_x.data(), d_nodes_x, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(best_nodes_y.data(), d_nodes_y, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> node_dist(0, num_nodes - 1);
        std::uniform_int_distribution<> x_dist(0, max_width);
        std::uniform_int_distribution<> y_dist(0, max_height);
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        
        // Main SA loop
        for (int iter = 0; iter < iterations; iter++) {
            // Select random node
            int node_id = node_dist(gen);
            
            // Generate new position (smart or random)
            int new_x, new_y;
            
            // Calculate step size (matches Python: max(1, int(temp)))
            int step_size = std::max(1, static_cast<int>(temperature));
            
            if (use_smart_moves && temperature <= smart_threshold) {
                // Low temperature: Use smart generation to avoid violations
                auto smart_pos = generate_smart_move(node_id, step_size, 10);
                new_x = smart_pos.first;
                new_y = smart_pos.second;
            } else {
                // High temperature or smart disabled: Random generation AROUND current position
                auto current_pos = get_node_position(node_id);
                int old_x = current_pos.first;
                int old_y = current_pos.second;
                
                // Generate offset in range [-step_size, step_size]
                std::uniform_int_distribution<int> offset_dist(-step_size, step_size);
                int offset_x = offset_dist(gen);
                int offset_y = offset_dist(gen);
                
                // Apply offset and clip to bounds
                new_x = std::max(0, std::min(max_width, old_x + offset_x));
                new_y = std::max(0, std::min(max_height, old_y + offset_y));
            }
            
            // Compute delta cost based on chosen cost function
            long long delta_cost;
            if (use_k_value) {
                // Direct K-value optimization
                delta_cost = compute_delta_k(node_id, new_x, new_y);
            } else if (use_bottleneck_p2) {
                // Quadratic bottleneck penalty: sum(X(e)^2)
                delta_cost = compute_delta_bottleneck(node_id, new_x, new_y, 2);
            } else if (use_bottleneck_p3) {
                // Cubic bottleneck penalty: sum(X(e)^3)
                delta_cost = compute_delta_bottleneck(node_id, new_x, new_y, 3);
            } else {
                // Default: total crossings (p=1)
                delta_cost = compute_delta_e(node_id, new_x, new_y);
            }
            
            // Metropolis acceptance criterion
            bool accept = false;
            if (delta_cost < 0) {
                // Always accept improvement
                accept = true;
            } else if (temperature > 1e-10) {
                // Accept with probability exp(-delta_cost / temperature)
                double prob = std::exp(-static_cast<double>(delta_cost) / temperature);
                if (prob_dist(gen) < prob) {
                    accept = true;
                }
            }
            
            if (accept) {
                // Apply move
                update_node_position(node_id, new_x, new_y);
                accepted_moves++;
                
                // Update current energy incrementally
                current_energy += delta_cost;
                
                // Check if this is the best state so far
                if (current_energy < best_energy) {
                    best_energy = current_energy;
                    best_k = calculate_k_value();  // Only calculate K when we find better solution
                    // Save best state
                    CUDA_CHECK(cudaMemcpy(best_nodes_x.data(), d_nodes_x, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(best_nodes_y.data(), d_nodes_y, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
                }
            } else {
                rejected_moves++;
            }
            
            // Cool down temperature
            temperature *= cooling_rate;
        }
        
        // Restore best state found during optimization
        CUDA_CHECK(cudaMemcpy(d_nodes_x, best_nodes_x.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes_y, best_nodes_y.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
        
        // Return best energy (not final!)
        stats["final_crossings"] = static_cast<double>(best_energy);
        stats["final_k"] = static_cast<double>(best_k);
        stats["iterations"] = static_cast<double>(iterations);
        stats["accepted_moves"] = static_cast<double>(accepted_moves);
        stats["rejected_moves"] = static_cast<double>(rejected_moves);
        stats["final_temperature"] = temperature;
        
        return stats;
    }
};

// ============================================================================
// pybind11 Module Definition
// ============================================================================

/**
 * Python module interface.
 * 
 * Exposes C++ class to Python with automatic type conversions:
 * - std::vector<int> ↔ Python list
 * - std::pair<int, int> ↔ Python tuple
 * 
 * Usage in Python:
 *   import planar_cuda
 *   solver = planar_cuda.PlanarSolver([0, 10], [0, 0], [(0, 1)])
 *   crossings = solver.calculate_total_crossings()
 */
PYBIND11_MODULE(planar_cuda, m) {
    m.doc() = "LCN Solver - CUDA Accelerated Backend (Cycle 3.5: GPU Spatial Hash)";
    
    // Expose PlanarSolver class
    py::class_<PlanarSolver>(m, "PlanarSolver")
        .def(py::init<
            const std::vector<int>&,
            const std::vector<int>&,
            const std::vector<std::pair<int, int>>&,
            int,
            int,
            int
        >(),
        py::arg("nodes_x"),
        py::arg("nodes_y"),
        py::arg("edges"),
        py::arg("cell_size") = -1,
        py::arg("width") = 1000000,
        py::arg("height") = 1000000,
        "Initialize solver with graph data. cell_size: 0=auto, >0=manual, <0=disabled. width/height: coordinate constraints")
        
        .def("calculate_total_crossings", &PlanarSolver::calculate_total_crossings,
            "Calculate total number of edge crossings using GPU")
        
        // NEW: K-Value Calculation
        .def("calculate_k_value", &PlanarSolver::calculate_k_value,
            "Calculate K-value (maximum crossing count on any single edge)")
        
        .def("get_edge_crossings", &PlanarSolver::get_edge_crossings,
            "Get crossing counts for all edges (returns list)")
        
        .def("get_coordinates", &PlanarSolver::get_coordinates,
            "Get current node coordinates")
        
        // Cycle 2: State Management Methods
        .def("update_node_position", &PlanarSolver::update_node_position,
            py::arg("node_id"),
            py::arg("new_x"),
            py::arg("new_y"),
            "Update a single node's position in GPU memory")
        
        .def("get_node_position", &PlanarSolver::get_node_position,
            py::arg("node_id"),
            "Get current position of a specific node")
        
        .def("compute_delta_e", &PlanarSolver::compute_delta_e,
            py::arg("node_id"),
            py::arg("new_x"),
            py::arg("new_y"),
            "Compute change in crossings for a hypothetical move (without applying it)")
        
        .def("compute_delta_k", &PlanarSolver::compute_delta_k,
            py::arg("node_id"),
            py::arg("new_x"),
            py::arg("new_y"),
            "Compute change in K-value for a hypothetical move (without applying it)")
        
        .def("reset_to_initial", &PlanarSolver::reset_to_initial,
            "Reset graph to initial configuration")
        
        // Cycle 3: Spatial Hash Methods
        .def("get_spatial_hash_stats", &PlanarSolver::get_spatial_hash_stats,
            "Get spatial hash statistics (cell_size, num_cells, etc.)")
        
        // Cycle 4: SA Optimizer
        .def("run_sa_optimization", &PlanarSolver::run_sa_optimization,
            py::arg("iterations"),
            py::arg("start_temp"),
            py::arg("cooling_rate"),
            py::arg("cost_function") = "total_crossings",
            py::arg("use_smart_moves") = false,
            py::arg("smart_threshold") = 10.0,
            "Run Simulated Annealing optimization on GPU. cost_function: 'total_crossings', 'bottleneck_p2', 'bottleneck_p3', or 'k_value'. use_smart_moves: enable violation-avoiding move generation. smart_threshold: temperature below which smart moves activate")
        
        // Smart Move Generator
        .def("generate_smart_move", &PlanarSolver::generate_smart_move,
            py::arg("node_id"),
            py::arg("step_size"),
            py::arg("max_retries") = 10,
            "Generate a move that avoids violations (duplicate positions, incident edge interiors)");
    
    // Module metadata
    m.attr("__version__") = "0.5.0-cycle4-smart";
    m.attr("cuda_enabled") = true;
}
