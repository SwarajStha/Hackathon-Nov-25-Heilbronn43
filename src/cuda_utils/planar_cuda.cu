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
    // - q1 and q2 are on opposite sides of p1-p2: d1 * d2 < 0
    // - p1 and p2 are on opposite sides of q1-q2: d3 * d4 < 0
    //
    // We use STRICTLY less than to exclude:
    // - Collinear cases (where one product is 0)
    // - Endpoint touching (where one product is 0)
    if (d1 * d2 < 0 && d3 * d4 < 0) {
        return true;
    }
    
    // All other cases (collinear, parallel, touching) are not crossings
    return false;
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
        num_nodes(nodes_x.size()), num_edges(edges.size()),
        bbox_cached(false), edge_cells_cached(false),
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
    
    /**
     * Compute delta-E (change in crossings) for a hypothetical move.
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
        
        // Step 1: Get current crossing count
        long long current_crossings = calculate_total_crossings();
        
        // Step 2: Save current position
        int old_x, old_y;
        CUDA_CHECK(cudaMemcpy(&old_x, d_nodes_x + node_id, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&old_y, d_nodes_y + node_id, sizeof(int), cudaMemcpyDeviceToHost));
        
        // Step 3: Check for collinear edge violations (CRITICAL CONSTRAINT)
        // Temporarily update coordinates in memory for checking
        int orig_x = all_x[node_id];
        int orig_y = all_y[node_id];
        all_x[node_id] = new_x;
        all_y[node_id] = new_y;
        
        int collinear_violations = count_collinear_violations(node_id, all_x, all_y);
        
        if (collinear_violations > 0) {
            // VIOLATION: Collinear edges detected!
            // Each violation gets 500 million penalty
            // This ensures SA will reject moves that create shared line segments
            all_x[node_id] = orig_x;
            all_y[node_id] = orig_y;
            return static_cast<long long>(collinear_violations) * 500000000LL;
        }
        
        // Step 3.5: Check for edges passing through nodes (CRITICAL CONSTRAINT)
        // NOTE: Coordinates are still at NEW position (from Step 3 above)
        int edge_through_node_violations = count_edge_through_node_violations(node_id, all_x, all_y);
        
        if (edge_through_node_violations > 0) {
            // VIOLATION: Edge passes through non-endpoint node!
            // Restore coordinates before returning
            all_x[node_id] = orig_x;
            all_y[node_id] = orig_y;
            // Each violation gets 300 million penalty
            // This ensures SA will reject moves where edges pass through nodes
            return static_cast<long long>(edge_through_node_violations) * 300000000LL;
        }
        
        // Restore for subsequent steps (crossing calculation)
        all_x[node_id] = orig_x;
        all_y[node_id] = orig_y;
        
        // Step 4: Temporarily apply the move
        CUDA_CHECK(cudaMemcpy(d_nodes_x + node_id, &new_x, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes_y + node_id, &new_y, sizeof(int), cudaMemcpyHostToDevice));
        
        // Step 5: Calculate new crossing count
        long long new_crossings = calculate_total_crossings();
        
        // Step 6: Restore original position
        CUDA_CHECK(cudaMemcpy(d_nodes_x + node_id, &old_x, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes_y + node_id, &old_y, sizeof(int), cudaMemcpyHostToDevice));
        
        // Step 7: Return delta
        return new_crossings - current_crossings;
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
     * Cycle 4: Run Simulated Annealing optimization on GPU.
     * 
     * This is the core SA loop running entirely in C++/CUDA:
     * - Random node selection
     * - Random position generation
     * - Delta-E computation (GPU)
     * - Metropolis acceptance criterion
     * - Temperature cooling
     * 
     * All operations happen on device - no host-device transfers during loop.
     * 
     * @param iterations: Number of SA iterations
     * @param start_temp: Initial temperature
     * @param cooling_rate: Temperature multiplier per iteration (0.9-0.99)
     * @return Statistics dictionary (initial/final crossings, accepted moves, etc.)
     */
    std::map<std::string, double> run_sa_optimization(
        int iterations,
        double start_temp,
        double cooling_rate
    ) {
        // Initialize statistics
        std::map<std::string, double> stats;
        
        // Get initial energy
        long long initial_crossings = calculate_total_crossings();
        stats["initial_crossings"] = static_cast<double>(initial_crossings);
        
        // SA parameters
        double temperature = start_temp;
        int accepted_moves = 0;
        int rejected_moves = 0;
        
        // Get current coordinates to compute bounds
        auto coords = get_coordinates();
        const auto& nodes_x = coords.first;
        const auto& nodes_y = coords.second;
        
        // Use coordinate constraints for position generation
        // Ensures all positions are within [0, max_width] x [0, max_height]
        
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
            
            // Generate random new position
            int new_x = x_dist(gen);
            int new_y = y_dist(gen);
            
            // Compute delta-E (only edges connected to node_id)
            int delta_e = compute_delta_e(node_id, new_x, new_y);
            
            // Metropolis acceptance criterion
            bool accept = false;
            if (delta_e < 0) {
                // Always accept improvement
                accept = true;
            } else if (temperature > 1e-10) {
                // Accept with probability exp(-delta_e / temperature)
                double prob = std::exp(-static_cast<double>(delta_e) / temperature);
                if (prob_dist(gen) < prob) {
                    accept = true;
                }
            }
            
            if (accept) {
                // Apply move
                update_node_position(node_id, new_x, new_y);
                accepted_moves++;
            } else {
                rejected_moves++;
            }
            
            // Cool down temperature
            temperature *= cooling_rate;
        }
        
        // Get final energy
        long long final_crossings = calculate_total_crossings();
        stats["final_crossings"] = static_cast<double>(final_crossings);
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
            "Run Simulated Annealing optimization on GPU");
    
    // Module metadata
    m.attr("__version__") = "0.5.0-cycle4";
    m.attr("cuda_enabled") = true;
}
