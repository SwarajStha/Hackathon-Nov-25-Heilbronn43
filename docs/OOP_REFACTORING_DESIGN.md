# OOP Refactoring Design Document

## 🎯 Objectives

根據 PERFORMANCE_ANALYSIS.md 的發現，進行 OOA/OOD/OOP 重構：

1. **修正核心問題**: Cost function 從優化 total crossings 改為優化 K-value
2. **減少記憶體瓶頸**: 減少 CPU-GPU 傳輸，實現 batch operations
3. **提升可測試性**: 採用 TDD，清晰的類設計和接口
4. **提升可維護性**: 策略模式支持多種 cost functions

---

## 📋 Phase 1: OOA (Object-Oriented Analysis)

### 1.1 Domain Model - 核心概念識別

```
┌─────────────────────────────────────────────────────────────┐
│                     Planar Graph Solver                      │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐    ┌──────────┐   ┌──────────┐
         │  Graph  │    │Optimizer │   │Evaluator │
         └─────────┘    └──────────┘   └──────────┘
              │               │               │
      ┌───────┴───────┐      │       ┌───────┴────────┐
      ▼               ▼       ▼       ▼                ▼
   ┌──────┐      ┌──────┐ ┌────┐ ┌─────────┐   ┌──────────┐
   │ Node │      │ Edge │ │ SA │ │Crossing │   │ K-Value  │
   └──────┘      └──────┘ └────┘ │ Counter │   │Calculator│
                                  └─────────┘   └──────────┘
```

### 1.2 Core Entities

#### 1. **Graph** (圖結構)
- **責任**: 維護節點和邊的結構關係
- **屬性**: nodes, edges, adjacency
- **行為**: add_node, add_edge, get_neighbors

#### 2. **Node** (節點)
- **責任**: 存儲節點坐標和屬性
- **屬性**: id, x, y, fixed
- **行為**: move_to, is_valid_position

#### 3. **Edge** (邊)
- **責任**: 連接兩個節點
- **屬性**: node1, node2
- **行為**: get_endpoints, get_length

#### 4. **CrossingCalculator** (交叉計算器)
- **責任**: 計算兩條邊是否相交
- **屬性**: GPU kernel reference
- **行為**: segments_intersect, count_all_crossings

#### 5. **KValueCalculator** (K值計算器)
- **責任**: 計算每條邊的交叉數，找出最大值
- **屬性**: edge_crossing_counts (array on GPU)
- **行為**: calculate_k_value, get_edge_crossings

#### 6. **CostFunction** (成本函數 - 抽象)
- **責任**: 定義優化目標
- **子類**: TotalCrossingCost, KValueCost
- **行為**: compute_cost, compute_delta

#### 7. **SAOptimizer** (模擬退火優化器)
- **責任**: 執行優化算法
- **屬性**: temperature, cooling_rate, cost_function
- **行為**: optimize, accept_move, cool_down

#### 8. **GPUMemoryManager** (GPU記憶體管理器)
- **責任**: 管理 CPU-GPU 數據傳輸
- **屬性**: device_buffers, sync_status
- **行為**: upload, download, sync, batch_upload

### 1.3 Use Cases

```
User Story 1: 計算當前 K-value
  Actor: Python User
  Flow: 
    1. solver = PlanarSolver(nodes, edges)
    2. k = solver.calculate_k_value()
    3. System returns max crossing count among all edges

User Story 2: 使用 K-value 優化圖結構
  Actor: Python User
  Flow:
    1. solver = PlanarSolver(nodes, edges)
    2. solver.set_cost_function("k_value")
    3. solver.run_optimization(iterations=50000)
    4. System minimizes K-value (not total crossings)

User Story 3: 比較不同 cost functions
  Actor: Developer
  Flow:
    1. Run SA with TotalCrossingCost
    2. Run SA with KValueCost
    3. Compare results and convergence
```

---

## 📐 Phase 2: OOD (Object-Oriented Design)

### 2.1 Class Hierarchy

```cpp
// ============================================================================
// Abstract Base: Cost Function Strategy
// ============================================================================

class ICostFunction {
public:
    virtual ~ICostFunction() = default;
    
    // Compute current cost
    virtual double compute_cost() = 0;
    
    // Compute delta cost for a proposed move
    // Returns: new_cost - old_cost
    virtual double compute_delta(int node_id, int new_x, int new_y) = 0;
    
    // Get cost name for logging
    virtual std::string get_name() const = 0;
};

// ============================================================================
// Concrete: Total Crossing Cost (Original)
// ============================================================================

class TotalCrossingCost : public ICostFunction {
private:
    PlanarGraph* graph_;
    CrossingCalculator* calculator_;
    
public:
    TotalCrossingCost(PlanarGraph* g, CrossingCalculator* calc);
    
    double compute_cost() override {
        return calculator_->count_all_crossings();
    }
    
    double compute_delta(int node_id, int new_x, int new_y) override;
    
    std::string get_name() const override { return "TotalCrossings"; }
};

// ============================================================================
// Concrete: K-Value Cost (NEW - Main Objective)
// ============================================================================

class KValueCost : public ICostFunction {
private:
    PlanarGraph* graph_;
    KValueCalculator* k_calculator_;
    
public:
    KValueCost(PlanarGraph* g, KValueCalculator* calc);
    
    double compute_cost() override {
        return k_calculator_->calculate_k_value();
    }
    
    double compute_delta(int node_id, int new_x, int new_y) override;
    
    std::string get_name() const override { return "KValue"; }
};

// ============================================================================
// Core: K-Value Calculator
// ============================================================================

class KValueCalculator {
private:
    PlanarGraph* graph_;
    CrossingCalculator* crossing_calc_;
    
    // GPU memory for per-edge crossing counts
    int* d_edge_crossings_;  // Device array [num_edges]
    int* h_edge_crossings_;  // Host mirror
    
    int num_edges_;
    bool needs_update_;
    
public:
    KValueCalculator(PlanarGraph* graph, CrossingCalculator* calc);
    ~KValueCalculator();
    
    // Calculate K-value (max crossing count among all edges)
    int calculate_k_value();
    
    // Get crossing count for specific edge
    int get_edge_crossing_count(int edge_id);
    
    // Get all edge crossing counts
    std::vector<int> get_all_edge_crossings();
    
    // Mark that graph changed, need recalculation
    void mark_dirty() { needs_update_ = true; }
    
private:
    // CUDA kernel to count crossings per edge
    void compute_edge_crossings_gpu();
    
    // Find max value on GPU
    int find_max_on_gpu();
};

// ============================================================================
// Core: Simulated Annealing Optimizer
// ============================================================================

class SAOptimizer {
private:
    PlanarGraph* graph_;
    ICostFunction* cost_function_;  // ✅ Strategy Pattern
    
    // SA parameters
    double temperature_;
    double start_temperature_;
    double cooling_rate_;
    int iterations_;
    
    // Random number generator
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_;
    
    // Statistics
    int accepted_moves_;
    int rejected_moves_;
    std::vector<double> cost_history_;
    
public:
    SAOptimizer(
        PlanarGraph* graph,
        ICostFunction* cost_func,
        double start_temp = 100.0,
        double cooling_rate = 0.95
    );
    
    // Run optimization
    OptimizationResult optimize(int iterations);
    
    // Set cost function (strategy switching)
    void set_cost_function(ICostFunction* cost_func) {
        cost_function_ = cost_func;
    }
    
    // SA parameters
    void set_temperature(double temp) { start_temperature_ = temp; }
    void set_cooling_rate(double rate) { cooling_rate_ = rate; }
    
private:
    // Metropolis acceptance criterion
    bool accept_move(double delta_cost);
    
    // Temperature schedule
    void cool_down();
    
    // Generate random move
    std::tuple<int, int, int> generate_move();
};

// ============================================================================
// Core: GPU Memory Manager (Reduce CPU-GPU Transfer)
// ============================================================================

class GPUMemoryManager {
private:
    // Device memory pools
    int* d_nodes_x_;
    int* d_nodes_y_;
    int* d_edges_;
    
    int num_nodes_;
    int num_edges_;
    
    // Sync flags
    bool nodes_synced_;
    bool edges_synced_;
    
public:
    GPUMemoryManager(int num_nodes, int num_edges);
    ~GPUMemoryManager();
    
    // Upload data once
    void upload_nodes(const std::vector<int>& x, const std::vector<int>& y);
    void upload_edges(const std::vector<std::pair<int,int>>& edges);
    
    // Incremental updates (single node)
    void update_node_position(int node_id, int x, int y);
    
    // Batch updates (multiple nodes)
    void batch_update_nodes(const std::vector<std::tuple<int,int,int>>& updates);
    
    // Download results
    void download_nodes(std::vector<int>& x, std::vector<int>& y);
    
    // Get device pointers
    int* get_device_nodes_x() { return d_nodes_x_; }
    int* get_device_nodes_y() { return d_nodes_y_; }
    int* get_device_edges() { return d_edges_; }
};

// ============================================================================
// Core: Planar Graph
// ============================================================================

class PlanarGraph {
private:
    std::vector<int> nodes_x_;
    std::vector<int> nodes_y_;
    std::vector<bool> nodes_fixed_;
    std::vector<std::pair<int, int>> edges_;
    
    GPUMemoryManager* gpu_mem_;
    
public:
    PlanarGraph(
        const std::vector<int>& x,
        const std::vector<int>& y,
        const std::vector<std::pair<int, int>>& edges
    );
    
    ~PlanarGraph();
    
    // Node operations
    void move_node(int node_id, int new_x, int new_y);
    bool is_node_fixed(int node_id) const;
    
    // Accessors
    int get_num_nodes() const { return nodes_x_.size(); }
    int get_num_edges() const { return edges_.size(); }
    
    const std::vector<int>& get_nodes_x() const { return nodes_x_; }
    const std::vector<int>& get_nodes_y() const { return nodes_y_; }
    const std::vector<std::pair<int,int>>& get_edges() const { return edges_; }
    
    GPUMemoryManager* get_gpu_memory() { return gpu_mem_; }
};
```

### 2.2 CUDA Kernels Design

```cpp
// ============================================================================
// NEW KERNEL: Count crossings for each edge
// ============================================================================

__global__ void count_edge_crossings_kernel(
    const int* nodes_x,
    const int* nodes_y,
    const int* edges,      // Flattened [e0_n1, e0_n2, e1_n1, e1_n2, ...]
    int num_edges,
    int* edge_crossings    // Output: crossing count per edge
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= num_edges) return;
    
    // Get endpoints of current edge
    int n1 = edges[edge_idx * 2];
    int n2 = edges[edge_idx * 2 + 1];
    int x1 = nodes_x[n1], y1 = nodes_y[n1];
    int x2 = nodes_x[n2], y2 = nodes_y[n2];
    
    int count = 0;
    
    // Compare with all other edges
    for (int other_idx = 0; other_idx < num_edges; other_idx++) {
        if (other_idx == edge_idx) continue;
        
        int n3 = edges[other_idx * 2];
        int n4 = edges[other_idx * 2 + 1];
        
        // Skip if shared endpoint
        if (n1 == n3 || n1 == n4 || n2 == n3 || n2 == n4) continue;
        
        int x3 = nodes_x[n3], y3 = nodes_y[n3];
        int x4 = nodes_x[n4], y4 = nodes_y[n4];
        
        if (segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4)) {
            count++;
        }
    }
    
    edge_crossings[edge_idx] = count;
}

// ============================================================================
// NEW KERNEL: Find maximum value in array (parallel reduction)
// ============================================================================

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
    
    // Parallel reduction (max)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicMax(result, sdata[0]);
    }
}
```

### 2.3 Dependency Injection

```cpp
// ============================================================================
// Dependency Injection Container (Optional, for advanced design)
// ============================================================================

class PlanarSolverFactory {
public:
    static std::unique_ptr<PlanarSolver> create_with_k_value_optimization(
        const std::vector<int>& nodes_x,
        const std::vector<int>& nodes_y,
        const std::vector<std::pair<int, int>>& edges
    ) {
        auto graph = std::make_unique<PlanarGraph>(nodes_x, nodes_y, edges);
        auto crossing_calc = std::make_unique<CrossingCalculator>(graph.get());
        auto k_calc = std::make_unique<KValueCalculator>(graph.get(), crossing_calc.get());
        auto cost_func = std::make_unique<KValueCost>(graph.get(), k_calc.get());
        auto optimizer = std::make_unique<SAOptimizer>(graph.get(), cost_func.get());
        
        return std::make_unique<PlanarSolver>(
            std::move(graph),
            std::move(crossing_calc),
            std::move(k_calc),
            std::move(cost_func),
            std::move(optimizer)
        );
    }
};
```

---

## 🧪 Phase 3: TDD (Test-Driven Development)

### 3.1 Test Structure

```
tests/
├── cpp/                          # C++ unit tests (Google Test)
│   ├── test_k_value_calculator.cpp
│   ├── test_cost_functions.cpp
│   ├── test_sa_optimizer.cpp
│   └── test_gpu_memory_manager.cpp
│
├── python/                       # Python integration tests
│   ├── test_k_value_basic.py
│   ├── test_k_value_vs_total.py
│   ├── test_optimization_convergence.py
│   └── test_benchmark_instances.py
│
└── fixtures/                     # Test data
    ├── simple_graph_k3.json      # K=3 expected
    ├── simple_graph_k5.json      # K=5 expected
    └── benchmark_70nodes.json
```

### 3.2 Test Cases (TDD Red-Green-Refactor)

#### Test 1: K-Value Calculator - Basic

```cpp
// tests/cpp/test_k_value_calculator.cpp

TEST(KValueCalculatorTest, SimpleTriangleGraph) {
    // Arrange: Create graph with known K-value
    //   0---1
    //    \ /
    //     2
    // Edges: (0,1), (1,2), (0,2)
    // Expected: K = 0 (no crossings in triangle)
    
    std::vector<int> x = {0, 10, 5};
    std::vector<int> y = {0, 0, 10};
    std::vector<std::pair<int, int>> edges = {{0,1}, {1,2}, {0,2}};
    
    PlanarGraph graph(x, y, edges);
    CrossingCalculator crossing_calc(&graph);
    KValueCalculator k_calc(&graph, &crossing_calc);
    
    // Act
    int k = k_calc.calculate_k_value();
    
    // Assert
    EXPECT_EQ(k, 0) << "Triangle graph should have K=0";
}

TEST(KValueCalculatorTest, K4CompleteGraph) {
    // K4 complete graph (non-planar)
    // Expected: K >= 1 (must have at least 1 crossing)
    
    std::vector<int> x = {0, 10, 10, 0};
    std::vector<int> y = {0, 0, 10, 10};
    std::vector<std::pair<int, int>> edges = {
        {0,1}, {1,2}, {2,3}, {3,0},  // Square
        {0,2}, {1,3}                  // Diagonals (cross!)
    };
    
    PlanarGraph graph(x, y, edges);
    CrossingCalculator crossing_calc(&graph);
    KValueCalculator k_calc(&graph, &crossing_calc);
    
    int k = k_calc.calculate_k_value();
    
    EXPECT_GE(k, 1) << "K4 must have at least 1 crossing";
    
    // Verify edge crossings
    auto edge_crossings = k_calc.get_all_edge_crossings();
    EXPECT_EQ(edge_crossings.size(), 6);
    
    // Diagonals should each cross once
    EXPECT_EQ(edge_crossings[4], 1) << "Diagonal (0,2) crosses (1,3)";
    EXPECT_EQ(edge_crossings[5], 1) << "Diagonal (1,3) crosses (0,2)";
}
```

#### Test 2: Cost Function Comparison

```python
# tests/python/test_k_value_vs_total.py

def test_cost_function_behavior():
    """
    Verify that K-value cost and Total crossing cost
    can give different optimization directions.
    """
    # Arrange: Create scenario where they differ
    #   Move A: K=10→12, Total=100→95  (bad for K, good for total)
    #   Move B: K=10→8,  Total=100→102 (good for K, bad for total)
    
    solver = planar_cuda.PlanarSolver(nodes_x, nodes_y, edges)
    
    # Test 1: Total Crossing Cost
    solver.set_cost_function("total_crossings")
    delta_total_A = solver.compute_delta(node_id, new_x_A, new_y_A)
    assert delta_total_A < 0, "Move A should be accepted by total crossing cost"
    
    # Test 2: K-Value Cost
    solver.set_cost_function("k_value")
    delta_k_A = solver.compute_delta(node_id, new_x_A, new_y_A)
    assert delta_k_A > 0, "Move A should be rejected by K-value cost"
    
    delta_k_B = solver.compute_delta(node_id, new_x_B, new_y_B)
    assert delta_k_B < 0, "Move B should be accepted by K-value cost"
```

#### Test 3: Optimization Convergence

```python
# tests/python/test_optimization_convergence.py

def test_k_value_optimization_converges():
    """
    Test that SA with K-value cost actually reduces K.
    """
    # Load benchmark instance
    data = load_instance("live-2025-example-instances/70.json")
    
    solver = planar_cuda.PlanarSolver(
        data['nodes_x'], data['nodes_y'], data['edges']
    )
    
    # Set K-value optimization
    solver.set_cost_function("k_value")
    
    # Initial K
    initial_k = solver.calculate_k_value()
    
    # Run optimization
    result = solver.run_sa_optimization(
        iterations=50000,
        start_temp=100.0,
        cooling_rate=0.99
    )
    
    # Final K
    final_k = solver.calculate_k_value()
    
    # Assert improvement
    assert final_k < initial_k, f"K should decrease: {initial_k} → {final_k}"
    assert final_k <= 10, f"70-nodes target K≤10, got {final_k}"
    
    # Check solution validity
    violations = solver.check_violations()
    assert violations == 0, "Solution must be valid"
```

---

## 🔄 Phase 4: Implementation Roadmap

### Step 1: Create Test Infrastructure (Day 1 Morning)

```bash
# Install Google Test
cd tests/cpp
git clone https://github.com/google/googletest.git
cd googletest && mkdir build && cd build
cmake .. && make && sudo make install

# Create CMakeLists.txt for tests
```

### Step 2: Write Failing Tests (Day 1 Afternoon)

```bash
# Write tests that expect K-value functionality
tests/cpp/test_k_value_calculator.cpp
tests/python/test_k_value_basic.py

# Run tests (should FAIL - Red phase)
cd tests/cpp && make test
cd tests/python && pytest
```

### Step 3: Implement K-Value Calculator (Day 2)

```bash
# Implement in order:
1. count_edge_crossings_kernel (CUDA)
2. find_max_kernel (CUDA)
3. KValueCalculator class (C++)
4. Python bindings

# Run tests (should PASS - Green phase)
```

### Step 4: Implement Cost Function Strategy (Day 2-3)

```bash
# Implement:
1. ICostFunction interface
2. TotalCrossingCost (refactor existing)
3. KValueCost (new)
4. SAOptimizer refactoring

# Run tests
```

### Step 5: Optimize GPU Memory (Day 3)

```bash
# Implement:
1. GPUMemoryManager class
2. Batch operations
3. Reduce CPU-GPU transfers

# Benchmark performance
```

### Step 6: Integration and Benchmarking (Day 4)

```bash
# Run full benchmark suite
python tests/python/test_benchmark_instances.py

# Compare results:
# - 70-nodes: Target K ≤ 10
# - 100-nodes: Target K ≤ 10
# - 150-nodes: Target K ≤ 15
```

---

## 📊 Expected Results

### Before Refactoring (Current)

| Instance  | K-value | Total Crossings | Time (s) |
|-----------|---------|-----------------|----------|
| 70-nodes  | 19      | ?               | 6.7      |
| 100-nodes | 20      | ?               | 6.0      |

### After Refactoring (Target)

| Instance  | K-value | Total Crossings | Time (s) | Improvement |
|-----------|---------|-----------------|----------|-------------|
| 70-nodes  | ≤ 10    | ?               | 3-5      | K: 50%↓     |
| 100-nodes | ≤ 10    | ?               | 3-5      | K: 50%↓     |
| 150-nodes | ≤ 15    | ?               | 5-7      | New target  |

---

## 🎓 OOP Design Patterns Used

1. **Strategy Pattern**: ICostFunction with TotalCrossingCost & KValueCost
2. **Dependency Injection**: Components receive dependencies via constructor
3. **Factory Pattern**: PlanarSolverFactory creates configured instances
4. **RAII (Resource Acquisition Is Initialization)**: GPU memory management
5. **Template Method**: SAOptimizer defines algorithm skeleton
6. **Observer Pattern** (Optional): Cost history tracking

---

## 📚 References

- **OOA/OOD**: "Object-Oriented Analysis and Design" by Grady Booch
- **TDD**: "Test Driven Development: By Example" by Kent Beck
- **CUDA Best Practices**: NVIDIA CUDA Programming Guide
- **Design Patterns**: "Design Patterns" by Gang of Four

---

**Document Version**: 1.0  
**Date**: November 29, 2025  
**Status**: Design Phase - Ready for Implementation
