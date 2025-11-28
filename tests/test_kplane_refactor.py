"""
Test Suite for K-Plane Solver Refactoring (TDD - Red Phase)

这个测试套件验证 Strategy Pattern 重构的正确性。
遵循 TDD 原则：先写测试（红灯），再写实现（绿灯）。

测试覆盖：
1. InitializationStrategy 接口合约
2. FMME 策略的高温配置
3. Planarization 策略的低温配置
4. KPlaneSolver 与策略的集成
5. 策略切换功能
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 这些导入在实现之前会失败（红灯阶段）
IMPORTS_AVAILABLE = False
try:
    from strategies import (
        InitializationStrategy,
        OptimizerConfig,
        FMMEStrategy,
        PlanarizationStrategy
    )
    from solver import KPlaneSolver
    IMPORTS_AVAILABLE = True
    print("[TDD GREEN PHASE] ✅ All imports successful!")
except ImportError as e:
    print(f"[TDD RED PHASE] ⚠️  Import failure: {e}")


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_nodes():
    """简单的 4 节点图（形成交叉）"""
    return [
        {'id': 0, 'x': 0.0, 'y': 0.0},
        {'id': 1, 'x': 10.0, 'y': 10.0},
        {'id': 2, 'x': 0.0, 'y': 10.0},
        {'id': 3, 'x': 10.0, 'y': 0.0}
    ]

@pytest.fixture
def sample_edges():
    """两条交叉的边"""
    return [
        {'source': 0, 'target': 1},  # (0,0) -> (10,10)
        {'source': 2, 'target': 3}   # (0,10) -> (10,0)
    ]

@pytest.fixture
def canvas_dimensions():
    """画布尺寸"""
    return {'width': 1000.0, 'height': 1000.0}


# ============================================================
# Test 1: Mock Strategy Integration
# ============================================================

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Waiting for implementation")
class TestSolverIntegration:
    """测试 KPlaneSolver 与 InitializationStrategy 的集成"""
    
    def test_solver_accepts_strategy_via_dependency_injection(self, sample_nodes, sample_edges, canvas_dimensions):
        """
        测试 1.1: Solver 通过依赖注入接受策略
        
        验证：
        - KPlaneSolver.__init__ 接受 strategy 参数
        - 不会在内部实例化任何具体策略
        """
        mock_strategy = Mock(spec=InitializationStrategy)
        mock_strategy.generate_layout.return_value = {
            'nodes_x': np.array([0.0, 10.0, 0.0, 10.0]),
            'nodes_y': np.array([0.0, 10.0, 10.0, 0.0])
        }
        mock_strategy.get_optimizer_config.return_value = OptimizerConfig(
            initial_temp=50.0,
            cooling_rate=0.99,
            reheat_threshold=500
        )
        
        # 创建 Solver
        solver = KPlaneSolver(
            sample_nodes,
            sample_edges,
            canvas_dimensions['width'],
            canvas_dimensions['height'],
            strategy=mock_strategy
        )
        
        # 验证策略被调用
        mock_strategy.generate_layout.assert_called_once()
        mock_strategy.get_optimizer_config.assert_called_once()
        
        # 验证 Solver 存储了策略
        assert solver.strategy is mock_strategy
    
    def test_solver_calls_generate_layout_with_correct_parameters(self, sample_nodes, sample_edges, canvas_dimensions):
        """
        测试 1.2: Solver 使用正确的参数调用 generate_layout
        
        验证参数传递：
        - nodes, edges, width, height 正确传递给策略
        """
        mock_strategy = Mock(spec=InitializationStrategy)
        mock_strategy.generate_layout.return_value = {
            'nodes_x': np.array([0.0, 10.0, 0.0, 10.0]),
            'nodes_y': np.array([0.0, 10.0, 10.0, 0.0])
        }
        mock_strategy.get_optimizer_config.return_value = OptimizerConfig(
            initial_temp=50.0,
            cooling_rate=0.99,
            reheat_threshold=500
        )
        
        solver = KPlaneSolver(
            sample_nodes,
            sample_edges,
            canvas_dimensions['width'],
            canvas_dimensions['height'],
            strategy=mock_strategy
        )
        
        # 验证调用参数
        call_args = mock_strategy.generate_layout.call_args
        assert call_args[0][0] == sample_nodes  # nodes
        assert call_args[0][1] == sample_edges  # edges
        assert call_args[0][2] == canvas_dimensions['width']  # width
        assert call_args[0][3] == canvas_dimensions['height']  # height
    
    def test_solver_uses_strategy_optimizer_config(self, sample_nodes, sample_edges, canvas_dimensions):
        """
        测试 1.3: Solver 使用策略提供的优化器配置
        
        验证：
        - Solver 读取 get_optimizer_config() 返回的参数
        - Solver.optimize() 使用这些参数作为默认值
        """
        mock_strategy = Mock(spec=InitializationStrategy)
        mock_strategy.generate_layout.return_value = {
            'nodes_x': np.array([0.0, 10.0, 0.0, 10.0]),
            'nodes_y': np.array([0.0, 10.0, 10.0, 0.0])
        }
        
        # 自定义配置
        custom_config = OptimizerConfig(
            initial_temp=88.0,
            cooling_rate=0.97,
            reheat_threshold=700
        )
        mock_strategy.get_optimizer_config.return_value = custom_config
        
        solver = KPlaneSolver(
            sample_nodes,
            sample_edges,
            canvas_dimensions['width'],
            canvas_dimensions['height'],
            strategy=mock_strategy
        )
        
        # 验证 Solver 存储了配置
        assert hasattr(solver, 'optimizer_config')
        assert solver.optimizer_config.initial_temp == 88.0
        assert solver.optimizer_config.cooling_rate == 0.97
        assert solver.optimizer_config.reheat_threshold == 700


# ============================================================
# Test 2: FMME Strategy (High Temperature)
# ============================================================

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Waiting for implementation")
class TestFMMEStrategy:
    """测试 FMME 策略（高温策略）"""
    
    def test_fmme_returns_high_temperature_config(self):
        """
        测试 2.1: FMME 策略返回高温配置
        
        要求：
        - initial_temp >= 80.0 (高温允许大幅度移动)
        - cooling_rate ~= 0.995 (标准冷却)
        """
        strategy = FMMEStrategy()
        config = strategy.get_optimizer_config()
        
        assert isinstance(config, OptimizerConfig)
        assert config.initial_temp >= 80.0, "FMME 应该使用高温策略（>=80）"
        assert 0.99 <= config.cooling_rate <= 0.999, "冷却率应该在标准范围"
    
    def test_fmme_generates_valid_layout(self, sample_nodes, sample_edges, canvas_dimensions):
        """
        测试 2.2: FMME 生成有效的布局
        
        验证：
        - 返回字典包含 'nodes_x' 和 'nodes_y'
        - 数组长度与节点数匹配
        - 坐标在画布边界内
        """
        strategy = FMMEStrategy()
        layout = strategy.generate_layout(
            sample_nodes,
            sample_edges,
            canvas_dimensions['width'],
            canvas_dimensions['height']
        )
        
        assert 'nodes_x' in layout
        assert 'nodes_y' in layout
        
        nodes_x = layout['nodes_x']
        nodes_y = layout['nodes_y']
        
        assert len(nodes_x) == len(sample_nodes)
        assert len(nodes_y) == len(sample_nodes)
        
        # 验证边界
        assert np.all(nodes_x >= 0) and np.all(nodes_x <= canvas_dimensions['width'])
        assert np.all(nodes_y >= 0) and np.all(nodes_y <= canvas_dimensions['height'])
    
    def test_fmme_uses_spring_layout_internally(self, sample_nodes, sample_edges, canvas_dimensions):
        """
        测试 2.3: FMME 内部使用 NetworkX Spring Layout
        
        验证：
        - 布局不是简单的输入坐标复制
        - 使用力导向算法分散节点
        """
        strategy = FMMEStrategy()
        layout = strategy.generate_layout(
            sample_nodes,
            sample_edges,
            canvas_dimensions['width'],
            canvas_dimensions['height']
        )
        
        # 获取原始输入坐标
        original_x = np.array([n['x'] for n in sample_nodes])
        
        # FMME 应该重新布局，而不是使用原始坐标
        # （允许一定误差，因为可能有缩放）
        assert not np.allclose(layout['nodes_x'][:2], original_x[:2]), \
            "FMME 应该重新计算布局，而非使用输入坐标"


# ============================================================
# Test 3: Planarization Strategy (Low Temperature)
# ============================================================

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Waiting for implementation")
class TestPlanarizationStrategy:
    """测试 Planarization 策略（低温策略）"""
    
    def test_planarization_returns_low_temperature_config(self):
        """
        测试 3.1: Planarization 策略返回低温配置
        
        要求：
        - 10 <= initial_temp <= 20 (低温仅微调)
        - cooling_rate < 0.995 (慢速冷却)
        - reheat_threshold > 500 (更长的重加热间隔)
        """
        strategy = PlanarizationStrategy()
        config = strategy.get_optimizer_config()
        
        assert isinstance(config, OptimizerConfig)
        assert 10.0 <= config.initial_temp <= 20.0, \
            "Planarization 应该使用低温策略（10-20）"
        assert config.cooling_rate < 0.995, \
            "Planarization 应该使用慢速冷却（<0.995）"
        assert config.reheat_threshold >= 500, \
            "Planarization 应该有更长的重加热间隔"
    
    def test_planarization_generates_valid_layout(self, sample_nodes, sample_edges, canvas_dimensions):
        """
        测试 3.2: Planarization 生成有效的布局
        
        验证：
        - 返回有效的坐标数组
        - 尝试最小化交叉（通过插入虚拟节点或重排）
        """
        strategy = PlanarizationStrategy(max_crossings=5)
        layout = strategy.generate_layout(
            sample_nodes,
            sample_edges,
            canvas_dimensions['width'],
            canvas_dimensions['height']
        )
        
        assert 'nodes_x' in layout
        assert 'nodes_y' in layout
        
        nodes_x = layout['nodes_x']
        nodes_y = layout['nodes_y']
        
        # 注意：Planarization 可能增加虚拟节点，所以长度可能 >= 原始节点数
        assert len(nodes_x) >= len(sample_nodes)
        assert len(nodes_y) >= len(sample_nodes)
        
        # 验证边界
        assert np.all(nodes_x >= 0) and np.all(nodes_x <= canvas_dimensions['width'])
        assert np.all(nodes_y >= 0) and np.all(nodes_y <= canvas_dimensions['height'])
    
    def test_planarization_respects_max_crossings_parameter(self):
        """
        测试 3.3: Planarization 遵守 max_crossings 参数
        
        验证：
        - 可以通过构造函数设置最大交叉数
        - 策略会尝试将交叉数控制在该阈值内
        """
        strategy = PlanarizationStrategy(max_crossings=3)
        assert hasattr(strategy, 'max_crossings')
        assert strategy.max_crossings == 3


# ============================================================
# Test 4: Strategy Switching
# ============================================================

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Waiting for implementation")
class TestStrategySwitching:
    """测试策略动态切换"""
    
    def test_solver_can_switch_strategy_dynamically(self, sample_nodes, sample_edges, canvas_dimensions):
        """
        测试 4.1: Solver 可以动态切换策略
        
        验证 OCP：
        - 无需修改 KPlaneSolver 代码即可更换策略
        - 新策略的配置会立即生效
        """
        # 初始使用 FMME
        fmme = FMMEStrategy()
        solver = KPlaneSolver(
            sample_nodes,
            sample_edges,
            canvas_dimensions['width'],
            canvas_dimensions['height'],
            strategy=fmme
        )
        
        initial_config = solver.optimizer_config
        assert initial_config.initial_temp >= 80.0  # 高温
        
        # 切换到 Planarization
        planarization = PlanarizationStrategy()
        solver.strategy = planarization
        solver._reinitialize_with_strategy()  # 重新应用策略
        
        new_config = solver.optimizer_config
        assert 10.0 <= new_config.initial_temp <= 20.0  # 低温
    
    def test_different_strategies_produce_different_configs(self):
        """
        测试 4.2: 不同策略产生不同的配置
        
        验证：
        - FMME 和 Planarization 的配置显著不同
        """
        fmme_config = FMMEStrategy().get_optimizer_config()
        planar_config = PlanarizationStrategy().get_optimizer_config()
        
        # 温度差异显著
        assert fmme_config.initial_temp > planar_config.initial_temp * 3, \
            "FMME 的温度应该显著高于 Planarization"
        
        # 冷却率不同
        assert fmme_config.cooling_rate != planar_config.cooling_rate


# ============================================================
# Test 5: Interface Contract
# ============================================================

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Waiting for implementation")
class TestInitializationStrategyContract:
    """测试 InitializationStrategy 接口合约"""
    
    def test_strategy_must_implement_generate_layout(self):
        """
        测试 5.1: 策略必须实现 generate_layout
        
        验证：
        - 抽象方法不能直接实例化
        - 子类必须实现该方法
        """
        # 尝试实例化抽象类应该失败
        with pytest.raises(TypeError):
            InitializationStrategy()
    
    def test_strategy_must_implement_get_optimizer_config(self):
        """
        测试 5.2: 策略必须实现 get_optimizer_config
        
        验证：
        - 返回类型必须是 OptimizerConfig
        """
        # 测试具体策略
        fmme = FMMEStrategy()
        config = fmme.get_optimizer_config()
        assert isinstance(config, OptimizerConfig)
        
        planar = PlanarizationStrategy()
        config = planar.get_optimizer_config()
        assert isinstance(config, OptimizerConfig)
    
    def test_optimizer_config_has_required_fields(self):
        """
        测试 5.3: OptimizerConfig 包含必需字段
        
        验证：
        - initial_temp, cooling_rate, reheat_threshold 存在
        - 类型正确
        """
        config = OptimizerConfig(
            initial_temp=100.0,
            cooling_rate=0.995,
            reheat_threshold=500
        )
        
        assert isinstance(config.initial_temp, (int, float))
        assert isinstance(config.cooling_rate, (int, float))
        assert isinstance(config.reheat_threshold, int)
        
        # 值范围合理
        assert 0 < config.initial_temp <= 1000
        assert 0.9 <= config.cooling_rate < 1.0
        assert config.reheat_threshold > 0


# ============================================================
# Test 6: Edge Cases
# ============================================================

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Waiting for implementation")
class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_graph(self, canvas_dimensions):
        """
        测试 6.1: 空图处理
        
        验证：
        - 0 个节点的图不会崩溃
        """
        strategy = FMMEStrategy()
        layout = strategy.generate_layout(
            nodes=[],
            edges=[],
            width=canvas_dimensions['width'],
            height=canvas_dimensions['height']
        )
        
        assert len(layout['nodes_x']) == 0
        assert len(layout['nodes_y']) == 0
    
    def test_single_node_graph(self, canvas_dimensions):
        """
        测试 6.2: 单节点图
        
        验证：
        - 单个节点可以正常处理
        """
        strategy = FMMEStrategy()
        layout = strategy.generate_layout(
            nodes=[{'id': 0, 'x': 50.0, 'y': 50.0}],
            edges=[],
            width=canvas_dimensions['width'],
            height=canvas_dimensions['height']
        )
        
        assert len(layout['nodes_x']) == 1
        assert len(layout['nodes_y']) == 1
    
    def test_disconnected_graph(self, canvas_dimensions):
        """
        测试 6.3: 不连通图
        
        验证：
        - 多个不连通的子图可以正常处理
        """
        nodes = [
            {'id': 0, 'x': 0.0, 'y': 0.0},
            {'id': 1, 'x': 10.0, 'y': 0.0},
            {'id': 2, 'x': 100.0, 'y': 100.0},
            {'id': 3, 'x': 110.0, 'y': 100.0}
        ]
        edges = [
            {'source': 0, 'target': 1},  # 子图 1
            {'source': 2, 'target': 3}   # 子图 2 (不连通)
        ]
        
        strategy = FMMEStrategy()
        layout = strategy.generate_layout(
            nodes, edges,
            canvas_dimensions['width'],
            canvas_dimensions['height']
        )
        
        assert len(layout['nodes_x']) == 4
        assert len(layout['nodes_y']) == 4


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == '__main__':
    print("=" * 80)
    print("K-Plane Solver Refactoring - TDD Test Suite")
    print("=" * 80)
    
    if not IMPORTS_AVAILABLE:
        print("\n[TDD RED PHASE] ⚠️  Implementation not found (expected)")
        print("This is normal for TDD - tests are written first.\n")
        print("Next steps:")
        print("1. Implement src/strategies.py")
        print("2. Refactor src/solver.py")
        print("3. Run: pytest tests/test_kplane_refactor.py -v")
        print("=" * 80)
    else:
        print("\n[TDD GREEN PHASE] ✅ Implementation found")
        print("Running tests...\n")
        pytest.main([__file__, '-v', '--tb=short'])
