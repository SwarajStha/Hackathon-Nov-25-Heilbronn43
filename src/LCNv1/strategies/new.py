"""
New Architecture Solver Strategy
Uses the new TDD-based architecture with exact delta updates.
"""
import json
import random
import math
from typing import Dict, Any

from .base import ISolverStrategy, SolverFactory
from ..core.geometry import Point
from ..core.graph import GraphData, GridState
from ..core.cost import SoftMaxCost
from ..core.move_generator import HybridMoveGenerator


class NewArchitectureSolverStrategy(ISolverStrategy):
    """
    Modern solver using new architecture:
    - Integer-only geometry
    - Spatial hash for O(E·k) queries
    - Exact delta updates
    - Clean separation of concerns
    """
    
    def __init__(self, w_cross: float = 100.0, w_len: float = 1.0, 
                 power: int = 2, cell_size: int = 50,
                 use_smart_moves: bool = True, smart_threshold: float = 10.0):
        """
        Initialize new architecture solver.
        
        Args:
            w_cross: Weight for crossing penalty
            w_len: Weight for edge length penalty
            power: Exponent for crossing penalty
            cell_size: Spatial hash cell size
            use_smart_moves: 使用智能移動生成（避免違規）
            smart_threshold: 溫度閾值（低於此值使用智能生成）
        """
        self.graph = None
        self.state = None
        self.cost_func = SoftMaxCost(
            w_cross=w_cross,
            w_len=w_len,
            power=power,
            cell_size=cell_size,
            enable_violation_check=(not use_smart_moves)  # 智能生成時關閉檢測
        )
        self.data = None
        
        # Move generation strategy
        self.use_smart_moves = use_smart_moves
        self.move_generator = HybridMoveGenerator(
            max_retries=10,
            smart_threshold=smart_threshold
        ) if use_smart_moves else None
        
        # Optimization state
        self.current_energy = None
        self.best_state = None
        self.best_energy = float('inf')
        self.iteration = 0
    
    def load_from_json(self, json_path: str):
        """Load graph from JSON into new architecture."""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Build graph topology
        edges = [(e['source'], e['target']) for e in self.data['edges']]
        self.graph = GraphData(
            num_nodes=len(self.data['nodes']),
            edges=edges
        )
        
        # Build initial positions
        positions = {}
        for node in self.data['nodes']:
            nid = node['id']
            positions[nid] = Point(int(node['x']), int(node['y']))
        
        self.state = GridState(
            positions,
            width=self.data['width'],
            height=self.data['height']
        )
        
        # Calculate initial energy
        self.current_energy = self.cost_func.calculate(self.graph, self.state)
        self.best_state = self.state.clone()
        self.best_energy = self.current_energy
    
    def solve(self, iterations: int = 1000, **kwargs) -> Dict[str, Any]:
        """
        Run simulated annealing with new architecture.
        
        Args:
            iterations: Number of iterations
            **kwargs: Can include 'initial_temp', 'cooling_rate', 'reheat_threshold'
        """
        if self.graph is None or self.state is None:
            raise RuntimeError("Must load data first using load_from_json()")
        
        # Extract parameters
        initial_temp = kwargs.get('initial_temp', 50.0)
        cooling_rate = kwargs.get('cooling_rate', 0.995)
        reheat_threshold = kwargs.get('reheat_threshold', 500)
        
        # Initialize
        current_temp = initial_temp
        accepted_count = 0
        rejected_count = 0
        steps_since_improvement = 0
        energy_history = []
        
        for i in range(iterations):
            self.iteration = i
            
            # Generate move
            node_id = random.randint(0, self.graph.num_nodes - 1)
            step_size = max(1, int(current_temp))
            
            # Use smart move generator or fallback to random
            if self.use_smart_moves:
                new_pos = self.move_generator.generate_move(
                    self.graph, self.state, node_id, step_size, current_temp
                )
            else:
                # Original random generation
                old_pos = self.state.get_position(node_id)
                new_x = old_pos.x + random.randint(-step_size, step_size)
                new_y = old_pos.y + random.randint(-step_size, step_size)
                new_x = max(0, min(self.state.width, new_x))
                new_y = max(0, min(self.state.height, new_y))
                new_pos = Point(new_x, new_y)
            
            # Calculate delta using exact update
            delta = self.cost_func.calculate_delta(
                self.graph, self.state, node_id, new_pos
            )
            
            # Metropolis criterion
            if delta < 0 or random.random() < math.exp(-delta / max(current_temp, 0.1)):
                # Accept move
                self.state.move_node(node_id, new_pos)
                self.current_energy += delta
                accepted_count += 1
                
                # Track best
                if self.current_energy < self.best_energy:
                    self.best_energy = self.current_energy
                    self.best_state = self.state.clone()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                # Reject move
                rejected_count += 1
                steps_since_improvement += 1
            
            # Cool down
            current_temp *= cooling_rate
            
            # Reheat if stuck
            if steps_since_improvement > reheat_threshold:
                current_temp = initial_temp * 0.5
                steps_since_improvement = 0
            
            # Record energy (sample every 10 iterations)
            if i % 10 == 0:
                energy_history.append(self.current_energy)
        
        # Use best state found
        self.state = self.best_state.clone()
        
        # Calculate final statistics
        k, total = self.cost_func.get_crossing_stats(self.graph, self.state)
        
        # Convert back to JSON format
        nodes = []
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            nodes.append({
                'id': node_id,
                'x': pos.x,
                'y': pos.y
            })
        
        return {
            'nodes': nodes,
            'stats': {
                'iterations': iterations,
                'accepted': accepted_count,
                'rejected': rejected_count,
                'acceptance_rate': accepted_count / iterations,
                'final_temp': current_temp,
                'energy_history': energy_history,
                'method': 'new_architecture_tdd'
            },
            'energy': self.best_energy,
            'k': k,
            'total_crossings': total
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        if self.graph is None or self.state is None:
            return {'error': 'No graph loaded'}
        
        k, total = self.cost_func.get_crossing_stats(self.graph, self.state)
        
        return {
            'energy': self.current_energy,
            'best_energy': self.best_energy,
            'k': k,
            'total_crossings': total,
            'num_nodes': self.graph.num_nodes,
            'num_edges': self.graph.num_edges,
            'iteration': self.iteration
        }
    
    def export_to_json(self, output_path: str):
        """Export current solution to JSON."""
        nodes = []
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            nodes.append({
                'id': node_id,
                'x': pos.x,
                'y': pos.y
            })
        
        output_data = {
            'nodes': nodes,
            'edges': self.data['edges'],
            'width': self.data['width'],
            'height': self.data['height']
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)


# Register with factory
SolverFactory.register_strategy('new', NewArchitectureSolverStrategy)
