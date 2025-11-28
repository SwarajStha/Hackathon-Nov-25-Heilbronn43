import unittest
import numpy as np
import pytest

# Skip this test file - scorer is not part of LCNv1 core modules
pytestmark = pytest.mark.skip(reason="scorer module is legacy, not in LCNv1 core")

class TestScorer(unittest.TestCase):
    def test_simple_crossing(self):
        # Two edges crossing: (0,0)-(2,2) and (0,2)-(2,0)
        # Nodes: 0:(0,0), 1:(2,2), 2:(0,2), 3:(2,0)
        nodes_x = np.array([0.0, 2.0, 0.0, 2.0])
        nodes_y = np.array([0.0, 2.0, 2.0, 0.0])
        edges_source = np.array([0, 2])
        edges_target = np.array([1, 3])
        
        edge_crossings, k, total = count_crossings(nodes_x, nodes_y, edges_source, edges_target)
        
        self.assertEqual(total, 1)
        self.assertEqual(k, 1)
        self.assertEqual(edge_crossings[0], 1)
        self.assertEqual(edge_crossings[1], 1)

    def test_no_crossing(self):
        # Parallel lines: (0,0)-(2,0) and (0,1)-(2,1)
        nodes_x = np.array([0.0, 2.0, 0.0, 2.0])
        nodes_y = np.array([0.0, 0.0, 1.0, 1.0])
        edges_source = np.array([0, 2])
        edges_target = np.array([1, 3])
        
        edge_crossings, k, total = count_crossings(nodes_x, nodes_y, edges_source, edges_target)
        
        self.assertEqual(total, 0)
        self.assertEqual(k, 0)

    def test_shared_endpoint(self):
        # Two edges sharing a point: (0,0)-(1,1) and (0,0)-(2,0)
        # Should NOT count as crossing
        nodes_x = np.array([0.0, 1.0, 2.0])
        nodes_y = np.array([0.0, 1.0, 0.0])
        edges_source = np.array([0, 0])
        edges_target = np.array([1, 2])
        
        edge_crossings, k, total = count_crossings(nodes_x, nodes_y, edges_source, edges_target)
        
        self.assertEqual(total, 0)

if __name__ == '__main__':
    unittest.main()
