"""
Graph construction and Dijkstra pathfinding for grid-based road network.
"""

import heapq
from typing import Dict, List, Tuple, Set
import numpy as np


class GridGraph:
    """
    Represents a 2D grid-based road network.
    
    Each intersection is a node, and roads connect adjacent nodes.
    """
    
    def __init__(self, grid_size: Tuple[int, int], block_spacing: float):
        """
        Initialize grid graph.
        
        Args:
            grid_size: (rows, cols) number of intersections
            block_spacing: distance in meters between adjacent intersections
        """
        self.grid_size = grid_size
        self.block_spacing = block_spacing
        self.rows, self.cols = grid_size
        
        # Build adjacency list
        self.adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._build_graph()
        
    def _build_graph(self):
        """Build adjacency list for grid graph."""
        for r in range(self.rows):
            for c in range(self.cols):
                node = (r, c)
                neighbors = []
                
                # Up
                if r > 0:
                    neighbors.append((r - 1, c))
                # Down
                if r < self.rows - 1:
                    neighbors.append((r + 1, c))
                # Left
                if c > 0:
                    neighbors.append((r, c - 1))
                # Right
                if c < self.cols - 1:
                    neighbors.append((r, c + 1))
                    
                self.adjacency[node] = neighbors
    
    def get_node_position(self, node: Tuple[int, int]) -> np.ndarray:
        """
        Convert grid coordinates to physical position.
        
        Args:
            node: (row, col) grid coordinates
            
        Returns:
            np.array([x, y]) position in meters
        """
        r, c = node
        # Map grid to physical coordinates
        # row 0 is at y = 0, row increases southward (y increases)
        # col 0 is at x = 0, col increases eastward (x increases)
        x = c * self.block_spacing
        y = r * self.block_spacing
        return np.array([x, y], dtype=np.float64)
    
    def dijkstra(self, start_node: Tuple[int, int], goal_node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Compute shortest path using Dijkstra's algorithm.
        
        Args:
            start_node: (row, col) starting intersection
            goal_node: (row, col) destination intersection
            
        Returns:
            List of nodes from start to goal (inclusive)
        """
        # Priority queue: (distance, node)
        pq = [(0.0, start_node)]
        distances = {start_node: 0.0}
        previous = {start_node: None}
        visited: Set[Tuple[int, int]] = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            # Goal reached
            if current_node == goal_node:
                break
            
            # Explore neighbors
            for neighbor in self.adjacency[current_node]:
                if neighbor in visited:
                    continue
                
                # Edge weight is Euclidean distance
                edge_weight = self.block_spacing  # All edges have same length in grid
                new_dist = current_dist + edge_weight
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct path
        if goal_node not in previous and goal_node != start_node:
            raise ValueError(f"No path found from {start_node} to {goal_node}")
        
        path = []
        current = goal_node
        while current is not None:
            path.append(current)
            current = previous[current]
        
        path.reverse()
        return path
    
    def get_all_edges(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Get all edges in the graph for rendering.
        
        Returns:
            List of (node1, node2) tuples
        """
        edges = []
        visited_edges = set()
        
        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                edge = tuple(sorted([node, neighbor]))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    edges.append((node, neighbor))
        
        return edges