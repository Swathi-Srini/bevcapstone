"""
Route geometry construction with left-lane offset for left-hand driving rules.
"""

import numpy as np
from typing import List, Tuple


class RouteGeometry:
    """
    Constructs continuous route from discrete node path.
    
    Handles:
    - Piecewise linear path interpolation
    - Tangent and normal vector computation
    - Arc-length parameterization
    - Left-lane centerline offset (India-style left-hand driving)
    """
    
    def __init__(
        self,
        node_positions: List[np.ndarray],
        interpolation_step: float = 0.5,
        lane_width: float = 3.0,
        road_width: float = 8.0,
        corner_radius: float = 6.0  # NEW: smooth corners with this radius
    ):
        """
        Initialize route geometry from node positions.
        
        Args:
            node_positions: List of [x, y] positions along the route
            interpolation_step: Spatial resolution for interpolation (meters)
            lane_width: Width of a single lane (meters)
            road_width: Total road width (meters)
            corner_radius: Radius for rounding sharp corners (meters)
        """
        self.node_positions = node_positions
        self.interpolation_step = interpolation_step
        self.lane_width = lane_width
        self.road_width = road_width
        self.corner_radius = corner_radius
        
        # Computed route data
        self.route_points = None  # Nx2 array of interpolated points
        self.route_tangents = None  # Nx2 array of unit tangent vectors
        self.route_normals = None  # Nx2 array of unit normal vectors
        self.route_s = None  # Nx1 array of arc-length values
        self.route_curvature = None  # Nx1 array of curvature values
        self.total_length = 0.0
        
        self._build_route()
    
    def _smooth_corners(self, nodes: np.ndarray) -> np.ndarray:
        """Smooth sharp corners with circular arcs."""
        if len(nodes) < 3 or self.corner_radius <= 0:
            return nodes  # No smoothing needed
        
        smoothed = [nodes[0]]  # Start point unchanged
        
        for i in range(1, len(nodes) - 1):
            p_prev = nodes[i - 1]
            p_curr = nodes[i]
            p_next = nodes[i + 1]
            
            # Vectors to current corner
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            
            if len1 < 1e-6 or len2 < 1e-6:
                smoothed.append(p_curr)
                continue
            
            # Normalize
            v1_unit = v1 / len1
            v2_unit = v2 / len2
            
            # Check if corner is sharp enough to smooth
            dot = np.dot(v1_unit, v2_unit)
            if dot > 0.9:  # Nearly straight, no smoothing
                smoothed.append(p_curr)
                continue
            
            # Distance from corner to arc tangent points
            angle = np.arccos(np.clip(dot, -1, 1))
            if angle < 0.1:  # Too straight
                smoothed.append(p_curr)
                continue
                
            dist = self.corner_radius * np.tan(angle / 2)
            dist = min(dist, len1 * 0.4, len2 * 0.4)  # Don't overshoot segments
            
            # Arc start and end points
            arc_start = p_curr - v1_unit * dist
            arc_end = p_curr + v2_unit * dist
            
            # Generate arc points
            n_arc_points = max(3, int(dist * 2 / self.interpolation_step))
            for j in range(n_arc_points):
                t = j / (n_arc_points - 1)
                # Simple circular interpolation
                point = (1 - t) * arc_start + t * arc_end
                # Bulge toward corner for curvature
                mid = (arc_start + arc_end) / 2
                bulge_vec = p_curr - mid
                bulge_amount = (1 - (2 * t - 1)**2)  # Parabolic
                point = point + bulge_vec * bulge_amount * 0.3
                smoothed.append(point)
        
        smoothed.append(nodes[-1])  # End point unchanged
        return np.array(smoothed)
    
    def _build_route(self):
        """Build interpolated route with all geometric properties."""
        # Step 1: Smooth corners first
        smoothed_nodes = self._smooth_corners(self.node_positions)
        
        # Step 2: Create centerline path from smoothed nodes
        centerline_points = []
        
        for i in range(len(smoothed_nodes) - 1):
            p1 = smoothed_nodes[i]
            p2 = smoothed_nodes[i + 1]
            
            # Compute segment length
            segment_vec = p2 - p1
            segment_length = np.linalg.norm(segment_vec)
            
            if segment_length < 1e-6:
                continue
            
            # Number of interpolation points for this segment
            n_points = max(2, int(np.ceil(segment_length / self.interpolation_step)))
            
            # Interpolate along segment (exclude last point to avoid duplication)
            for j in range(n_points - 1):
                alpha = j / (n_points - 1)
                point = p1 + alpha * segment_vec
                centerline_points.append(point)
        
        # Add final point
        centerline_points.append(smoothed_nodes[-1])
        
        centerline_points = np.array(centerline_points)
        
        # Step 2: Compute tangent vectors
        tangents = []
        for i in range(len(centerline_points)):
            if i == 0:
                # Forward difference
                tangent = centerline_points[i + 1] - centerline_points[i]
            elif i == len(centerline_points) - 1:
                # Backward difference
                tangent = centerline_points[i] - centerline_points[i - 1]
            else:
                # Central difference
                tangent = centerline_points[i + 1] - centerline_points[i - 1]
            
            # Normalize
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-9:
                tangent = tangent / tangent_norm
            else:
                tangent = np.array([1.0, 0.0])
            
            tangents.append(tangent)
        
        tangents = np.array(tangents)
        
        # Step 3: Compute normal vectors (perpendicular to tangent, pointing left)
        # For left-hand driving, left is the "inside" lane
        # Normal vector is 90° counterclockwise rotation of tangent
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]  # x component
        normals[:, 1] = tangents[:, 0]   # y component
        
        # Step 4: Compute arc-length parameterization
        s_values = np.zeros(len(centerline_points))
        for i in range(1, len(centerline_points)):
            segment_length = np.linalg.norm(centerline_points[i] - centerline_points[i - 1])
            s_values[i] = s_values[i - 1] + segment_length
        
        self.total_length = s_values[-1]
        
        # Step 5: NO LANE OFFSET - Use centerline directly for direction-agnostic behavior
        # FIXED: Removing lane offset makes the agent work in both directions
        # Agent learns to follow the centerline regardless of direction
        lane_offset = 0.0  # Changed from self.lane_width / 2.0
        left_lane_points = centerline_points + lane_offset * normals
        
        # Step 6: Compute curvature
        curvature = self._compute_curvature(tangents, s_values)
        
        # Store results
        self.route_points = left_lane_points
        self.route_tangents = tangents
        self.route_normals = normals
        self.route_s = s_values
        self.route_curvature = curvature
    
    def _compute_curvature(self, tangents: np.ndarray, s_values: np.ndarray) -> np.ndarray:
        """
        Compute curvature from tangent vectors.
        
        Curvature κ = |dθ/ds| where θ is the heading angle.
        
        Args:
            tangents: Nx2 array of tangent vectors
            s_values: Nx1 array of arc-length values
            
        Returns:
            Nx1 array of curvature values
        """
        curvature = np.zeros(len(tangents))
        
        for i in range(len(tangents)):
            if i == 0 or i == len(tangents) - 1:
                curvature[i] = 0.0
            else:
                # Compute heading angles
                theta_prev = np.arctan2(tangents[i - 1][1], tangents[i - 1][0])
                theta_next = np.arctan2(tangents[i + 1][1], tangents[i + 1][0])
                
                # Angle difference
                d_theta = theta_next - theta_prev
                
                # Normalize to [-pi, pi]
                while d_theta > np.pi:
                    d_theta -= 2 * np.pi
                while d_theta < -np.pi:
                    d_theta += 2 * np.pi
                
                # Arc length difference
                ds = s_values[i + 1] - s_values[i - 1]
                
                if ds > 1e-9:
                    curvature[i] = abs(d_theta) / ds
                else:
                    curvature[i] = 0.0
        
        return curvature
    
    def project_point(self, point: np.ndarray) -> Tuple[int, float, float, float]:
        """
        Project a point onto the route.
        
        Args:
            point: [x, y] position to project
            
        Returns:
            (closest_idx, s_proj, lateral_deviation, heading_at_projection)
            - closest_idx: index of closest route point
            - s_proj: arc-length at projection
            - lateral_deviation: signed distance from route (positive = right of route)
            - heading_at_projection: heading angle of route at projection point
        """
        # Find closest point on route
        distances = np.linalg.norm(self.route_points - point, axis=1)
        closest_idx = np.argmin(distances)
        
        # Arc-length at projection
        s_proj = self.route_s[closest_idx]
        
        # Lateral deviation (signed)
        # Positive means right of route, negative means left
        vec_to_point = point - self.route_points[closest_idx]
        lateral_deviation = np.dot(vec_to_point, self.route_normals[closest_idx])
        
        # Heading at projection
        heading = np.arctan2(self.route_tangents[closest_idx][1], 
                            self.route_tangents[closest_idx][0])
        
        return closest_idx, s_proj, lateral_deviation, heading
    
    def get_curvature_ahead(self, current_idx: int, lookahead: int = 10) -> float:
        """
        Get average curvature ahead of current position.
        
        Args:
            current_idx: current route index
            lookahead: number of points to look ahead
            
        Returns:
            Average curvature ahead
        """
        end_idx = min(current_idx + lookahead, len(self.route_curvature) - 1)
        if end_idx <= current_idx:
            return 0.0
        
        return np.mean(self.route_curvature[current_idx:end_idx])
    
    def get_centerline_points(self, node_positions: List[np.ndarray]) -> np.ndarray:
        """
        Get centerline points (without lane offset) for rendering.
        
        Args:
            node_positions: List of node positions
            
        Returns:
            Nx2 array of centerline points
        """
        centerline_points = []
        
        for i in range(len(node_positions) - 1):
            p1 = node_positions[i]
            p2 = node_positions[i + 1]
            
            segment_vec = p2 - p1
            segment_length = np.linalg.norm(segment_vec)
            
            n_points = max(2, int(np.ceil(segment_length / self.interpolation_step)))
            
            for j in range(n_points - 1):
                alpha = j / (n_points - 1)
                point = p1 + alpha * segment_vec
                centerline_points.append(point)
        
        centerline_points.append(node_positions[-1])
        
        return np.array(centerline_points)
