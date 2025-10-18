"""
Geometry optimization module for cleaning up room contours
"""

import cv2
import numpy as np
import logging
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class GeometryOptimizer:
    """
    Optimizes room geometries through vertex snapping and simplification
    """
    
    def __init__(self, snap_threshold_pixels: float = 5.0, 
                 min_vertices_for_polygon: int = 3,
                 min_edge_length_pixels: float = 1.0):
        """
        Initialize geometry optimizer
        
        Args:
            snap_threshold_pixels: Distance threshold for vertex snapping
            min_vertices_for_polygon: Minimum vertices for valid polygon
            min_edge_length_pixels: Minimum edge length to keep
        """
        self.snap_threshold = float(snap_threshold_pixels)
        self.min_vertices = int(min_vertices_for_polygon)
        self.min_edge_length = float(min_edge_length_pixels)
        
        logger.info(f"GeometryOptimizer initialized: snap={self.snap_threshold}px, "
                   f"min_vertices={self.min_vertices}, min_edge={self.min_edge_length}px")
    
    def optimize_room_geometries(self, room_contours: List[np.ndarray], 
                               original_image: Optional[np.ndarray] = None,
                               roi_offset: Tuple[int, int] = (0, 0)) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Optimize room geometries through snapping and cleanup
        
        Args:
            room_contours: List of room contours
            original_image: Original image for visualization
            roi_offset: Offset for visualization
            
        Returns:
            Tuple of (optimized_contours, visualization_image)
        """
        logger.info(f"Starting geometry optimization for {len(room_contours)} rooms")
        
        if not room_contours:
            logger.warning("No room contours to optimize")
            vis_img = self._get_empty_visualization(original_image)
            return [], vis_img
        
        # Collect all vertices with room and vertex indices
        all_vertices = self._collect_all_vertices(room_contours)
        
        # Group vertices that should snap together
        vertex_groups = self._group_snap_vertices(all_vertices)
        
        # Update vertices with group centroids
        self._update_vertices_with_centroids(all_vertices, vertex_groups)
        
        # Reconstruct and validate contours
        optimized_contours = self._reconstruct_contours(all_vertices, room_contours)
        
        logger.info(f"Optimization complete. {len(optimized_contours)} valid rooms")
        
        # Create visualization
        visualization = None
        if original_image is not None:
            visualization = self._create_visualization(
                original_image, room_contours, optimized_contours, roi_offset
            )
        
        return optimized_contours, visualization
    
    def _collect_all_vertices(self, room_contours: List[np.ndarray]) -> List[Dict]:
        """Collect all vertices with metadata"""
        all_vertices = []
        
        for room_idx, contour in enumerate(room_contours):
            if not isinstance(contour, np.ndarray) or contour.ndim < 2:
                continue
            
            for vertex_idx, point_wrapper in enumerate(contour):
                # Handle OpenCV contour format [[x, y]]
                if isinstance(point_wrapper, (list, np.ndarray)) and len(point_wrapper) > 0:
                    if isinstance(point_wrapper[0], (list, np.ndarray)) and len(point_wrapper[0]) == 2:
                        x, y = float(point_wrapper[0][0]), float(point_wrapper[0][1])
                        
                        vertex_info = {
                            'x_orig': x,
                            'y_orig': y,
                            'x_new': x,
                            'y_new': y,
                            'room_idx': room_idx,
                            'vertex_idx': vertex_idx,
                            'global_id': len(all_vertices),
                            'snap_group': -1
                        }
                        all_vertices.append(vertex_info)
        
        return all_vertices
    
    def _group_snap_vertices(self, all_vertices: List[Dict]) -> Dict[int, List[int]]:
        """Group vertices that should snap together using Union-Find"""
        # Initialize Union-Find structure
        parent = {v['global_id']: v['global_id'] for v in all_vertices}
        
        def find(vertex_id):
            if parent[vertex_id] != vertex_id:
                parent[vertex_id] = find(parent[vertex_id])
            return parent[vertex_id]
        
        def union(id1, id2):
            root1, root2 = find(id1), find(id2)
            if root1 != root2:
                parent[root2] = root1
        
        # Find vertices within snap threshold
        for i in range(len(all_vertices)):
            for j in range(i + 1, len(all_vertices)):
                v1, v2 = all_vertices[i], all_vertices[j]
                
                # Skip if already in same group
                if find(v1['global_id']) == find(v2['global_id']):
                    continue
                
                # Calculate distance
                dist_sq = (v1['x_orig'] - v2['x_orig'])**2 + (v1['y_orig'] - v2['y_orig'])**2
                
                if dist_sq < self.snap_threshold**2:
                    union(v1['global_id'], v2['global_id'])
        
        # Collect final groups
        groups = defaultdict(list)
        for vertex in all_vertices:
            root = find(vertex['global_id'])
            vertex['snap_group'] = root
            groups[root].append(vertex['global_id'])
        
        return groups
    
    def _update_vertices_with_centroids(self, all_vertices: List[Dict], groups: Dict[int, List[int]]):
        """Update vertex positions with group centroids"""
        vertex_map = {v['global_id']: v for v in all_vertices}
        
        for group_id, member_ids in groups.items():
            if not member_ids:
                continue
            
            # Calculate centroid
            sum_x = sum(vertex_map[vid]['x_orig'] for vid in member_ids)
            sum_y = sum(vertex_map[vid]['y_orig'] for vid in member_ids)
            centroid_x = sum_x / len(member_ids)
            centroid_y = sum_y / len(member_ids)
            
            # Update all vertices in group
            for vertex_id in member_ids:
                vertex_map[vertex_id]['x_new'] = centroid_x
                vertex_map[vertex_id]['y_new'] = centroid_y
    
    def _reconstruct_contours(self, all_vertices: List[Dict], original_contours: List[np.ndarray]) -> List[np.ndarray]:
        """Reconstruct contours from optimized vertices"""
        # Group vertices by room
        room_vertices = defaultdict(list)
        for vertex in all_vertices:
            room_vertices[vertex['room_idx']].append(vertex)
        
        optimized_contours = []
        
        for room_idx in range(len(original_contours)):
            if room_idx not in room_vertices:
                continue
            
            # Sort vertices by original order
            vertices = sorted(room_vertices[room_idx], key=lambda v: v['vertex_idx'])
            
            # Extract points
            points = np.array([[v['x_new'], v['y_new']] for v in vertices], dtype=np.float32)
            
            if len(points) < self.min_vertices:
                continue
            
            # Clean up redundant points
            cleaned_points = self._clean_polygon(points)
            
            if len(cleaned_points) >= self.min_vertices:
                # Convert to OpenCV contour format
                contour = np.array([[[int(round(p[0])), int(round(p[1]))]] 
                                   for p in cleaned_points], dtype=np.int32)
                optimized_contours.append(contour)
        
        return optimized_contours
    
    def _clean_polygon(self, points: np.ndarray) -> np.ndarray:
        """Remove redundant points from polygon"""
        if len(points) < 2:
            return points
        
        cleaned = [points[0]]
        
        for i in range(1, len(points)):
            current = points[i]
            previous = cleaned[-1]
            
            # Skip if too close to previous point
            dist_sq = (current[0] - previous[0])**2 + (current[1] - previous[1])**2
            if dist_sq < self.min_edge_length**2:
                continue
            
            cleaned.append(current)
        
        # Check closing edge
        if len(cleaned) > 1:
            dist_sq = (cleaned[-1][0] - cleaned[0][0])**2 + (cleaned[-1][1] - cleaned[0][1])**2
            if dist_sq < self.min_edge_length**2:
                cleaned.pop()
        
        return np.array(cleaned, dtype=np.float32)
    
    def _get_empty_visualization(self, base_image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Get empty visualization image"""
        if base_image is None:
            return None
        
        if len(base_image.shape) == 2:
            return cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        return base_image.copy()
    
    def _create_visualization(self, base_image: np.ndarray, original_contours: List[np.ndarray],
                            optimized_contours: List[np.ndarray], offset: Tuple[int, int]) -> np.ndarray:
        """Create visualization showing original vs optimized contours"""
        vis_img = self._get_empty_visualization(base_image)
        if vis_img is None:
            return None
        
        # Apply offset function
        def apply_offset(contour, off):
            if contour is None or len(contour) == 0:
                return None
            return contour + np.array([off[0], off[1]]).reshape(-1, 1, 2)
        
        # Draw original contours in green
        for contour in original_contours:
            contour_offset = apply_offset(contour, offset)
            if contour_offset is not None:
                cv2.drawContours(vis_img, [contour_offset], -1, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Draw optimized contours in red with vertices
        for contour in optimized_contours:
            contour_offset = apply_offset(contour, offset)
            if contour_offset is not None and len(contour_offset) > 0:
                cv2.drawContours(vis_img, [contour_offset], -1, (255, 0, 0), 3, cv2.LINE_AA)
                
                # Draw vertices
                for point in contour_offset:
                    cv2.circle(vis_img, tuple(point[0]), 3, (0, 0, 255), -1)
        
        return vis_img