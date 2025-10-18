"""
Contour extraction module for finding building outlines
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ContourExtractor:
    """
    Extracts and filters contours to find building outlines
    """
    
    def find_contours(self, image: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """
        Find all contours in the image
        
        Args:
            image: Binary image
            
        Returns:
            Tuple of (contours, hierarchy)
        """
        # Ensure binary image
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        logger.info(f"Found {len(contours)} contours")
        return contours, hierarchy
    
    def find_building_outline(self, image: np.ndarray, contours: List, 
                            hierarchy: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Find the main building outline from contours
        
        Args:
            image: Source image for shape reference
            contours: List of contours
            hierarchy: Contour hierarchy
            params: Filtering parameters
            
        Returns:
            Tuple of (visualization_image, building_outline)
        """
        if hierarchy is None or len(hierarchy) == 0:
            logger.warning("No hierarchy information available")
            return self._create_empty_visualization(image), None
        
        # Flatten hierarchy if needed
        if len(hierarchy.shape) == 3 and hierarchy.shape[0] == 1:
            hierarchy = hierarchy[0]
        
        # Filter and select outline
        filtered_contours, selected_outline = self._filter_and_select_outline(
            image.shape, contours, hierarchy, params
        )
        
        # Create visualization
        visualization = self._create_visualization(image, filtered_contours, selected_outline)
        
        return visualization, selected_outline
    
    def _filter_and_select_outline(self, image_shape: Tuple, contours: List, 
                                  hierarchy: np.ndarray, params: Dict[str, Any]) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Filter contours and select the main building outline
        """
        image_height, image_width = image_shape[:2]
        image_area = float(image_height * image_width)
        
        # Get parameters
        min_area_ratio = params.get('min_area_ratio', 0.02)
        min_area_threshold = image_area * min_area_ratio
        
        max_area_ratio = params.get('max_area_ratio_to_image', 1.0)
        max_area_threshold = image_area * max_area_ratio
        
        min_children = params.get('min_children_count', 15)
        approx_epsilon = params.get('approx_epsilon', 0.005)
        min_vertices = params.get('min_vertices', 4)
        max_vertices = params.get('max_vertices', 100)
        min_aspect_ratio = params.get('min_aspect_ratio', 0.1)
        max_aspect_ratio = params.get('max_aspect_ratio', 10.0)
        min_solidity = params.get('min_solidity', 0.07)
        border_margin = params.get('border_margin', 15)
        
        filtered_contours = []
        candidates = []
        
        logger.info(f"Filtering {len(contours)} contours...")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < min_area_threshold:
                continue
            
            # Get contour properties
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, approx_epsilon * perimeter, True)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate metrics
            vertices_count = len(approx)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Count children
            children_count = 0
            for j in range(len(hierarchy)):
                if hierarchy[j][3] == i:  # Parent index
                    children_count += 1
            
            # Check if touches border
            touches_border = (x <= border_margin or y <= border_margin or 
                            x + w >= image_width - border_margin or 
                            y + h >= image_height - border_margin)
            
            contour_info = {
                'id': i,
                'contour': contour,
                'contour_approx': approx,
                'area': area,
                'children': children_count,
                'vertices': vertices_count,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'bounding_rect': (x, y, w, h),
                'touches_border': touches_border
            }
            
            filtered_contours.append(contour_info)
            
            # Check if candidate for building outline
            if (min_vertices <= vertices_count <= max_vertices and
                min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
                solidity >= min_solidity and
                children_count >= min_children):
                candidates.append(contour_info)
        
        # Select best candidate
        selected_outline = None
        if candidates:
            # Sort by children count, then area
            candidates.sort(key=lambda c: (c['children'], c['area']), reverse=True)
            
            # Check area constraints
            max_sensible_area = image_area * 0.8  # Building shouldn't be more than 80% of image
            
            for candidate in candidates:
                if candidate['area'] <= max_sensible_area:
                    selected_outline = candidate['contour_approx']
                    logger.info(f"Selected outline: ID={candidate['id']}, "
                              f"Area={candidate['area']:.0f}, "
                              f"Children={candidate['children']}")
                    break
        
        if selected_outline is None:
            logger.warning("No suitable building outline found")
        
        return filtered_contours, selected_outline
    
    def _create_visualization(self, image: np.ndarray, filtered_contours: List[Dict], 
                            selected_outline: Optional[np.ndarray]) -> np.ndarray:
        """Create visualization of contour detection results"""
        # Convert to BGR for visualization
        if len(image.shape) == 2:
            vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = image.copy()
        
        # Draw all filtered contours
        for contour_info in filtered_contours:
            contour = contour_info['contour']
            cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 1)
            
            # Draw bounding box
            x, y, w, h = contour_info['bounding_rect']
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            
            # Draw ID
            cv2.putText(vis_img, f"ID:{contour_info['id']}", 
                       (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Highlight selected outline
        if selected_outline is not None:
            cv2.drawContours(vis_img, [selected_outline], -1, (255, 0, 255), 3)
        
        return vis_img
    
    def _create_empty_visualization(self, image: np.ndarray) -> np.ndarray:
        """Create empty visualization when no contours found"""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()