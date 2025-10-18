"""
Room extraction module for finding rooms within building outlines
"""

import cv2
import numpy as np
import logging
import os
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class RoomExtractor:
    """
    Extracts individual rooms from within a building outline
    """
    
    def extract_rooms_from_roi(self, image: np.ndarray, building_outline: np.ndarray, 
                              params: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Extract rooms from within the building outline ROI
        
        Args:
            image: Binary preprocessed image
            building_outline: Building outline contour
            params: Extraction parameters
            
        Returns:
            Tuple of (room_contours, debug_info)
        """
        logger.info("Starting room extraction from ROI")
        
        # Get parameters
        min_area = params.get('min_area', 10000)
        max_area = params.get('max_area', 100000)
        save_debug = params.get('save_debug', True)
        roi_margin = params.get('roi_margin', 0)
        
        # Parameters with defaults
        wall_closing_kernel_size = params.get('wall_closing_kernel_size', 1)
        space_opening_kernel_size = params.get('space_opening_kernel_size', 1)
        dist_transform_threshold_ratio = params.get('dist_transform_threshold_ratio', 0.1)
        room_simplify_epsilon_factor = params.get('room_simplify_epsilon_factor', 0.02)
        min_area_for_missed_rooms_ratio = params.get('min_area_for_missed_rooms_ratio', 0.5)
        
        debug_info = {}
        
        # Create debug directory if needed
        if save_debug:
            debug_dir = 'debug_room_extraction'
            os.makedirs(debug_dir, exist_ok=True)
        
        # Step 1: Extract ROI from building outline
        x, y, w, h = cv2.boundingRect(building_outline)
        
        # Apply margin
        x = max(0, x - roi_margin)
        y = max(0, y - roi_margin)
        w = min(image.shape[1] - x, w + 2 * roi_margin)
        h = min(image.shape[0] - y, h + 2 * roi_margin)
        
        roi = image[y:y+h, x:x+w].copy()
        
        if save_debug:
            cv2.imwrite(os.path.join(debug_dir, '01_roi_original_with_margin.png'), roi)
        
        # Step 2: Create binary ROI (invert if needed so space=255, wall=0)
        mean_val = np.mean(roi)
        if mean_val > 127:  # Mostly white, need to invert
            roi_binary = cv2.bitwise_not(roi)
        else:
            roi_binary = roi.copy()
        
        if save_debug:
            cv2.imwrite(os.path.join(debug_dir, '02_binary_roi_initial.png'), roi_binary)
        
        # Step 3: Wall closing - improve wall continuity
        if wall_closing_kernel_size > 0:
            kernel = np.ones((wall_closing_kernel_size, wall_closing_kernel_size), np.uint8)
            roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, kernel)
            
            if save_debug:
                cv2.imwrite(os.path.join(debug_dir, '03_binary_roi_after_wall_closing.png'), roi_binary)
        
        # Step 4: Space opening - clean up space areas
        if space_opening_kernel_size > 0:
            kernel = np.ones((space_opening_kernel_size, space_opening_kernel_size), np.uint8)
            roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel)
            
            if save_debug:
                cv2.imwrite(os.path.join(debug_dir, '04_binary_roi_after_space_opening.png'), roi_binary)
        
        # Step 5: Distance transform to find room centers
        dist_transform = cv2.distanceTransform(roi_binary, cv2.DIST_L2, 5)
        
        # Normalize for visualization
        dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if save_debug:
            cv2.imwrite(os.path.join(debug_dir, '05_dist_transform_normalized.png'), dist_normalized)
        
        # Step 6: Find seed points for rooms
        threshold = dist_transform.max() * dist_transform_threshold_ratio
        _, seed_mask = cv2.threshold(dist_transform, threshold, 255, cv2.THRESH_BINARY)
        seed_mask = seed_mask.astype(np.uint8)
        
        if save_debug:
            cv2.imwrite(os.path.join(debug_dir, '06_seed_mask.png'), seed_mask)
        
        # Find connected components as seeds
        num_labels, labels = cv2.connectedComponents(seed_mask)
        
        # Step 7: Flood fill from each seed to extract rooms
        room_contours = []
        overall_mask = np.zeros(roi_binary.shape, dtype=np.uint8)
        
        for label_id in range(1, num_labels):  # Skip background (0)
            # Get seed point
            seed_points = np.where(labels == label_id)
            if len(seed_points[0]) == 0:
                continue
            
            seed_y, seed_x = seed_points[0][0], seed_points[1][0]
            
            # Create mask for flood fill
            h_ff, w_ff = roi_binary.shape
            mask = np.zeros((h_ff + 2, w_ff + 2), dtype=np.uint8)
            
            # Copy current binary image
            ff_image = roi_binary.copy()
            
            # Flood fill
            cv2.floodFill(ff_image, mask, (seed_x, seed_y), 128)
            
            # Extract filled region
            room_mask = (ff_image == 128).astype(np.uint8) * 255
            
            if save_debug:
                cv2.imwrite(os.path.join(debug_dir, f'07_ff_mask_seed{label_id}.png'), room_mask)
            
            # Find contours of the room
            room_contours_temp, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in room_contours_temp:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    # Simplify contour
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = room_simplify_epsilon_factor * perimeter
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Adjust coordinates to original image space
                    adjusted_contour = simplified + np.array([x, y])
                    room_contours.append(adjusted_contour)
                    
                    # Update overall mask
                    cv2.drawContours(overall_mask, [contour], -1, 255, -1)
        
        # Step 8: Find missed rooms in unfilled areas
        unfilled_mask = cv2.bitwise_and(roi_binary, cv2.bitwise_not(overall_mask))
        
        if save_debug:
            cv2.imwrite(os.path.join(debug_dir, '08_unfilled_area_mask_for_missed.png'), unfilled_mask)
        
        # Process unfilled areas
        if np.any(unfilled_mask):
            # Apply morphological operations to connect small gaps
            kernel = np.ones((3, 3), np.uint8)
            unfilled_processed = cv2.morphologyEx(unfilled_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            unfilled_processed = cv2.morphologyEx(unfilled_processed, cv2.MORPH_OPEN, kernel)
            
            # Find contours in unfilled areas
            missed_contours, _ = cv2.findContours(unfilled_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_area_missed = min_area * min_area_for_missed_rooms_ratio
            
            for contour in missed_contours:
                area = cv2.contourArea(contour)
                if min_area_missed <= area <= max_area:
                    # Simplify and add
                    perimeter = cv2.arcLength(contour, True)
                    epsilon = room_simplify_epsilon_factor * perimeter
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    adjusted_contour = simplified + np.array([x, y])
                    room_contours.append(adjusted_contour)
        
        # Create final visualization
        if save_debug:
            final_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for i, room in enumerate(room_contours):
                color = ((i * 67) % 256, (i * 123) % 256, (i * 199) % 256)
                cv2.drawContours(final_vis, [room], -1, color, 2)
            cv2.imwrite(os.path.join(debug_dir, '11_final_all_rooms_overlay.png'), final_vis)
        
        logger.info(f"Extracted {len(room_contours)} rooms")
        
        debug_info['num_rooms'] = len(room_contours)
        debug_info['roi_bounds'] = (x, y, w, h)
        
        return room_contours, debug_info