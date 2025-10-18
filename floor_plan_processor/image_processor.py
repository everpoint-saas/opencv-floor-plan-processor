"""
Image preprocessing module for architectural floor plans
Handles thresholding, morphological operations, and filtering
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Handles image preprocessing operations
    """
    
    def process(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Apply preprocessing steps to image
        
        Args:
            image: Input grayscale image
            params: Processing parameters from workflow
            
        Returns:
            Processed binary image
        """
        if image is None:
            logger.error("Input image is None")
            return None
        
        # Ensure grayscale
        if len(image.shape) == 3:
            logger.warning("Converting color image to grayscale")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        logger.debug("Starting image preprocessing")
        
        # Apply CLAHE if enabled
        if params.get('use_clahe', False):
            logger.debug("Applying CLAHE")
            clahe = cv2.createCLAHE(
                clipLimit=params.get('clahe_clip_limit', 2.0),
                tileGridSize=(params.get('clahe_tile_size', 8), params.get('clahe_tile_size', 8))
            )
            gray = clahe.apply(gray)
        
        # Apply denoising
        denoise_strength = params.get('denoise_strength', 0)
        if denoise_strength == 1:  # Light median
            logger.debug("Applying light median blur")
            gray = cv2.medianBlur(gray, 3)
        elif denoise_strength == 2:  # Medium median
            logger.debug("Applying medium median blur")
            gray = cv2.medianBlur(gray, 5)
        elif denoise_strength == 3:  # Strong NLM
            logger.debug("Applying NLM denoising")
            gray = cv2.fastNlMeansDenoising(
                gray, None,
                h=params.get('nlm_h', 10),
                templateWindowSize=params.get('nlm_template_size', 7),
                searchWindowSize=params.get('nlm_search_size', 21)
            )
        
        # Apply blur if specified
        blur_size = params.get('blur_size', 1)
        if blur_size > 1:
            blur_ksize = blur_size if blur_size % 2 == 1 else blur_size + 1
            logger.debug(f"Applying Gaussian blur with kernel size {blur_ksize}")
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        
        # Apply thresholding
        logger.debug("Applying thresholding")
        binary = self._apply_threshold(gray, params)
        
        if binary is None:
            logger.error("Thresholding failed")
            return None
        
        # Remove hatching if enabled
        if params.get('remove_hatching', False):
            logger.debug("Removing hatching patterns")
            binary = self._remove_hatching(binary, params.get('hatching_kernel_size', 5))
        
        # Apply morphological operations
        morph_ops = params.get('morph_operations', [])
        if morph_ops:
            logger.debug(f"Applying {len(morph_ops)} morphological operations")
            binary = self._apply_morphology(binary, morph_ops)
        
        # Apply Canny edge detection if enabled
        if params.get('use_canny', False):
            logger.debug("Applying Canny edge detection")
            binary = cv2.Canny(
                binary,
                params.get('canny_low', 50),
                params.get('canny_high', 150)
            )
        
        # Apply additional filters
        if params.get('use_sharpen', False) and not params.get('use_canny', False):
            logger.debug("Applying sharpening filter")
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            binary = cv2.filter2D(binary, -1, kernel)
        
        if params.get('use_thinning', False) and not params.get('use_canny', False):
            logger.debug("Applying thinning")
            binary = self._apply_thinning(binary)
        
        logger.info("Image preprocessing completed")
        return binary
    
    def _apply_threshold(self, gray: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply thresholding based on method specified in params"""
        thresh_method = params.get('thresh_method', 0)
        
        if thresh_method == 0:  # Binary
            _, binary = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_BINARY)
        elif thresh_method == 1:  # Binary Inverted
            _, binary = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_BINARY_INV)
        elif thresh_method == 2:  # Truncated
            _, binary = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_TRUNC)
        elif thresh_method == 3:  # To Zero
            _, binary = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_TOZERO)
        elif thresh_method == 4:  # Adaptive Mean
            block_size = params.get('adaptive_block_size', 11)
            if block_size % 2 == 0:
                block_size += 1
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                block_size, params.get('adaptive_c', 2)
            )
        elif thresh_method == 5:  # Adaptive Gaussian
            block_size = params.get('adaptive_block_size', 11)
            if block_size % 2 == 0:
                block_size += 1
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                block_size, params.get('adaptive_c', 2)
            )
        elif thresh_method == 6:  # Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Default to basic thresholding
            _, binary = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_BINARY)
        
        return binary
    
    def _remove_hatching(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Remove horizontal and vertical hatching patterns"""
        # Ensure binary
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Create kernels
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        
        # Apply morphological opening
        horizontal_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine results
        result = cv2.bitwise_and(horizontal_opened, vertical_opened)
        
        return result
    
    def _apply_morphology(self, binary: np.ndarray, operations: List[Dict[str, Any]]) -> np.ndarray:
        """Apply a sequence of morphological operations"""
        result = binary.copy()
        
        for op in operations:
            morph_type = op.get('morph_type', 0)
            if morph_type == 0:  # None
                continue
            
            # Create kernel
            kernel_size = op.get('morph_size', 3)
            kernel_shape = op.get('kernel_shape', 0)
            
            if kernel_shape == 0:  # Rectangle
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            elif kernel_shape == 1:  # Ellipse
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            elif kernel_shape == 2:  # Cross
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            else:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            iterations = op.get('morph_iterations', 1)
            
            # Apply operation
            if morph_type == 1:  # Dilate
                result = cv2.dilate(result, kernel, iterations=iterations)
            elif morph_type == 2:  # Erode
                result = cv2.erode(result, kernel, iterations=iterations)
            elif morph_type == 3:  # Open
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iterations)
            elif morph_type == 4:  # Close
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            elif morph_type == 5:  # Gradient
                result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
            elif morph_type == 6:  # Top Hat
                result = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
            elif morph_type == 7:  # Black Hat
                result = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
        
        return result
    
    def _apply_thinning(self, image: np.ndarray) -> np.ndarray:
        """Apply thinning operation"""
        # Ensure binary
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Check if ximgproc is available
        try:
            # OpenCV's thinning (requires opencv-contrib-python)
            thinned = cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        except AttributeError:
            logger.warning("cv2.ximgproc not available. Using simple erosion instead.")
            # Fallback to simple erosion
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            thinned = cv2.erode(binary, kernel, iterations=1)
        
        return thinned