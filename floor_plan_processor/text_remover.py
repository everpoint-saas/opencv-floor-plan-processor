"""
Text detection and removal module using OCR and inpainting
"""

import cv2
import numpy as np
import easyocr
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class TextRemover:
    """
    Handles text detection and removal from floor plans
    """
    
    def __init__(self, languages: List[str] = ['ko', 'en']):
        """
        Initialize text remover with OCR languages
        
        Args:
            languages: List of language codes for OCR
        """
        self.languages = languages
        self.ocr_reader = None
        self.text_regions = []
        logger.info(f"TextRemover initialized with languages: {languages}")
    
    def _initialize_ocr(self):
        """Initialize EasyOCR reader if not already done"""
        if self.ocr_reader is None:
            try:
                logger.info("Initializing EasyOCR reader...")
                self.ocr_reader = easyocr.Reader(self.languages, gpu=False)
                logger.info("EasyOCR reader initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OCR: {e}")
                self.ocr_reader = None
    
    def detect_text(self, image: np.ndarray, enhance: bool = True) -> List[Tuple]:
        """
        Detect text regions in the image
        
        Args:
            image: Input image (grayscale or color)
            enhance: Whether to enhance image before OCR
            
        Returns:
            List of text regions (bounding_box, text, confidence)
        """
        self._initialize_ocr()
        
        if self.ocr_reader is None:
            logger.error("OCR reader not available")
            self.text_regions = []
            return []
        
        try:
            logger.info("Starting text detection...")
            
            # Prepare image for OCR
            if enhance:
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)
                ocr_image = enhanced_gray
                logger.debug("Using enhanced image for OCR")
            else:
                ocr_image = image
                logger.debug("Using original image for OCR")
            
            # Run OCR
            ocr_results = self.ocr_reader.readtext(ocr_image, detail=1, paragraph=False)
            
            self.text_regions = ocr_results
            logger.info(f"Text detection complete. Found {len(self.text_regions)} regions")
            
            return self.text_regions
            
        except Exception as e:
            logger.error(f"Error during text detection: {e}")
            self.text_regions = []
            return []
    
    def remove_text(self, image: np.ndarray, padding: int = 5, 
                   inpaint_radius: int = 5, inpaint_method: str = 'ns') -> np.ndarray:
        """
        Remove detected text from image using inpainting
        
        Args:
            image: Input image
            padding: Padding around text regions
            inpaint_radius: Radius for inpainting
            inpaint_method: Inpainting method ('ns' or 'telea')
            
        Returns:
            Image with text removed
        """
        if not self.text_regions:
            logger.info("No text regions to remove")
            return image
        
        logger.info(f"Removing {len(self.text_regions)} text regions")
        
        # Create mask for text regions
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for region in self.text_regions:
            bbox_points = region[0]  # Bounding box coordinates
            
            # Convert to integer coordinates
            points = np.array(bbox_points, dtype=np.int32)
            
            # Add padding
            if padding > 0:
                # Calculate center and expand
                center = points.mean(axis=0)
                points = ((points - center) * (1 + padding/100) + center).astype(np.int32)
            
            # Draw filled polygon on mask
            cv2.fillPoly(mask, [points], 255)
        
        # Apply inpainting
        method = cv2.INPAINT_NS if inpaint_method == 'ns' else cv2.INPAINT_TELEA
        
        if len(image.shape) == 3:
            # Color image
            result = cv2.inpaint(image, mask, inpaint_radius, method)
        else:
            # Grayscale image
            result = cv2.inpaint(image, mask, inpaint_radius, method)
        
        logger.info("Text removal completed")
        return result
    
    def create_text_mask(self, image_shape: Tuple[int, int], padding: int = 5) -> np.ndarray:
        """
        Create a mask showing text regions
        
        Args:
            image_shape: Shape of the image (height, width)
            padding: Padding around text regions
            
        Returns:
            Binary mask with text regions as white
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for region in self.text_regions:
            bbox_points = region[0]
            points = np.array(bbox_points, dtype=np.int32)
            
            if padding > 0:
                center = points.mean(axis=0)
                points = ((points - center) * (1 + padding/100) + center).astype(np.int32)
            
            cv2.fillPoly(mask, [points], 255)
        
        return mask