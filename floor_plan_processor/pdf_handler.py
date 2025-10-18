"""
PDF handling module for loading and rendering PDF pages
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class PDFHandler:
    """
    Handles PDF loading and page rendering
    """
    
    def __init__(self, dpi: int = 600):
        """
        Initialize PDF handler
        
        Args:
            dpi: Resolution for rendering PDF pages
        """
        self.dpi = dpi
        self.pdf_document = None
        self.total_pages = 0
        logger.info(f"PDFHandler initialized with DPI: {dpi}")
    
    def load_pdf(self, pdf_path: str) -> Tuple[bool, int]:
        """
        Load a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (success, total_pages)
        """
        try:
            self.pdf_document = fitz.open(pdf_path)
            self.total_pages = len(self.pdf_document)
            logger.info(f"PDF loaded successfully: {self.total_pages} pages")
            return True, self.total_pages
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            return False, 0
    
    def get_page_image(self, page_number: int) -> Optional[np.ndarray]:
        """
        Render a PDF page as an image
        
        Args:
            page_number: Page number to render (0-based)
            
        Returns:
            Rendered page as numpy array (BGR format) or None if failed
        """
        if self.pdf_document is None:
            logger.error("No PDF document loaded")
            return None
        
        if page_number < 0 or page_number >= self.total_pages:
            logger.error(f"Invalid page number: {page_number}")
            return None
        
        try:
            # Get the page
            page = self.pdf_document[page_number]
            
            # Calculate zoom factor for desired DPI
            zoom = self.dpi / 72.0  # PDF default is 72 DPI
            mat = fitz.Matrix(zoom, zoom)
            
            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array
            img_data = np.frombuffer(pix.samples, np.uint8)
            img_data = img_data.reshape(pix.height, pix.width, pix.n)
            
            # Convert from RGB to BGR for OpenCV
            if pix.n == 3:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            elif pix.n == 4:
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = img_data
            
            logger.info(f"Page {page_number} rendered successfully: {img_bgr.shape}")
            return img_bgr
            
        except Exception as e:
            logger.error(f"Failed to render page {page_number}: {e}")
            return None
    
    def close(self):
        """Close the PDF document"""
        if self.pdf_document:
            self.pdf_document.close()
            self.pdf_document = None
            logger.info("PDF document closed")