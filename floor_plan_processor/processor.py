"""
Main processor class for floor plan extraction
Integrates all processing steps with workflow configuration
"""

import os
import json
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List

from .pdf_handler import PDFHandler
from .image_processor import ImageProcessor
from .text_remover import TextRemover
from .contour_extractor import ContourExtractor
from .room_extractor import RoomExtractor
from .geometry_optimizer import GeometryOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FloorPlanProcessor:
    """
    Main class for processing floor plans from PDF to extracted rooms
    """
    
    def __init__(self, workflow_path: str = 'workflow.json'):
        """
        Initialize the processor with workflow configuration
        
        Args:
            workflow_path: Path to workflow configuration JSON file
        """
        self.workflow_path = workflow_path
        self.config = self._load_workflow()
        
        # Initialize components
        self.pdf_handler = PDFHandler(dpi=self.config.get('render_dpi', 600))
        self.image_processor = ImageProcessor()
        self.text_remover = TextRemover(
            languages=self.config.get('ocr_langs', ['ko', 'en'])
        )
        self.contour_extractor = ContourExtractor()
        self.room_extractor = RoomExtractor()
        self.geometry_optimizer = GeometryOptimizer()
        
        logger.info(f"FloorPlanProcessor initialized with workflow: {workflow_path}")
    
    def _load_workflow(self) -> Dict[str, Any]:
        """Load workflow configuration from JSON file"""
        try:
            with open(self.workflow_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Workflow loaded successfully from {self.workflow_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
            raise
    
    def process_pdf(self, pdf_path: str, page_number: int = 0) -> Dict[str, Any]:
        """
        Process a PDF file to extract floor plan rooms
        
        Args:
            pdf_path: Path to input PDF file
            page_number: Page number to process (0-based)
            
        Returns:
            Dictionary containing:
                - 'original_image': Original rendered image
                - 'processed_image': Preprocessed binary image
                - 'building_outline': Main building contour
                - 'rooms': List of room contours
                - 'optimized_rooms': Geometrically optimized room contours
                - 'visualization': Visualization image
        """
        logger.info(f"Starting processing for: {pdf_path}, page {page_number}")
        
        # Step 1: Load PDF and render page
        logger.info("Step 1: Loading PDF and rendering page")
        self.pdf_handler.load_pdf(pdf_path)
        original_image = self.pdf_handler.get_page_image(page_number)
        
        if original_image is None:
            raise ValueError(f"Failed to render page {page_number} from PDF")
        
        # Convert to grayscale if specified
        if self.config.get('use_grayscale', True) and len(original_image.shape) == 3:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = original_image.copy()
        
        # Step 2: Text detection and removal
        processed_image = gray_image.copy()
        if self.config.get('remove_text', True):
            logger.info("Step 2: Detecting and removing text")
            text_regions = self.text_remover.detect_text(
                gray_image, 
                enhance=self.config.get('enhance_ocr', True)
            )
            
            if text_regions:
                processed_image = self.text_remover.remove_text(
                    gray_image,
                    padding=self.config.get('text_padding', 5),
                    inpaint_radius=self.config.get('inpaint_radius', 5),
                    inpaint_method=self.config.get('inpaint_method', 'ns')
                )
        
        # Step 3: Image preprocessing
        logger.info("Step 3: Applying image preprocessing")
        processed_image = self.image_processor.process(processed_image, self.config)
        
        # Step 4: Extract building outline (ROI)
        logger.info("Step 4: Extracting building outline")
        contours, hierarchy = self.contour_extractor.find_contours(processed_image)
        
        building_outline = None
        if contours and hierarchy is not None:
            _, building_outline = self.contour_extractor.find_building_outline(
                processed_image, contours, hierarchy, self.config
            )
        
        # Step 5: Extract rooms from ROI
        rooms = []
        optimized_rooms = []
        room_visualization = None
        
        if self.config.get('extract_rooms', True) and building_outline is not None:
            logger.info("Step 5: Extracting rooms from building outline")
            
            room_params = {
                'min_area': self.config.get('min_room_area', 10000),
                'max_area': self.config.get('max_room_area', 100000),
                'save_debug': True  # Save debug images
            }
            
            rooms, debug_info = self.room_extractor.extract_rooms_from_roi(
                processed_image, building_outline, room_params
            )
            
            # Step 6: Optimize room geometries
            if rooms:
                logger.info("Step 6: Optimizing room geometries")
                optimized_rooms, room_visualization = self.geometry_optimizer.optimize_room_geometries(
                    rooms, original_image
                )
        
        # Create final visualization
        visualization = self._create_visualization(
            original_image, processed_image, building_outline, optimized_rooms
        )
        
        logger.info("Processing completed successfully")
        
        return {
            'original_image': original_image,
            'processed_image': processed_image,
            'building_outline': building_outline,
            'rooms': rooms,
            'optimized_rooms': optimized_rooms,
            'visualization': visualization,
            'room_visualization': room_visualization
        }
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image directly (without PDF loading)
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Same as process_pdf
        """
        logger.info("Processing image directly")
        
        # Convert to grayscale if needed
        if self.config.get('use_grayscale', True) and len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # Follow same processing steps as process_pdf
        # (Implementation similar to process_pdf but starting from image)
        # ... (similar code as above, starting from Step 2)
        
        return result
    
    def _create_visualization(self, original: np.ndarray, processed: np.ndarray, 
                            building_outline: Optional[np.ndarray], 
                            rooms: List[np.ndarray]) -> np.ndarray:
        """Create a visualization showing all processing results"""
        # Convert to BGR for visualization
        if len(original.shape) == 2:
            vis_base = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis_base = original.copy()
        
        # Draw building outline
        if building_outline is not None:
            cv2.drawContours(vis_base, [building_outline], -1, (0, 255, 0), 3)
        
        # Draw rooms
        if rooms:
            for i, room in enumerate(rooms):
                if room is not None and len(room) > 0:
                    # Use different colors for each room
                    color = (
                        (i * 67) % 256,
                        (i * 123) % 256,
                        (i * 199) % 256
                    )
                    cv2.drawContours(vis_base, [room], -1, color, 2)
        
        return vis_base
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """
        Save processing results to files
        
        Args:
            results: Dictionary from process_pdf or process_image
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save images
        if results.get('processed_image') is not None:
            cv2.imwrite(os.path.join(output_dir, 'processed.png'), results['processed_image'])
        
        if results.get('visualization') is not None:
            cv2.imwrite(os.path.join(output_dir, 'visualization.png'), results['visualization'])
        
        if results.get('room_visualization') is not None:
            cv2.imwrite(os.path.join(output_dir, 'room_optimization.png'), results['room_visualization'])
        
        # Save contours as JSON
        contour_data = {
            'building_outline': self._contour_to_list(results.get('building_outline')),
            'rooms': [self._contour_to_list(room) for room in results.get('rooms', [])],
            'optimized_rooms': [self._contour_to_list(room) for room in results.get('optimized_rooms', [])]
        }
        
        with open(os.path.join(output_dir, 'contours.json'), 'w') as f:
            json.dump(contour_data, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _contour_to_list(self, contour: Optional[np.ndarray]) -> Optional[List]:
        """Convert numpy contour to JSON-serializable list"""
        if contour is None:
            return None
        return contour.tolist()