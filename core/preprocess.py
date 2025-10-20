"""
Main preprocessing module for architectural floor plan images.
Integrates PDF loading, image processing, and OCR functionality.
Applies inpainting for text removal before binarization.
"""
import cv2
import numpy as np
import logging

from core.pdf_loader import PDFLoader
from core.image_processor import ImageProcessor
# ocr_handler_inpainting_v1의 OCRHandler를 임포트한다고 가정
from core.ocr_handler import OCRHandler 


class Preprocessor:
    """
    Class for preprocessing PDF architectural drawings.
    Acts as a facade for the PDF loading, image processing, and OCR components.
    """
    
    def __init__(self):
        """Initialize the preprocessor with its component handlers"""
        self.pdf_loader = PDFLoader()
        self.image_processor = ImageProcessor()
        self.ocr_handler = OCRHandler()
        self.text_regions = [] # Preprocessor가 text_regions를 관리
    
    def load_pdf(self, pdf_path):
        """Load a PDF file and return the number of pages."""
        return self.pdf_loader.load_pdf(pdf_path)
    
    def get_page_image(self, page_number):
        """Convert a PDF page to an image."""
        return self.pdf_loader.get_page_image(page_number)
    
    def set_resolution(self, dpi):
        """Set the resolution for PDF rendering."""
        self.pdf_loader.set_resolution(dpi)
    
    def process_image(self, image, params):
        """
        Apply various preprocessing steps to the image, including inpainting for text removal.
        
        Args:
            image (numpy.ndarray): Input image (usually color from PDF).
            params (dict): Dictionary of preprocessing parameters.
            
        Returns:
            numpy.ndarray: Processed binary image.
        """
        if image is None:
            logging.error("Input image to process_image is None.")
            return None

        logging.info("Starting image preprocessing...")
        processed_img = image.copy()

        # 1. Convert to Grayscale (if color)
        # Most subsequent operations work better on grayscale
        if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
            logging.debug("Converting image to grayscale.")
            gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        elif len(processed_img.shape) == 2:
            gray = processed_img # Already grayscale
        else:
            logging.error(f"Unsupported image shape for processing: {processed_img.shape}")
            return None
        
        # --- Optional early processing on grayscale (before inpainting) ---
        # Example: Apply CLAHE if needed for better contrast before text removal
        if params.get('use_clahe', False):
             logging.debug("Applying CLAHE.")
             clahe = cv2.createCLAHE(clipLimit=params.get('clahe_clip_limit', 2.0), 
                                     tileGridSize=(params.get('clahe_tile_size', 8), params.get('clahe_tile_size', 8)))
             gray = clahe.apply(gray)

        # Example: Apply light denoising if needed
        denoise_strength = params.get('denoise_strength', 0)
        if denoise_strength == 1: # Light Median
             logging.debug("Applying light median blur (k=3).")
             gray = cv2.medianBlur(gray, 3)
        elif denoise_strength == 2: # Medium Median
             logging.debug("Applying medium median blur (k=5).")
             gray = cv2.medianBlur(gray, 5)
        # Strong NLM might be too aggressive before inpainting

        # --- 2. Text Removal using Inpainting (Applied on grayscale image) ---
        if params.get('remove_text', False) and self.text_regions:
            logging.info("Applying inpainting for text removal on grayscale image...")
            padding = params.get('text_padding', 3) # Use a small padding for inpainting mask
            radius = params.get('inpaint_radius', 5) 
            method = params.get('inpaint_method', 'telea') 
            
            inpainted_gray = self.ocr_handler.mask_text_regions(
                gray, # Pass the grayscale image
                padding=padding, 
                inpaint_radius=radius, 
                inpaint_method=method
            )
            if inpainted_gray is None:
                 logging.error("Inpainting failed, proceeding without text removal.")
                 # gray 변수는 그대로 유지됨
            else:
                 logging.info("Inpainting applied successfully.")
                 gray = inpainted_gray # Update gray image with inpainted result
        elif params.get('remove_text', False) and not self.text_regions:
             logging.warning("'Remove Text' is checked, but no text regions were detected (or OCR hasn't run). Skipping inpainting.")
        else:
             logging.debug("Text removal (inpainting) is disabled or no text regions available.")


        # --- 3. Further Processing on potentially inpainted grayscale image ---
        # Apply remaining processing steps from ImageProcessor, but operate on 'gray'
        # We need to adapt ImageProcessor's logic or replicate parts here.
        # For simplicity, let's replicate some key steps.
        
        processed_intermediate = gray.copy() # Start with the (potentially inpainted) gray image

        # Noise Reduction (if not applied earlier or needs more)
        # Example: Apply stronger denoising after inpainting if needed
        if denoise_strength == 3: # Strong NLM
             logging.debug("Applying strong NLM denoising.")
             # NLM parameters from params dict
             h_nlm = params.get('nlm_h', 10)
             template_size = params.get('nlm_template_size', 7)
             search_size = params.get('nlm_search_size', 21)
             processed_intermediate = cv2.fastNlMeansDenoising(processed_intermediate, None, h=h_nlm, templateWindowSize=template_size, searchWindowSize=search_size)

        # Blur (Gaussian Blur is common before thresholding)
        blur_size = params.get('blur_size', 1)
        if blur_size > 1:
            blur_ksize = blur_size if blur_size % 2 == 1 else blur_size + 1
            logging.debug(f"Applying Gaussian Blur with ksize={blur_ksize}.")
            processed_intermediate = cv2.GaussianBlur(processed_intermediate, (blur_ksize, blur_ksize), 0)

        # --- 4. Binarization (Thresholding) ---
        logging.debug("Applying thresholding...")
        # Use the _apply_thresholding method from ImageProcessor instance
        # Note: This assumes ImageProcessor has an _apply_thresholding method
        binary_image = self.image_processor._apply_thresholding(processed_intermediate, params)
        if binary_image is None:
             logging.error("Thresholding failed.")
             return None
        logging.debug("Thresholding complete.")

        # --- 5. Post-Binarization Processing (Morphology, etc.) ---
        processed_binary = binary_image.copy()

        # Morphological Operations
        morph_ops_list = params.get('morph_operations', [])
        if morph_ops_list: # 리스트가 비어있지 않으면 실행
             logging.debug("Applying morphological operations...")
             # ImageProcessor의 _apply_morphology는 연산 리스트를 받습니다.
             processed_binary = self.image_processor._apply_morphology(processed_binary, morph_ops_list)
             logging.debug("Morphological operations complete.")
        
        # Remove small connected components (Noise cleaning on binary image)
        # 'remove_small' 대신 'remove_small_components' 키를 사용합니다.
        if params.get('remove_small_components', False): 
             min_comp_size = params.get('min_component_size', 100)
             logging.debug(f"Removing small connected components (min_size={min_comp_size}).")
             # self.image_processor.remove_noise는 ImageProcessor에 이미 구현되어 있습니다.
             processed_binary = self.image_processor.remove_noise(processed_binary, min_comp_size)
             logging.debug("Small component removal complete.")

        # Other binary operations from ImageProcessor if needed (Canny, Thinning, etc.)
        # Example: Canny Edge Detection (if selected)
        if params.get('use_canny', False):
            logging.debug("Applying Canny edge detection.")
            canny_low = params.get('canny_low', 50)
            canny_high = params.get('canny_high', 150)
            processed_binary = cv2.Canny(processed_binary, canny_low, canny_high)
            logging.debug("Canny edge detection complete.")

        logging.info("Image preprocessing finished.")
        return processed_binary # Return the final processed binary image
    
    # --- OCR 관련 메서드들은 OCRHandler 호출로 위임 ---
    def initialize_ocr(self):
        """Initialize OCR engine via OCRHandler."""
        self.ocr_handler.initialize_ocr()
    
    def set_ocr_languages(self, languages):
        """Set OCR languages via OCRHandler."""
        self.ocr_handler.set_languages(languages)
    
    def detect_text(self, image, enhance=True, use_mser=True):
        """Detect text regions via OCRHandler and store them."""
        # 결과는 self.text_regions에 저장됨
        self.text_regions = self.ocr_handler.detect_text(image, enhance, use_mser) 
        return self.text_regions
    
    def draw_text_regions(self, image, color=(0, 255, 0), thickness=2):
        """Draw detected text regions via OCRHandler."""
        # OCRHandler가 자체적으로 text_regions를 가지고 있으므로, 인자로 전달할 필요 없음
        return self.ocr_handler.draw_text_regions(image, color, thickness)
    
    # mask_text_regions는 이제 process_image 내부에서 호출됨
    # def mask_text_regions(self, image, padding=5):
    #     """ Mask text regions via OCRHandler (now uses inpainting). """
    #     # 이 메서드를 직접 호출하기보다 process_image 내에서 처리하는 것이 좋음
    #     # 필요하다면 파라미터를 추가하여 호출 가능
    #     # return self.ocr_handler.mask_text_regions(image, padding=padding) 
    #     pass 

    # --- 기타 유틸리티 메서드 (ImageProcessor 호출 위임) ---
    def remove_noise(self, image, min_component_size=100):
        """Remove small noise components via ImageProcessor."""
        return self.image_processor.remove_noise(image, min_component_size)
    
    def detect_lines(self, image, min_line_length=50, max_line_gap=5):
        """Detect lines via ImageProcessor."""
        return self.image_processor.detect_lines(image, min_line_length, max_line_gap)

