"""
OCR handling module for architectural floor plan images.
Handles text detection and text masking using inpainting.
"""
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
# from PySide6.QtWidgets import QApplication # Not used directly here
# from PySide6.QtCore import QThread # Not used directly here
import easyocr
import logging # 로깅 추가

class OCRHandler:
    """
    Class for handling OCR operations on architectural floor plan images.
    """
    
    def __init__(self):
        """Initialize the OCR handler"""
        self.ocr_reader = None
        self.ocr_langs = ['ko', 'en'] 
        self.text_regions = [] # List to store tuples: (bounding_box_points, text, confidence)
        
    def initialize_ocr(self):
        """Initialize EasyOCR reader"""
        if self.ocr_reader:
            return # 이미 초기화됨
        try:
            logging.info("Initializing EasyOCR Reader...")
            # Use gpu=False explicitly if GPU is not intended or causing issues
            self.ocr_reader = easyocr.Reader(self.ocr_langs, gpu=False) 
            logging.info("EasyOCR Reader initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing OCR: {e}", exc_info=True)
            self.ocr_reader = None
    
    def set_languages(self, languages):
        """Set the OCR languages."""
        new_langs = [lang for lang in languages if lang] 
        if set(new_langs) != set(self.ocr_langs) or self.ocr_reader is None:
             self.ocr_langs = new_langs
             self.ocr_reader = None # Reset reader to force reinitialization with new languages
             logging.info(f"OCR languages set to: {self.ocr_langs}. Reader will reinitialize on next use.")
        else:
             logging.info(f"OCR languages already set to: {self.ocr_langs}.")

    
    def detect_text(self, image, enhance=True, use_mser=True):
        """
        Detect text in an image using OCR.
        Stores results in self.text_regions.
        """
        if self.ocr_reader is None:
            self.initialize_ocr()
            
        if self.ocr_reader is None:
            logging.error("OCR Reader not available. Cannot detect text.")
            self.text_regions = []
            return []
        
        try:
            logging.info("Starting text detection...")
            target_image = image.copy() # 원본 이미지 복사본 사용

            # Ensure image is suitable for OCR (e.g., grayscale or color)
            # EasyOCR can handle both, but preprocessing might be beneficial
            if enhance:
                if len(target_image.shape) == 3:
                    gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = target_image
                
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_gray = clahe.apply(gray)
                
                # Use the enhanced grayscale image for OCR
                # Or, if EasyOCR performs better on color, enhance channels or use original color
                target_image_for_ocr = enhanced_gray # 예시: 향상된 그레이스케일 사용
                logging.debug("Using enhanced grayscale image for OCR.")
            else:
                target_image_for_ocr = target_image
                logging.debug("Using original image for OCR.")

            # MSER 부분은 제외하거나 선택적으로 사용 가능 (현재는 직접 OCR 실행)
            # if use_mser: ... (MSER 로직) ...
            
            # Run OCR
            # detail=1 gives coordinates, text, and confidence
            # paragraph=False treats each detected box independently
            ocr_results = self.ocr_reader.readtext(target_image_for_ocr, detail=1, paragraph=False) 
            
            self.text_regions = ocr_results # 결과 저장
            logging.info(f"Text detection complete. Found {len(self.text_regions)} regions.")
            return self.text_regions
        
        except Exception as e:
            logging.error(f"Error during text detection: {e}", exc_info=True)
            self.text_regions = []
            return []

    def mask_text_regions(self, image, padding=3, inpaint_radius=5, inpaint_method='telea'):
        """
        Mask detected text regions using inpainting to reconstruct underlying lines.

        Args:
            image (numpy.ndarray): Input image (grayscale or color). 
                                   Inpainting works best on the image *before* final binarization.
            padding (int): Padding around the text bounding box to create the mask.
                           Should be small to minimize impact on nearby lines.
            inpaint_radius (int): Neighborhood radius for inpainting algorithm.
            inpaint_method (str): 'telea' or 'ns' (Navier-Stokes).

        Returns:
            numpy.ndarray: Image with text regions inpainted.
        """
        if not self.text_regions:
            logging.info("No text regions detected to mask.")
            return image

        if image is None:
            logging.warning("Input image for masking is None.")
            return None

        result_image = image.copy()
        # 마스크 생성 (텍스트 영역을 흰색(255)으로 표시)
        # 마스크는 입력 이미지와 동일한 크기의 단일 채널(그레이스케일)이어야 함
        mask = np.zeros(image.shape[:2], dtype=np.uint8) 

        logging.info(f"Creating mask for {len(self.text_regions)} text regions with padding={padding}.")
        for detection in self.text_regions:
            # detection[0] contains the coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            points = np.array(detection[0], dtype=np.int32)
            
            # 바운딩 박스를 약간 확장하여 마스크 생성 (패딩 적용)
            try:
                rect = cv2.boundingRect(points)
                x, y, w, h = rect
                
                # 적용할 패딩 계산 (이미지 경계 고려)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(mask.shape[1], x + w + padding) # mask width
                y2 = min(mask.shape[0], y + h + padding) # mask height

                # 확장된 영역을 마스크에 흰색으로 채움
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                # 또는, 더 정확하게 하려면 컨벡스 헐(convex hull)을 사용
                # hull = cv2.convexHull(points)
                # cv2.drawContours(mask, [hull], -1, 255, -1) 
                # # 컨벡스 헐에 패딩 적용은 더 복잡 (예: 모폴로지 팽창 사용)
                # kernel = np.ones((padding*2+1, padding*2+1), np.uint8)
                # mask = cv2.dilate(mask, kernel, iterations=1) # 마스크 영역 팽창

            except Exception as e:
                 logging.error(f"Error processing points for masking: {points}. Error: {e}", exc_info=True)
                 continue # 오류 발생 시 해당 영역은 건너뜀

        # 인페인팅 방법 선택
        if inpaint_method.lower() == 'ns':
            method_flag = cv2.INPAINT_NS
            logging.info(f"Performing inpainting using Navier-Stokes method with radius={inpaint_radius}.")
        else:
            method_flag = cv2.INPAINT_TELEA
            logging.info(f"Performing inpainting using Telea method with radius={inpaint_radius}.")

        # 인페인팅 수행
        try:
            # cv2.inpaint는 컬러 또는 그레이스케일 이미지 모두에 작동
            inpainted_image = cv2.inpaint(result_image, mask, inpaintRadius=inpaint_radius, flags=method_flag)
            logging.info("Inpainting complete.")
            return inpainted_image
        except Exception as e:
            logging.error(f"Error during inpainting: {e}", exc_info=True)
            return result_image # 오류 발생 시 원본 이미지 반환

    # --- 기존 함수들 (filter_text_regions, draw_text_regions) ---
    # draw_text_regions는 디버깅/시각화 용도로 유용하게 사용 가능
    
    def filter_text_regions(self, confidence_threshold=0.5):
        """Filter text regions based on confidence threshold."""
        return [region for region in self.text_regions if region[2] >= confidence_threshold]
    
    def draw_text_regions(self, image, color=(0, 255, 0), thickness=2):
        """Draw rectangles around detected text regions."""
        if not self.text_regions:
            return image
        
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        for detection in self.text_regions:
            points = detection[0]
            text = detection[1]
            points_int = np.array(points, dtype=np.int32)
            
            cv2.polylines(result, [points_int], isClosed=True, color=color, thickness=thickness)
            
            # Put text label (optional)
            # label_pos = (points_int[0][0], points_int[0][1] - 10)
            # cv2.putText(result, text[:15], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result

