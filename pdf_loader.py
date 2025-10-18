"""
PDF loading and conversion module for architectural floor plan images.
Handles PDF loading and conversion to images.
"""
import os
import fitz  # PyMuPDF
import numpy as np
import cv2


class PDFLoader:
    """
    Class for loading and converting PDF architectural drawings to images.
    """
    
    def __init__(self):
        """Initialize the PDF loader"""
        self.pdf_doc = None
        self.pdf_path = None
        self.image_resolution = 300  # DPI for PDF rendering
    
    def load_pdf(self, pdf_path):
        """
        Load a PDF file and return the number of pages.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            tuple: (success, page_count)
        """
        print("1. PDF 로드 시작")
        if not os.path.exists(pdf_path):
            print(f"파일이 존재하지 않습니다: {pdf_path}")
            return False, 0
        
        print("2. PDF 파일 존재 확인 완료")    
        try:
            print("3. fitz.open 호출 시작")
            self.pdf_doc = fitz.open(pdf_path)
            print("4. fitz.open 호출 완료")
            self.pdf_path = pdf_path
            page_count = len(self.pdf_doc)
            print(f"5. PDF 페이지 수: {page_count}")
            return True, page_count
        except Exception as e:
            print(f"PDF 로딩 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, 0
    
    def get_page_image(self, page_number):
        """
        Convert a PDF page to an image.
        
        Args:
            page_number (int): Page number (0-based)
            
        Returns:
            numpy.ndarray: Image as a numpy array
        """
        print("get_page_image 시작")
        if self.pdf_doc is None:
            print("PDF 문서가 None입니다")
            return None
        
        if page_number < 0 or page_number >= len(self.pdf_doc):
            print(f"페이지 번호가 범위를 벗어남: {page_number}, 총 페이지: {len(self.pdf_doc)}")
            return None
        
        try:
            print("페이지 가져오기 시작")
            # Get the page
            page = self.pdf_doc[page_number]
            print("페이지 가져오기 완료")
            
            # Set the resolution for rendering
            zoom = self.image_resolution / 72  # Default PDF resolution is 72 DPI
            matrix = fitz.Matrix(zoom, zoom)
            print(f"변환 매트릭스 생성 완료: 해상도 {self.image_resolution}dpi")
            
            print("픽스맵 렌더링 시작 - 이 부분에서 메모리 사용량이 높아질 수 있음")
            # Render page to an image
            pix = page.get_pixmap(matrix=matrix)
            print(f"픽스맵 렌더링 완료: 크기 {pix.width}x{pix.height}, 채널 {pix.n}")
            
            print("NumPy 배열로 변환 시작")
            # Convert to a numpy array
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            print(f"NumPy 배열 변환 완료: 크기 {img_array.shape}")
            
            # Convert to RGB if it's in RGBA format (typical for PDFs with transparency)
            if pix.n == 4:
                print("RGBA에서 RGB로 변환 시작")
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                print("RGB 변환 완료")
            
            # Memory cleanup (explicitly release objects)
            print("메모리 정리 시작")
            pix = None
            page = None
            print("메모리 정리 완료")
            
            return img_array
        except Exception as e:
            import traceback
            print(f"get_page_image 함수에서 예외 발생: {str(e)}")
            traceback.print_exc()
            return None
    
    def set_resolution(self, dpi):
        """
        Set the resolution for PDF rendering.
        
        Args:
            dpi (int): Resolution in dots per inch
        """
        self.image_resolution = dpi
