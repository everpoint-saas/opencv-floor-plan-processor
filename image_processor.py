"""
Image processing module for architectural floor plan images.
Handles various image processing operations.
"""
import cv2
import numpy as np
import logging # 로깅 추가


class ImageProcessor:
    """
    Class for processing architectural floor plan images.
    """
    
    def __init__(self):
        """Initialize the image processor"""
        pass
    
    def process_image(self, image, params):
        """
        Apply various preprocessing steps to the image.
        (Corrected order for hatching removal)
        
        Args:
            image (numpy.ndarray): Input image (usually grayscale from Preprocessor)
            params (dict): Dictionary of preprocessing parameters
            
        Returns:
            numpy.ndarray: Processed binary image, or None if processing fails.
        """
        if image is None:
            logging.error("Input image to ImageProcessor.process_image is None.")
            return None
            
        # Make a copy to avoid modifying the original array passed to the function
        # Preprocessor에서 이미 grayscale로 변환해서 넘겨준다고 가정
        if len(image.shape) == 3: 
             # 혹시 컬러 이미지가 들어오면 그레이스케일 변환 (안전 장치)
             logging.warning("ImageProcessor received a color image, converting to grayscale.")
             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
             gray = image.copy()
        else:
             logging.error(f"Unsupported image shape in ImageProcessor: {image.shape}")
             return None

        logging.debug("Starting ImageProcessor processing steps...")

        # --- 그레이스케일 이미지에 대한 처리 (이진화 전) ---
        # Apply adaptive histogram equalization if selected
        if params.get('use_clahe', False):
            logging.debug("Applying CLAHE in ImageProcessor.")
            clahe = cv2.createCLAHE(clipLimit=params.get('clahe_limit', 2.0), 
                                    tileGridSize=(params.get('clahe_tile_size', 8), params.get('clahe_tile_size', 8)))
            gray = clahe.apply(gray)
            
        # Apply noise reduction if selected (Median or NLM)
        denoise_strength = params.get('denoise_strength', 0)
        if denoise_strength == 1:  # Light Median
            logging.debug("Applying light median blur (k=3) in ImageProcessor.")
            gray = cv2.medianBlur(gray, 3)
        elif denoise_strength == 2:  # Medium Median
            logging.debug("Applying medium median blur (k=5) in ImageProcessor.")
            gray = cv2.medianBlur(gray, 5)
        elif denoise_strength == 3:  # Strong NLM
            logging.debug("Applying strong NLM denoising in ImageProcessor.")
            h_nlm = params.get('nlm_h', 10)
            template_size = params.get('nlm_template_size', 7)
            search_size = params.get('nlm_search_size', 21)
            # NLM은 uint8 타입 입력 필요
            if gray.dtype != np.uint8: gray = np.clip(gray, 0, 255).astype(np.uint8)
            gray = cv2.fastNlMeansDenoising(gray, None, h=h_nlm, templateWindowSize=template_size, searchWindowSize=search_size)
        
        # Apply Gaussian blur if blur size is greater than 1
        blur_size = params.get('blur_size', 1)
        if blur_size > 1:
            blur_ksize = blur_size if blur_size % 2 == 1 else blur_size + 1
            logging.debug(f"Applying Gaussian Blur with ksize={blur_ksize} in ImageProcessor.")
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
        
        # --- 1. Binarization (Thresholding) ---
        logging.debug("Applying thresholding in ImageProcessor...")
        binary = self._apply_thresholding(gray, params)
        if binary is None: # Check if thresholding failed
            logging.error("Thresholding failed in ImageProcessor.")
            return None # Return None if binarization fails
        logging.debug("Thresholding complete.")

        # --- 2. Post-Binarization Processing ---
        processed_binary = binary.copy() # Start with the binarized image

        # !!! 해치 제거는 반드시 이진화된 이미지(processed_binary)에 대해 수행 !!!
        if params.get('remove_hatching', False):
            kernel_size = params.get('hatching_kernel_size', 5)
            logging.debug(f"Applying hatching removal with kernel size {kernel_size}.")
            try:
                 # remove_hatching_and_thin_lines 함수가 수정된 이진 이미지를 반환한다고 가정
                 processed_binary = self.remove_hatching_and_thin_lines(processed_binary, kernel_size) 
                 logging.debug("Hatching removal applied.")
            except Exception as hatch_err:
                 logging.error(f"Error during hatching removal: {hatch_err}", exc_info=True)
                 # 오류 발생 시 이전 이진 이미지를 계속 사용
        
        # Apply morphological operations if selected
        morph_ops = params.get('morph_operations', [])  # UI에서 설정한 연산 목록을 가져옴
        if morph_ops:
            logging.debug("Applying morphological operations in ImageProcessor...")
            processed_binary = self._apply_morphology(processed_binary, morph_ops)
            logging.debug("Morphological operations complete.")
        
        # Apply Canny edge detection if selected (Operates on the result so far)
        if params.get('use_canny', False):
            logging.debug("Applying Canny edge detection in ImageProcessor.")
            canny_low = params.get('canny_low', 50)
            canny_high = params.get('canny_high', 150)
            # Canny는 그레이스케일 또는 단일 채널 이미지에 적용하는 것이 일반적이지만,
            # 여기서는 이전 단계의 이진 이미지에 적용
            if len(processed_binary.shape) == 3: # 혹시 컬러로 잘못 전달된 경우 대비
                processed_binary_for_canny = cv2.cvtColor(processed_binary, cv2.COLOR_BGR2GRAY)
            else:
                processed_binary_for_canny = processed_binary
            
            # Canny 입력은 uint8이어야 함
            if processed_binary_for_canny.dtype != np.uint8:
                 processed_binary_for_canny = np.clip(processed_binary_for_canny, 0, 255).astype(np.uint8)

            edges = cv2.Canny(processed_binary_for_canny, canny_low, canny_high)
            processed_binary = edges # Canny 결과(엣지맵)를 최종 결과로 사용
            logging.debug("Canny edge detection complete.")
        
        # Apply additional filters if selected (Operate on the result so far)
        if params.get('use_sharpen', False) and not params.get('use_canny', False): # Canny 후에는 보통 안 함
            logging.debug("Applying sharpening filter.")
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            # filter2D 입력은 uint8, float32 등 가능
            processed_binary = cv2.filter2D(processed_binary, -1, kernel)
            
        if params.get('use_skeletonize', False) and not params.get('use_canny', False):
            logging.debug("Applying skeletonization.")
            processed_binary = self.skeletonize(processed_binary)
            
        if params.get('use_distance_transform', False) and not params.get('use_canny', False):
            logging.debug("Applying distance transform.")
            # Distance transform 입력은 이진 이미지 (0 또는 255)
            if np.max(processed_binary) > 1 and len(np.unique(processed_binary)) > 2: # 아직 이진화 안 됐다면? (이론상 여기서는 이진화 상태여야 함)
                 _, binary_for_dist = cv2.threshold(processed_binary, 127, 255, cv2.THRESH_BINARY)
            else:
                 binary_for_dist = processed_binary
            
            # 객체(흰색)까지의 거리를 계산하려면 보통 반전된 이미지 사용
            dist = cv2.distanceTransform(cv2.bitwise_not(binary_for_dist), cv2.DIST_L2, 5)
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX) # 0-1 범위로 정규화
            # 결과를 다시 이진화하거나 다른 방식으로 사용 가능. 여기서는 예시로 정규화된 거리맵을 반환.
            # 또는 특정 임계값으로 다시 이진화: _, processed_binary = cv2.threshold(dist, 0.5, 1.0, cv2.THRESH_BINARY)
            processed_binary = (dist * 255).astype(np.uint8) # 시각화 위해 0-255로 변환
        
        if params.get('use_thinning', False) and not params.get('use_canny', False):
            logging.debug("Applying thinning.")
            processed_binary = self.thinning(processed_binary)
            
        logging.info("ImageProcessor processing finished.")
        return processed_binary
    
    def _apply_thresholding(self, gray, params):
        """
        Apply thresholding to a grayscale image based on parameters.
        
        Args:
            gray (numpy.ndarray): Grayscale input image
            params (dict): Dictionary of preprocessing parameters
            
        Returns:
            numpy.ndarray: Binary image
        """
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
                block_size += 1  # Ensure odd number
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, block_size, 
                                          params.get('adaptive_c', 2))
        elif thresh_method == 5:  # Adaptive Gaussian
            block_size = params.get('adaptive_block_size', 11)
            if block_size % 2 == 0:
                block_size += 1  # Ensure odd number
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block_size, 
                                          params.get('adaptive_c', 2))
        elif thresh_method == 6:  # Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            # Default to basic thresholding if method not recognized
            _, binary = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_BINARY)
            
        return binary
    
    def _apply_morphology(self, binary, morph_ops): # morph_type -> morph_ops로 변경
        """
        Apply morphological operations based on a list of operations.
        
        Args:
            binary (numpy.ndarray): Binary input image
            morph_ops (list): List of morphological operation parameters
            
        Returns:
            numpy.ndarray: Processed binary image
        """
        result = binary.copy()
        for op in morph_ops:
            morph_type = op.get('morph_type', 0)
            if morph_type == 0: # None
                continue

            kernel_size = op.get('morph_size', 3)
            kernel_shape = op.get('kernel_shape', 0)
            
            # Create kernel based on shape
            if kernel_shape == 0:  # Rectangle
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            elif kernel_shape == 1:  # Ellipse
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            elif kernel_shape == 2:  # Cross
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
            else:
                # Default to rectangle
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            morph_iterations = op.get('morph_iterations', 1)
            
            if morph_type == 1:  # Dilate
                result = cv2.dilate(result, kernel, iterations=morph_iterations)
            elif morph_type == 2:  # Erode
                result = cv2.erode(result, kernel, iterations=morph_iterations)
            elif morph_type == 3:  # Open (Erode then Dilate)
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
            elif morph_type == 4:  # Close (Dilate then Erode)
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
            elif morph_type == 5:  # Gradient
                result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel, iterations=morph_iterations)
            elif morph_type == 6:  # Top Hat
                result = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel, iterations=morph_iterations)
            elif morph_type == 7:  # Black Hat
                result = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel, iterations=morph_iterations)
            # else: # None은 위에서 처리
                
        return result
    
    def remove_noise(self, image, min_component_size=100):
        """
        Remove small noise components from the binary image.
        
        Args:
            image (numpy.ndarray): Binary input image
            min_component_size (int): Minimum size of components to keep
            
        Returns:
            numpy.ndarray: Cleaned image
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
        
        # Create a new image with only the large enough components
        result = np.zeros_like(image)
        
        # Start from 1 to skip the background
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                result[labels == i] = 255
        
        return result
    
    def detect_lines(self, image, min_line_length=50, max_line_gap=5):
        """
        Detect lines in the image using Hough Line Transform.
        
        Args:
            image (numpy.ndarray): Input image
            min_line_length (int): Minimum line length
            max_line_gap (int): Maximum gap between line segments
            
        Returns:
            list: Detected lines
        """
        lines = cv2.HoughLinesP(
            image, 1, np.pi/180, threshold=50, 
            minLineLength=min_line_length, maxLineGap=max_line_gap
        )
        
        return lines if lines is not None else []
    
    def remove_hatching_and_thin_lines(self, image, kernel_size=5):
        """
        수평 및 수직 해치 패턴과 얇은 선을 제거합니다.
        
        Args:
            image (numpy.ndarray): 입력 이미지
            kernel_size (int): 모폴로지 연산에 사용할 커널 크기
            
        Returns:
            numpy.ndarray: 해치 및 얇은 선이 제거된 이미지
        """
        # 이미지가 흑백인지 확인
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 이진화 (필요한 경우)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 수평 커널 생성 (1 x kernel_size)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1))
        
        # 수직 커널 생성 (kernel_size x 1)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size))
        
        # 수평 방향 열림 연산 (수평 해치 제거)
        horizontal_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 수직 방향 열림 연산 (수직 해치 제거)
        vertical_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # 두 결과를 OR 연산으로 결합
        combined = cv2.bitwise_and(horizontal_opened, vertical_opened)
        
        return combined
    def skeletonize(self, image):
        """
        Perform skeletonization of binary image.
        
        Args:
            image (numpy.ndarray): Binary input image
            
        Returns:
            numpy.ndarray: Skeletonized image
        """
        # Ensure binary image (0 and 255)
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # Invert if needed (skeleton operates on foreground as 1s)
        if cv2.countNonZero(binary) > binary.size / 2:
            binary = cv2.bitwise_not(binary)
        
        # Normalize to 0 and 1
        skeleton = binary // 255
        
        # Create output skeleton
        size = np.size(skeleton)
        skeleton_result = np.zeros(skeleton.shape, np.uint8)
        
        # Get structuring elements for morphological operations
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        # Perform skeletonization
        done = False
        while not done:
            eroded = cv2.erode(skeleton, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(skeleton, temp)
            skeleton_result = cv2.bitwise_or(skeleton_result, temp)
            skeleton = eroded.copy()
            
            # Check if done
            zeros = size - cv2.countNonZero(skeleton)
            if zeros == size:
                done = True
        
        # Return binary image
        return skeleton_result * 255
    
    def thinning(self, image):
        """
        Perform thinning operation (Zhang-Suen algorithm).
        
        Args:
            image (numpy.ndarray): Binary input image
            
        Returns:
            numpy.ndarray: Thinned image
        """
        # Ensure binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        binary = binary // 255  # Convert to 0 and 1
        
        # Make copy for iterations
        thinned = binary.copy()
        
        # Define structuring elements for hit-and-miss transform
        kernel1 = np.array([[-1, -1, -1], [-1, 1, -1], [1, 1, 1]], dtype=np.int8)
        kernel2 = np.array([[1, -1, -1], [1, 1, -1], [1, -1, -1]], dtype=np.int8)
        
        # Iterate until no more change
        prev = np.zeros_like(thinned)
        
        while np.sum(np.abs(thinned - prev)) > 0:
            prev = thinned.copy()
            
            # Step 1
            k1 = cv2.morphologyEx(thinned, cv2.MORPH_HITMISS, kernel1)
            thinned = thinned - k1
            
            # Step 2
            k2 = cv2.morphologyEx(thinned, cv2.MORPH_HITMISS, kernel2)
            thinned = thinned - k2
        
        # Convert back to 8-bit image
        return thinned * 255
    
    def remove_text_simple(self, image, min_text_height=5, max_text_height=40):
        """
        Attempt to remove text-like elements from the floor plan using contour analysis.
        This is a simplified approach that may need adjustment based on the specific plans.
        
        Args:
            image (numpy.ndarray): Binary input image
            min_text_height (int): Minimum height for potential text regions
            max_text_height (int): Maximum height for potential text regions
            
        Returns:
            numpy.ndarray: Image with potential text removed
        """
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask for non-text elements
        mask = np.ones_like(image)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Filter based on text-like properties
            if (min_text_height <= h <= max_text_height and 
                aspect_ratio > 1.5 and 
                area < 1000):
                # Fill potential text regions with black
                cv2.drawContours(mask, [contour], 0, 0, -1)
        
        # Apply the mask
        result = cv2.bitwise_and(image, mask)
        
        return result
