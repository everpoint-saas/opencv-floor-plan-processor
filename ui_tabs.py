# ui_tabs.py

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QSlider, QGroupBox, QComboBox,
                           QSpinBox, QCheckBox, QDoubleSpinBox, QPushButton, QScrollArea)
from PySide6.QtCore import Qt
import logging
# 맵 정의 (문자열 <-> 정수 변환)
MORPH_TYPE_MAP = {
    "없음": 0, "팽창": 1, "침식": 2, "열림": 3, "닫힘": 4,
    "그래디언트": 5, "탑햇": 6, "블랙햇": 7
}
MORPH_TYPE_MAP_REVERSE = {v: k for k, v in MORPH_TYPE_MAP.items()}

KERNEL_SHAPE_MAP = {
    "사각형": 0, "타원": 1, "십자형": 2
}
KERNEL_SHAPE_MAP_REVERSE = {v: k for k, v in KERNEL_SHAPE_MAP.items()}
class TabManager:
    """
    다양한 파라미터 탭과 컨트롤을 관리하는 클래스
    """
    def __init__(self, main_window):
        self.main_window = main_window

    def create_all_tabs(self, tab_widget):
        """모든 파라미터 탭 생성"""
        self.create_basic_preprocess_tab(tab_widget)
        self.create_advanced_preprocess_tab(tab_widget)
        self.create_morphology_tab(tab_widget)
        self.create_filter_tab(tab_widget)
        self.create_ocr_tab(tab_widget)
        self.create_contour_tab(tab_widget) # 이 함수 내부가 수정됩니다.

    def create_basic_preprocess_tab(self, tab_widget):
        """기본 전처리 파라미터 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- PDF 렌더링 DPI 설정 그룹 추가 ---
        dpi_group = QGroupBox("PDF 렌더링 설정")
        dpi_layout = QHBoxLayout() # 가로로 배치
        dpi_layout.addWidget(QLabel("렌더링 DPI:"))
        self.main_window.render_dpi = QComboBox()
        # 일반적인 DPI 값들을 추가, 필요시 더 추가하거나 수정 가능
        self.main_window.render_dpi.addItems(["150", "200", "300", "400", "600"])
        self.main_window.render_dpi.setCurrentText("300") # 기본값 300 DPI
        # DPI 변경 시 즉시 preprocessor에 반영하도록 연결 (선택적, 또는 PDF 로드 시점에만 반영)
        # self.main_window.render_dpi.currentTextChanged.connect(self.main_window.handlers.on_dpi_changed) # 핸들러 필요
        dpi_layout.addWidget(self.main_window.render_dpi)
        dpi_group.setLayout(dpi_layout)
        layout.addWidget(dpi_group)

        # 그레이스케일 변환 그룹
        conversion_group = QGroupBox("이미지 변환")
        conversion_layout = QVBoxLayout()
        self.main_window.use_grayscale = self.create_checkbox_control("그레이스케일 변환", conversion_layout)
        self.main_window.use_grayscale.setChecked(True)
        conversion_group.setLayout(conversion_layout)
        layout.addWidget(conversion_group)

        # 이진화 그룹
        thresh_group = QGroupBox("이진화 설정")
        thresh_layout = QVBoxLayout()

        thresh_method_layout = QHBoxLayout()
        thresh_method_layout.addWidget(QLabel("이진화 방법:"))
        self.main_window.thresh_method = QComboBox()
        self.main_window.thresh_method.addItems([
            "기본 이진화", "반전 이진화", "절단 이진화", "0으로 이진화",
            "적응형 평균", "적응형 가우시안", "Otsu 이진화"
        ])
        self.main_window.thresh_method.setCurrentIndex(6) # Otsu 기본값
        thresh_method_layout.addWidget(self.main_window.thresh_method)
        thresh_layout.addLayout(thresh_method_layout)

        self.main_window.threshold_slider = self.create_slider_control("임계값 (기본 이진화용):", 0, 255, 127, thresh_layout)
        
        self.main_window.threshold_value_for_contour = self.create_slider_control("컨투어 추출시 임계값:", 0, 255, 127, thresh_layout)


        adaptive_group = QGroupBox("적응형 이진화 설정")
        adaptive_layout = QVBoxLayout()
        self.main_window.adaptive_block_size = self.create_slider_control(
            "블록 크기:", 3, 51, 11, adaptive_layout, step=2
        )
        c_layout = QHBoxLayout()
        c_layout.addWidget(QLabel("C 값:"))
        self.main_window.adaptive_c = QSpinBox()
        self.main_window.adaptive_c.setRange(-20, 20)
        self.main_window.adaptive_c.setValue(2)
        self.main_window.adaptive_c.setSingleStep(1)
        c_layout.addWidget(self.main_window.adaptive_c)
        adaptive_layout.addLayout(c_layout)
        adaptive_group.setLayout(adaptive_layout)
        thresh_layout.addWidget(adaptive_group)

        thresh_group.setLayout(thresh_layout)
        layout.addWidget(thresh_group)

        blur_group = QGroupBox("블러 설정")
        blur_layout = QVBoxLayout()
        blur_type_layout = QHBoxLayout()
        blur_type_layout.addWidget(QLabel("블러 유형:"))
        self.main_window.blur_type = QComboBox()
        self.main_window.blur_type.addItems([
            "가우시안 블러", "박스 블러", "중간값 블러", "양방향 필터"
        ])
        blur_type_layout.addWidget(self.main_window.blur_type)
        blur_layout.addLayout(blur_type_layout)
        self.main_window.blur_slider = self.create_slider_control(
            "블러 크기:", 1, 21, 5, blur_layout, step=2
        )
        blur_group.setLayout(blur_layout)
        layout.addWidget(blur_group)

        tab_widget.addTab(tab, "기본 전처리")

    def create_advanced_preprocess_tab(self, tab_widget):
        """고급 전처리 파라미터 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        noise_group = QGroupBox("노이즈 제거")
        noise_layout = QVBoxLayout()
        self.main_window.denoise_strength = self.create_combobox_control(
            "노이즈 제거:", ["없음", "약함 (중간값)", "중간 (중간값)", "강함 (NLM)"], noise_layout
        )
        nlm_group = QGroupBox("NLM 파라미터")
        nlm_layout = QVBoxLayout()
        self.main_window.nlm_h = self.create_slider_control("필터 강도(h):", 1, 30, 10, nlm_layout)
        self.main_window.nlm_template_size = self.create_slider_control("템플릿 크기:", 3, 15, 7, nlm_layout, step=2)
        self.main_window.nlm_search_size = self.create_slider_control("검색 크기:", 9, 35, 21, nlm_layout, step=2)
        nlm_group.setLayout(nlm_layout)
        noise_layout.addWidget(nlm_group)
        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)

        hist_group = QGroupBox("히스토그램 처리")
        hist_layout = QVBoxLayout()
        self.main_window.use_clahe = self.create_checkbox_control("CLAHE 적용", hist_layout)
        clahe_params_layout = QHBoxLayout()
        clahe_params_layout.addWidget(QLabel("클립 한계:"))
        self.main_window.clahe_clip_limit = QDoubleSpinBox()
        self.main_window.clahe_clip_limit.setRange(0.1, 10.0)
        self.main_window.clahe_clip_limit.setValue(2.0)
        self.main_window.clahe_clip_limit.setSingleStep(0.1)
        clahe_params_layout.addWidget(self.main_window.clahe_clip_limit)
        clahe_params_layout.addWidget(QLabel("타일 크기:"))
        self.main_window.clahe_tile_size = QSpinBox()
        self.main_window.clahe_tile_size.setRange(2, 16)
        self.main_window.clahe_tile_size.setValue(8)
        clahe_params_layout.addWidget(self.main_window.clahe_tile_size)
        hist_layout.addLayout(clahe_params_layout)
        self.main_window.use_equalize_hist = self.create_checkbox_control("히스토그램 평활화", hist_layout)
        hist_group.setLayout(hist_layout)
        layout.addWidget(hist_group)

        hatching_group = QGroupBox("해치 및 얇은 선 제거")
        hatching_layout = QVBoxLayout()
        self.main_window.remove_hatching = self.create_checkbox_control("해치 및 얇은 선 제거", hatching_layout)
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("커널 크기:"))
        self.main_window.hatching_kernel_size = QSpinBox()
        self.main_window.hatching_kernel_size.setRange(3, 21)
        self.main_window.hatching_kernel_size.setValue(5)
        self.main_window.hatching_kernel_size.setSingleStep(2)
        kernel_layout.addWidget(self.main_window.hatching_kernel_size)
        hatching_layout.addLayout(kernel_layout)
        hatching_group.setLayout(hatching_layout)
        layout.addWidget(hatching_group)

        canny_group = QGroupBox("Canny 에지 검출")
        canny_layout = QVBoxLayout()
        self.main_window.use_canny = self.create_checkbox_control("Canny 에지 검출 적용", canny_layout)
        self.main_window.canny_low = self.create_slider_control("하한 임계값:", 0, 255, 50, canny_layout)
        self.main_window.canny_high = self.create_slider_control("상한 임계값:", 0, 255, 150, canny_layout)
        canny_group.setLayout(canny_layout)
        layout.addWidget(canny_group)

        tab_widget.addTab(tab, "고급 전처리")

    def create_morphology_tab(self, tab_widget):
        """형태학적 연산 파라미터 탭 생성 (다중 연산 지원, 프리셋 추가, 연결 요소 제거 추가)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 프리셋 선택 콤보 박스
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("모폴로지 연산 프리셋:"))
        self.morph_preset_combo = QComboBox()
        self.morph_preset_combo.addItems(["사용자 정의", "노이즈 제거", "선 연결", "윤곽선 정제"]) # 프리셋 목록
        self.morph_preset_combo.currentIndexChanged.connect(self.apply_morphology_preset)
        preset_layout.addWidget(self.morph_preset_combo)
        layout.addLayout(preset_layout)

        self.morph_ops_scroll_area = QScrollArea()
        self.morph_ops_widget = QWidget()
        self.morph_ops_layout = QVBoxLayout(self.morph_ops_widget)

        self.morph_ops_scroll_area.setWidgetResizable(True)
        self.morph_ops_scroll_area.setWidget(self.morph_ops_widget)

        add_button = QPushButton("연산 추가")
        add_button.clicked.connect(self.add_morph_operation_ui)
        layout.addWidget(add_button)

        layout.addWidget(self.morph_ops_scroll_area)

        # --- 작은 연결 요소 제거 기능 추가 ---
        components_group = QGroupBox("연결 요소 필터링")
        components_layout = QVBoxLayout()

        # QCheckBox를 self.main_window에 직접 할당하도록 수정
        self.main_window.remove_small_components_checkbox = QCheckBox("작은 연결 요소 제거")
        components_layout.addWidget(self.main_window.remove_small_components_checkbox)

        min_size_layout = QHBoxLayout()
        min_size_layout.addWidget(QLabel("최소 크기 (픽셀):"))
        # QSpinBox를 self.main_window에 직접 할당하도록 수정
        self.main_window.min_component_size_spinbox = QSpinBox()
        self.main_window.min_component_size_spinbox.setRange(1, 10000)
        self.main_window.min_component_size_spinbox.setValue(100) # 기본값
        min_size_layout.addWidget(self.main_window.min_component_size_spinbox)
        components_layout.addLayout(min_size_layout)

        components_group.setLayout(components_layout)
        layout.addWidget(components_group)
        # --- 추가 완료 ---

        tab_widget.addTab(tab, "형태학적 연산")

        self.morph_op_uis = [] # 생성된 UI 요소들을 저장하는 리스트

        # 기본적으로 몇 개의 연산 UI를 생성할지 결정 (예: 1개 또는 기존 로직 유지)
        # 여기서는 기존 로직대로 5개를 생성하고 '없음'으로 초기화하는 부분을 유지하되,
        # 필요에 따라 초기 생성 개수를 조절할 수 있습니다.
        for _ in range(5): # 또는 필요한 만큼의 기본 연산 UI 생성
            self.add_morph_operation_ui()
            # 새로 추가된 UI의 연산 타입을 "없음"으로 설정
            if self.morph_op_uis:
                morph_op_ui = self.morph_op_uis[-1]
                # QGroupBox 내부의 레이아웃에서 위젯을 찾아야 함
                # add_morph_operation_ui 내부 구조에 따라 접근 경로가 달라질 수 있음
                # 예시: 첫 번째 QHBoxLayout의 두 번째 위젯이 QComboBox라고 가정
                try:
                    op_layout = morph_op_ui.layout() # QVBoxLayout
                    type_combo_layout = op_layout.itemAt(0).layout() # QHBoxLayout for type
                    morph_type_combo = type_combo_layout.itemAt(1).widget() # QComboBox
                    if morph_type_combo:
                        morph_type_combo.setCurrentText("없음")
                except AttributeError as e:
                    print(f"Error accessing morph_type_combo for initialization: {e}")
            else:
                print("Error: Could not set initial morphology operation to '없음' as morph_op_uis is empty.")

    def add_morph_operation_ui(self):
        """새로운 모폴로지 연산 UI를 추가"""

        morph_op_group = QGroupBox("형태학적 연산")
        morph_op_layout = QVBoxLayout(morph_op_group)

        # 연산 타입 콤보 박스
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("연산 타입:"))
        morph_type_combo = QComboBox()
        morph_type_combo.addItems(["없음", "팽창", "침식", "열림", "닫힘", "그래디언트", "탑햇", "블랙햇"])
        type_layout.addWidget(morph_type_combo)
        morph_op_layout.addLayout(type_layout)

        # 커널 형태 콤보 박스
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("커널 형태:"))
        kernel_shape_combo = QComboBox()
        kernel_shape_combo.addItems(["사각형", "타원", "십자형"])
        kernel_layout.addWidget(kernel_shape_combo)
        morph_op_layout.addLayout(kernel_layout)

        # 커널 크기 스핀 박스
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("커널 크기:"))
        kernel_size_spinbox = QSpinBox()
        kernel_size_spinbox.setRange(1, 21)
        kernel_size_spinbox.setValue(3)
        size_layout.addWidget(kernel_size_spinbox)
        morph_op_layout.addLayout(size_layout)

        # 반복 횟수 스핀 박스
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("반복 횟수:"))
        iterations_spinbox = QSpinBox()
        iterations_spinbox.setRange(1, 10)
        iterations_spinbox.setValue(1)
        iter_layout.addWidget(iterations_spinbox)
        morph_op_layout.addLayout(iter_layout)

        remove_button = QPushButton("제거")
        remove_button.clicked.connect(lambda: self.remove_morph_operation_ui(morph_op_group))
        morph_op_layout.addWidget(remove_button)

        self.morph_ops_layout.addWidget(morph_op_group)
        self.morph_op_uis.append(morph_op_group)

        # UI가 제대로 추가되었는지 확인
        if not self.morph_op_uis:
            print("오류: 모폴로지 연산 UI가 추가되지 않았습니다.")
            return  # 함수 종료

        try:
            # 연산 타입에 따라 커널 설정 UI를 동적으로 표시/숨김 (예시)
            def update_kernel_visibility():
                is_none = morph_type_combo.currentText() == "없음"
                kernel_shape_combo.setVisible(not is_none)
                kernel_size_spinbox.setVisible(not is_none)
                iterations_spinbox.setVisible(not is_none)

            morph_type_combo.currentIndexChanged.connect(update_kernel_visibility)
            update_kernel_visibility() # 초기 상태 설정
        except Exception as e:
            print(f"모폴로지 연산 UI 초기화 중 오류: {e}")

    def remove_morph_operation_ui(self, morph_op_group):
        """모폴로지 연산 UI를 제거"""
        self.morph_ops_layout.removeWidget(morph_op_group)
        morph_op_group.deleteLater()
        self.morph_op_uis.remove(morph_op_group)

    def create_filter_tab(self, tab_widget):
        """필터 및 특수 효과 파라미터 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        effects_group = QGroupBox("이미지 효과")
        effects_layout = QVBoxLayout()
        self.main_window.use_sharpen = self.create_checkbox_control("선명화", effects_layout)
        dist_transform_layout = QVBoxLayout()
        self.main_window.use_distance_transform = self.create_checkbox_control("거리 변환", dist_transform_layout)
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("거리 유형:"))
        self.main_window.distance_type = QComboBox()
        self.main_window.distance_type.addItems(["L1", "L2", "C"])
        dist_layout.addWidget(self.main_window.distance_type)
        dist_transform_layout.addLayout(dist_layout)
        effects_layout.addLayout(dist_transform_layout)

        skeleton_group = QGroupBox("골격화 및 세선화")
        skeleton_layout = QVBoxLayout()
        self.main_window.use_skeletonize = self.create_checkbox_control("골격화", skeleton_layout)
        self.main_window.use_thinning = self.create_checkbox_control("세선화 (Zhang-Suen)", skeleton_layout)
        skeleton_group.setLayout(skeleton_layout)
        effects_layout.addWidget(skeleton_group)
        effects_group.setLayout(effects_layout)
        layout.addWidget(effects_group)

        contour_group = QGroupBox("윤곽 검출 (보조)")
        contour_layout = QVBoxLayout()
        self.main_window.use_laplacian = self.create_checkbox_control("라플라시안 에지 검출", contour_layout)
        self.main_window.use_sobel = self.create_checkbox_control("소벨 에지 검출", contour_layout)
        sobel_dir_layout = QHBoxLayout()
        sobel_dir_layout.addWidget(QLabel("소벨 방향:"))
        self.main_window.sobel_direction = QComboBox()
        self.main_window.sobel_direction.addItems(["X", "Y", "X+Y"])
        sobel_dir_layout.addWidget(self.main_window.sobel_direction)
        contour_layout.addLayout(sobel_dir_layout)
        contour_group.setLayout(contour_layout)
        layout.addWidget(contour_group)

        tab_widget.addTab(tab, "필터")

    def create_ocr_tab(self, tab_widget):
        """OCR 파라미터 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        lang_group = QGroupBox("OCR 설정")
        lang_layout = QVBoxLayout()
        self.main_window.lang_korean = self.create_checkbox_control("한국어", lang_layout)
        self.main_window.lang_korean.setChecked(True)
        self.main_window.lang_english = self.create_checkbox_control("영어", lang_layout)
        self.main_window.lang_english.setChecked(True)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("신뢰도 임계값:"))
        self.main_window.ocr_confidence = QDoubleSpinBox()
        self.main_window.ocr_confidence.setRange(0.0, 1.0)
        self.main_window.ocr_confidence.setValue(0.5)
        self.main_window.ocr_confidence.setSingleStep(0.1)
        self.main_window.ocr_confidence.setDecimals(2)
        conf_layout.addWidget(self.main_window.ocr_confidence)
        lang_layout.addLayout(conf_layout)

        text_removal_group = QGroupBox("텍스트 제거 (인페인팅)") # 라벨 명확히
        text_removal_layout = QVBoxLayout()
        
        self.main_window.remove_text_check = self.create_checkbox_control("텍스트 영역 제거 (인페인팅 적용)", text_removal_layout)
        
        padding_layout = QHBoxLayout()
        padding_layout.addWidget(QLabel("텍스트 마스크 패딩 (작게):")) 
        self.main_window.text_padding = QSpinBox()
        self.main_window.text_padding.setRange(0, 20) 
        self.main_window.text_padding.setValue(3)    
        self.main_window.text_padding.setSingleStep(1)
        padding_layout.addWidget(self.main_window.text_padding)
        text_removal_layout.addLayout(padding_layout)

        inpaint_radius_layout = QHBoxLayout()
        inpaint_radius_layout.addWidget(QLabel("인페인팅 반경:"))
        self.main_window.inpaint_radius = QSpinBox()
        self.main_window.inpaint_radius.setRange(1, 15)
        self.main_window.inpaint_radius.setValue(5) 
        inpaint_radius_layout.addWidget(self.main_window.inpaint_radius)
        text_removal_layout.addLayout(inpaint_radius_layout)

        inpaint_method_layout = QHBoxLayout()
        inpaint_method_layout.addWidget(QLabel("인페인팅 방법:"))
        self.main_window.inpaint_method = QComboBox()
        self.main_window.inpaint_method.addItems(["Telea", "Navier-Stokes (NS)"])
        inpaint_method_layout.addWidget(self.main_window.inpaint_method)
        text_removal_layout.addLayout(inpaint_method_layout)

        text_removal_group.setLayout(text_removal_layout)
        lang_layout.addWidget(text_removal_group) # 수정: lang_layout에 추가

        advanced_ocr_group = QGroupBox("고급 OCR 설정")
        advanced_ocr_layout = QVBoxLayout()
        self.main_window.enhance_ocr = self.create_checkbox_control("향상된 OCR 사용", advanced_ocr_layout)
        self.main_window.enhance_ocr.setChecked(True)
        self.main_window.use_mser = self.create_checkbox_control("MSER 텍스트 후보 검출 사용", advanced_ocr_layout)
        self.main_window.use_mser.setChecked(True) # 기본값은 True로 유지 (사용자 선택 가능)
        advanced_ocr_group.setLayout(advanced_ocr_layout)
        lang_layout.addWidget(advanced_ocr_group) # 수정: lang_layout에 추가
        
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)


        tab_widget.addTab(tab, "OCR 설정")

    def create_contour_tab(self, tab_widget):
        """외곽선 추출 및 필터링 파라미터 탭 생성 (min_area_ratio 수정됨)"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- 기본 외곽선 파라미터 ---
        base_contour_group = QGroupBox("기본 외곽선 파라미터")
        base_contour_layout = QVBoxLayout()
        self.main_window.approx_slider = self.create_slider_control(
            "근사 정확도 (epsilon factor * 1000):", 1, 100, 10, base_contour_layout 
        )
        
        # --- min_area를 min_area_ratio로 변경 ---
        min_area_permille_layout = QHBoxLayout()
        # 레이블 변경: "최소 면적 비율 (0.001 ~ 0.5):" -> "최소 면적 (이미지 대비 만분율, 10~5000):"
        min_area_permille_layout.addWidget(QLabel("최소 면적 (이미지 대비 만분율, 10~5000):"))
        self.main_window.min_area_permille = QSpinBox() # QDoubleSpinBox -> QSpinBox
        self.main_window.min_area_permille.setRange(10, 5000)  # 예: 0.1% (10) ~ 50% (5000)
        self.main_window.min_area_permille.setValue(100)      # 기본값 1% (100)
        self.main_window.min_area_permille.setSingleStep(10)  # 단계 0.1% (10)
        min_area_permille_layout.addWidget(self.main_window.min_area_permille)
        base_contour_layout.addLayout(min_area_permille_layout)
        # --- 변경 완료 ---
        
        base_contour_group.setLayout(base_contour_layout)
        layout.addWidget(base_contour_group)

        # --- 고급 외곽선 필터링 ---
        adv_filter_group = QGroupBox("고급 외곽선 필터링")
        adv_filter_layout = QVBoxLayout()

        border_filter_layout = QHBoxLayout()
        border_filter_layout.addWidget(QLabel("이미지 테두리 최대 면적 비율:"))
        self.main_window.max_area_ratio_to_image = QDoubleSpinBox()
        self.main_window.max_area_ratio_to_image.setRange(0.1, 1.0)
        self.main_window.max_area_ratio_to_image.setValue(0.95) 
        self.main_window.max_area_ratio_to_image.setSingleStep(0.01)
        self.main_window.max_area_ratio_to_image.setDecimals(2)
        border_filter_layout.addWidget(self.main_window.max_area_ratio_to_image)
        adv_filter_layout.addLayout(border_filter_layout)

        border_margin_layout = QHBoxLayout()
        border_margin_layout.addWidget(QLabel("이미지 경계 여백 (픽셀):"))
        self.main_window.border_margin_filter = QSpinBox()
        self.main_window.border_margin_filter.setRange(0, 100)
        self.main_window.border_margin_filter.setValue(10)
        border_margin_layout.addWidget(self.main_window.border_margin_filter)
        adv_filter_layout.addLayout(border_margin_layout)

        hierarchy_filter_layout = QHBoxLayout()
        hierarchy_filter_layout.addWidget(QLabel("최소 자식 윤곽선 수:"))
        self.main_window.min_children_count = QSpinBox()
        self.main_window.min_children_count.setRange(0, 100)
        self.main_window.min_children_count.setValue(2) 
        hierarchy_filter_layout.addWidget(self.main_window.min_children_count)
        adv_filter_layout.addLayout(hierarchy_filter_layout)

        vertices_layout = QHBoxLayout()
        vertices_layout.addWidget(QLabel("꼭짓점 수 (최소/최대):"))
        self.main_window.min_vertices = QSpinBox()
        self.main_window.min_vertices.setRange(3, 50)
        self.main_window.min_vertices.setValue(4)
        vertices_layout.addWidget(self.main_window.min_vertices)
        self.main_window.max_vertices = QSpinBox()
        self.main_window.max_vertices.setRange(10, 500) 
        self.main_window.max_vertices.setValue(100) 
        vertices_layout.addWidget(self.main_window.max_vertices)
        adv_filter_layout.addLayout(vertices_layout)

        aspect_ratio_layout = QHBoxLayout()
        aspect_ratio_layout.addWidget(QLabel("가로세로비 (최소/최대):"))
        self.main_window.min_aspect_ratio = QDoubleSpinBox()
        self.main_window.min_aspect_ratio.setRange(0.01, 10.0)
        self.main_window.min_aspect_ratio.setValue(0.1)
        self.main_window.min_aspect_ratio.setSingleStep(0.01)
        self.main_window.min_aspect_ratio.setDecimals(2)
        aspect_ratio_layout.addWidget(self.main_window.min_aspect_ratio)
        self.main_window.max_aspect_ratio = QDoubleSpinBox()
        self.main_window.max_aspect_ratio.setRange(0.1, 50.0) 
        self.main_window.max_aspect_ratio.setValue(10.0)
        self.main_window.max_aspect_ratio.setSingleStep(0.1)
        self.main_window.max_aspect_ratio.setDecimals(2)
        aspect_ratio_layout.addWidget(self.main_window.max_aspect_ratio)
        adv_filter_layout.addLayout(aspect_ratio_layout)

        solidity_layout = QHBoxLayout()
        solidity_layout.addWidget(QLabel("채움 정도 (Solidity 최소값):"))
        self.main_window.min_solidity = QDoubleSpinBox()
        self.main_window.min_solidity.setRange(0.0, 1.0)
        self.main_window.min_solidity.setValue(0.80) 
        self.main_window.min_solidity.setSingleStep(0.01)
        self.main_window.min_solidity.setDecimals(2)
        solidity_layout.addWidget(self.main_window.min_solidity)
        adv_filter_layout.addLayout(solidity_layout)

        self.main_window.filter_by_rectangle_adv = self.create_checkbox_control("직사각형 형태 필터링 (고급)", adv_filter_layout)
        self.main_window.filter_by_rectangle_adv.setChecked(True) 
        
        rect_ratio_layout = QHBoxLayout()
        rect_ratio_layout.addWidget(QLabel("직사각형 유사도 범위 (최소/최대):"))
        self.main_window.min_rect_ratio_adv = QDoubleSpinBox()
        self.main_window.min_rect_ratio_adv.setRange(0.1, 1.0)
        self.main_window.min_rect_ratio_adv.setValue(0.6) 
        self.main_window.min_rect_ratio_adv.setSingleStep(0.01)
        self.main_window.min_rect_ratio_adv.setDecimals(2)
        rect_ratio_layout.addWidget(self.main_window.min_rect_ratio_adv)

        self.main_window.max_rect_ratio_adv = QDoubleSpinBox()
        self.main_window.max_rect_ratio_adv.setRange(1.0, 2.0)
        self.main_window.max_rect_ratio_adv.setValue(1.4) 
        self.main_window.max_rect_ratio_adv.setSingleStep(0.01)
        self.main_window.max_rect_ratio_adv.setDecimals(2)
        rect_ratio_layout.addWidget(self.main_window.max_rect_ratio_adv)
        adv_filter_layout.addLayout(rect_ratio_layout)

        adv_filter_group.setLayout(adv_filter_layout)
        layout.addWidget(adv_filter_group)

        other_contour_group = QGroupBox("기타 외곽선 처리")
        other_contour_layout = QVBoxLayout()
        self.main_window.merge_overlapping = self.create_checkbox_control("겹치는 외곽선 병합 (후처리)", other_contour_layout)
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("병합 임계값 (Overlap Ratio):"))
        self.main_window.overlap_threshold = QDoubleSpinBox()
        self.main_window.overlap_threshold.setRange(0.01, 1.0)
        self.main_window.overlap_threshold.setValue(0.1) 
        self.main_window.overlap_threshold.setSingleStep(0.01)
        self.main_window.overlap_threshold.setDecimals(2)
        overlap_layout.addWidget(self.main_window.overlap_threshold)
        other_contour_layout.addLayout(overlap_layout)
        other_contour_group.setLayout(other_contour_layout)
        layout.addWidget(other_contour_group)

        detect_group = QGroupBox("벽 및 방 검출 (보조 기능)")
        detect_layout = QVBoxLayout()
        wall_group = QGroupBox("벽 검출")
        wall_layout = QVBoxLayout()
        self.main_window.extract_walls = self.create_checkbox_control("벽 검출", wall_layout)
        self.main_window.min_line_length = self.create_slider_control("최소 선 길이:", 10, 200, 50, wall_layout)
        self.main_window.max_line_gap = self.create_slider_control("최대 선 간격:", 1, 50, 5, wall_layout)
        wall_group.setLayout(wall_layout)
        detect_layout.addWidget(wall_group)

        room_group = QGroupBox("방 검출")
        room_layout = QVBoxLayout()
        self.main_window.extract_rooms = self.create_checkbox_control("방 검출", room_layout)
        min_room_layout = QHBoxLayout()
        min_room_layout.addWidget(QLabel("최소 방 면적:"))
        self.main_window.min_room_area = QSpinBox()
        self.main_window.min_room_area.setRange(100, 100000)
        self.main_window.min_room_area.setValue(1000)
        self.main_window.min_room_area.setSingleStep(100)
        min_room_layout.addWidget(self.main_window.min_room_area)
        room_layout.addLayout(min_room_layout)
        max_room_layout = QHBoxLayout()
        max_room_layout.addWidget(QLabel("최대 방 면적:"))
        self.main_window.max_room_area = QSpinBox()
        self.main_window.max_room_area.setRange(1000, 1000000)
        self.main_window.max_room_area.setValue(100000)
        self.main_window.max_room_area.setSingleStep(1000)
        max_room_layout.addWidget(self.main_window.max_room_area)
        room_layout.addLayout(max_room_layout)
        room_group.setLayout(room_layout)
        detect_layout.addWidget(room_group)
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)

        tab_widget.addTab(tab, "외곽선 추출 및 필터링")


    def get_processing_parameters(self):
        """모든 처리 파라미터 수집 (min_area_ratio 수정됨)"""
        params = {
            # PDF 렌더링 DPI
            'render_dpi': int(self.main_window.render_dpi.currentText()), # DPI 값을 정수로 가져옴
            # 기본 전처리
            'use_grayscale': self.main_window.use_grayscale.isChecked(),
            'threshold': self.main_window.threshold_slider.value(),
            'threshold_value_for_contour': self.main_window.threshold_value_for_contour.value(),
            'thresh_method': self.main_window.thresh_method.currentIndex(),
            'blur_size': self.main_window.blur_slider.value(),
            'blur_type': self.main_window.blur_type.currentIndex(),
            'adaptive_block_size': self.main_window.adaptive_block_size.value(),
            'adaptive_c': self.main_window.adaptive_c.value(),

            # 고급 전처리
            'denoise_strength': self.main_window.denoise_strength.currentIndex(),
            'nlm_h': self.main_window.nlm_h.value(),
            'nlm_template_size': self.main_window.nlm_template_size.value(),
            'nlm_search_size': self.main_window.nlm_search_size.value(),
            'use_clahe': self.main_window.use_clahe.isChecked(),
            'clahe_clip_limit': self.main_window.clahe_clip_limit.value(),
            'clahe_tile_size': self.main_window.clahe_tile_size.value(),
            'use_equalize_hist': self.main_window.use_equalize_hist.isChecked(),
            'remove_hatching': self.main_window.remove_hatching.isChecked(),
            'hatching_kernel_size': self.main_window.hatching_kernel_size.value(),
            'use_canny': self.main_window.use_canny.isChecked(),
            'canny_low': self.main_window.canny_low.value(),
            'canny_high': self.main_window.canny_high.value(),

            # 형태학적 연산
            'morph_operations': self.get_morphology_operations(),
            
            # --- 연결 요소 제거 파라미터 추가 ---
            'remove_small_components': self.main_window.remove_small_components_checkbox.isChecked(),
            'min_component_size': self.main_window.min_component_size_spinbox.value(),

            # 필터
            'use_sharpen': self.main_window.use_sharpen.isChecked(),
            'use_distance_transform': self.main_window.use_distance_transform.isChecked(),
            'distance_type': self.main_window.distance_type.currentIndex(),
            'use_skeletonize': self.main_window.use_skeletonize.isChecked(),
            'use_thinning': self.main_window.use_thinning.isChecked(),
            'use_laplacian': self.main_window.use_laplacian.isChecked(),
            'use_sobel': self.main_window.use_sobel.isChecked(),
            'sobel_direction': self.main_window.sobel_direction.currentIndex(),

            # OCR
            'ocr_langs': ['ko' if self.main_window.lang_korean.isChecked() else None,
                         'en' if self.main_window.lang_english.isChecked() else None],
            'ocr_confidence': self.main_window.ocr_confidence.value(),
            'remove_text': self.main_window.remove_text_check.isChecked(),
            'text_padding': self.main_window.text_padding.value(),
            'enhance_ocr': self.main_window.enhance_ocr.isChecked(),
            'use_mser': self.main_window.use_mser.isChecked(),
            'inpaint_radius': self.main_window.inpaint_radius.value(), 
            'inpaint_method': 'telea' if self.main_window.inpaint_method.currentText() == "Telea" else 'ns',

            # 외곽선 추출 및 필터링
            'approx_epsilon': self.main_window.approx_slider.value() / 1000.0,
            # min_area_ratio -> min_area_permille 값을 읽어 비율로 변환
            'min_area_ratio': self.main_window.min_area_permille.value() / 10000.0,

            # 고급 필터링 파라미터
            'max_area_ratio_to_image': self.main_window.max_area_ratio_to_image.value(),
            'border_margin': self.main_window.border_margin_filter.value(), 
            'min_children_count': self.main_window.min_children_count.value(),
            'min_vertices': self.main_window.min_vertices.value(),
            'max_vertices': self.main_window.max_vertices.value(),
            'min_aspect_ratio': self.main_window.min_aspect_ratio.value(),
            'max_aspect_ratio': self.main_window.max_aspect_ratio.value(),
            'min_solidity': self.main_window.min_solidity.value(),
            'filter_by_rectangle': self.main_window.filter_by_rectangle_adv.isChecked(),
            'min_rect_ratio': self.main_window.min_rect_ratio_adv.value(), 
            'max_rect_ratio': self.main_window.max_rect_ratio_adv.value(), 

            # 기타 외곽선 처리
            'merge_overlapping': self.main_window.merge_overlapping.isChecked(),
            'overlap_threshold': self.main_window.overlap_threshold.value(),

            # 벽 및 방 검출
            'extract_walls': self.main_window.extract_walls.isChecked(),
            'min_line_length': self.main_window.min_line_length.value(),
            'max_line_gap': self.main_window.max_line_gap.value(),
            'extract_rooms': self.main_window.extract_rooms.isChecked(),
            'min_room_area': self.main_window.min_room_area.value(),
            'max_room_area': self.main_window.max_room_area.value(),
        }
        params['ocr_langs'] = [lang for lang in params['ocr_langs'] if lang] 
        return params
    
    def get_morphology_operations(self):
        """UI에서 설정된 모폴로지 연산 목록을 가져옴"""
        morph_ops = []
        # self.morph_op_uis 리스트에는 각 모폴로지 연산 UI 그룹(QGroupBox)들이 들어있어야 합니다.
        logging.debug(f"Number of morphology UI groups (morph_op_uis): {len(self.morph_op_uis)}")

        for i, morph_op_ui in enumerate(self.morph_op_uis): # morph_op_ui는 QGroupBox 입니다.
            logging.debug(f"Processing morph_op_ui group {i}")
            try:
                op_main_layout = morph_op_ui.layout() # QGroupBox의 QVBoxLayout

                # 연산 타입 (첫 번째 QHBoxLayout의 두 번째 위젯)
                type_q_layout = op_main_layout.itemAt(0).layout() # QHBoxLayout for type
                morph_type_combo = type_q_layout.itemAt(1).widget() # QComboBox
                morph_type = morph_type_combo.currentText()
                logging.debug(f"  UI Group {i} - Type: {morph_type}")

                if morph_type == "없음":
                    logging.debug(f"  UI Group {i} - Skipped (Type is '없음')")
                    continue

                # 커널 형태 (두 번째 QHBoxLayout의 두 번째 위젯)
                kernel_shape_q_layout = op_main_layout.itemAt(1).layout() # QHBoxLayout for kernel shape
                kernel_shape_combo = kernel_shape_q_layout.itemAt(1).widget() # QComboBox
                kernel_shape = kernel_shape_combo.currentText()
                logging.debug(f"  UI Group {i} - Kernel Shape: {kernel_shape}")

                # 커널 크기 (세 번째 QHBoxLayout의 두 번째 위젯)
                kernel_size_q_layout = op_main_layout.itemAt(2).layout() # QHBoxLayout for kernel size
                kernel_size_spinbox = kernel_size_q_layout.itemAt(1).widget() # QSpinBox
                kernel_size = kernel_size_spinbox.value()
                logging.debug(f"  UI Group {i} - Kernel Size: {kernel_size}")

                # 반복 횟수 (네 번째 QHBoxLayout의 두 번째 위젯)
                iterations_q_layout = op_main_layout.itemAt(3).layout() # QHBoxLayout for iterations
                iterations_spinbox = iterations_q_layout.itemAt(1).widget() # QSpinBox
                iterations = iterations_spinbox.value()
                logging.debug(f"  UI Group {i} - Iterations: {iterations}")

                operation_details = {
                    'morph_type': MORPH_TYPE_MAP[morph_type],
                    'kernel_shape': KERNEL_SHAPE_MAP[kernel_shape],
                    'morph_size': kernel_size,
                    'morph_iterations': iterations,
                }
                morph_ops.append(operation_details)
                logging.debug(f"  UI Group {i} - Appended to morph_ops: {operation_details}")

            except AttributeError as e:
                logging.error(f"Error accessing widget in get_morphology_operations for UI group {i}: {e}", exc_info=True)
                logging.error(f"Problematic morph_op_ui: {morph_op_ui}")
                continue
            except Exception as ex: # 다른 예외도 로깅
                logging.error(f"Generic error processing UI group {i} in get_morphology_operations: {ex}", exc_info=True)
                continue

        logging.info(f"Final collected morphology operations: {morph_ops}")
        return morph_ops
    
    def set_parameters(self, params):
        """파라미터 값을 UI 컨트롤에 설정 (min_area_ratio 수정됨)"""
        try:
            # PDF 렌더링 DPI
            self.main_window.render_dpi.setCurrentText(str(params.get('render_dpi', 300)))
            # 기본 전처리
            self.main_window.use_grayscale.setChecked(params.get('use_grayscale', True))
            self.main_window.threshold_slider.setValue(params.get('threshold', 127))
            self.main_window.threshold_value_for_contour.setValue(params.get('threshold_value_for_contour', 127))
            self.main_window.thresh_method.setCurrentIndex(params.get('thresh_method', 6))
            self.main_window.blur_slider.setValue(params.get('blur_size', 5))
            self.main_window.blur_type.setCurrentIndex(params.get('blur_type', 0))
            self.main_window.adaptive_block_size.setValue(params.get('adaptive_block_size', 11))
            self.main_window.adaptive_c.setValue(params.get('adaptive_c', 2))

            # 고급 전처리
            self.main_window.denoise_strength.setCurrentIndex(params.get('denoise_strength', 0))
            self.main_window.nlm_h.setValue(params.get('nlm_h', 10))
            self.main_window.nlm_template_size.setValue(params.get('nlm_template_size', 7))
            self.main_window.nlm_search_size.setValue(params.get('nlm_search_size', 21))
            self.main_window.use_clahe.setChecked(params.get('use_clahe', False))
            self.main_window.clahe_clip_limit.setValue(params.get('clahe_clip_limit', 2.0))
            self.main_window.clahe_tile_size.setValue(params.get('clahe_tile_size', 8))
            self.main_window.use_equalize_hist.setChecked(params.get('use_equalize_hist', False))
            self.main_window.remove_hatching.setChecked(params.get('remove_hatching', False))
            self.main_window.hatching_kernel_size.setValue(params.get('hatching_kernel_size', 5))
            self.main_window.use_canny.setChecked(params.get('use_canny', False))
            self.main_window.canny_low.setValue(params.get('canny_low', 50))
            self.main_window.canny_high.setValue(params.get('canny_high', 150))

            # 형태학적 연산
            morph_ops = params.get('morph_operations', [])
            self.set_morphology_operations(morph_ops)
            self.morph_preset_combo.setCurrentIndex(0) # "사용자 정의" 선택
            self.main_window.remove_small_components_checkbox.setChecked(params.get('remove_small_components', False))
            self.main_window.min_component_size_spinbox.setValue(params.get('min_component_size', 100))

            # 필터
            self.main_window.use_sharpen.setChecked(params.get('use_sharpen', False))
            self.main_window.use_distance_transform.setChecked(params.get('use_distance_transform', False))
            self.main_window.distance_type.setCurrentIndex(params.get('distance_type', 0))
            self.main_window.use_skeletonize.setChecked(params.get('use_skeletonize', False))
            self.main_window.use_thinning.setChecked(params.get('use_thinning', False))
            self.main_window.use_laplacian.setChecked(params.get('use_laplacian', False))
            self.main_window.use_sobel.setChecked(params.get('use_sobel', False))
            self.main_window.sobel_direction.setCurrentIndex(params.get('sobel_direction', 0))

            # OCR
            ocr_langs = params.get('ocr_langs', ['ko', 'en'])
            self.main_window.lang_korean.setChecked('ko' in ocr_langs)
            self.main_window.lang_english.setChecked('en' in ocr_langs)
            self.main_window.ocr_confidence.setValue(params.get('ocr_confidence',0.5))
            self.main_window.remove_text_check.setChecked(params.get('remove_text', False))
            self.main_window.text_padding.setValue(params.get('text_padding', 3)) # 기본값 수정
            self.main_window.enhance_ocr.setChecked(params.get('enhance_ocr', True))
            self.main_window.use_mser.setChecked(params.get('use_mser', True))
            self.main_window.inpaint_radius.setValue(params.get('inpaint_radius', 5)) 
            inpaint_method_str = params.get('inpaint_method', 'telea') 
            self.main_window.inpaint_method.setCurrentText("Telea" if inpaint_method_str == 'telea' else "Navier-Stokes (NS)")
            
            # 외곽선 추출 및 필터링
            self.main_window.approx_slider.setValue(int(params.get('approx_epsilon', 0.01) * 1000.0))
            # min_area_ratio 값을 읽어 만분율로 변환하여 min_area_permille에 설정
            self.main_window.min_area_permille.setValue(int(params.get('min_area_ratio', 0.01) * 10000.0))

            self.main_window.max_area_ratio_to_image.setValue(params.get('max_area_ratio_to_image', 0.95))
            self.main_window.border_margin_filter.setValue(params.get('border_margin', 10))
            self.main_window.min_children_count.setValue(params.get('min_children_count', 2))
            self.main_window.min_vertices.setValue(params.get('min_vertices', 4))
            self.main_window.max_vertices.setValue(params.get('max_vertices', 100))
            self.main_window.min_aspect_ratio.setValue(params.get('min_aspect_ratio', 0.1))
            self.main_window.max_aspect_ratio.setValue(params.get('max_aspect_ratio', 10.0))
            self.main_window.min_solidity.setValue(params.get('min_solidity', 0.80))
            self.main_window.filter_by_rectangle_adv.setChecked(params.get('filter_by_rectangle', True))
            self.main_window.min_rect_ratio_adv.setValue(params.get('min_rect_ratio', 0.6))
            self.main_window.max_rect_ratio_adv.setValue(params.get('max_rect_ratio', 1.4))
            
            self.main_window.merge_overlapping.setChecked(params.get('merge_overlapping', False))
            self.main_window.overlap_threshold.setValue(params.get('overlap_threshold', 0.1))

            # 벽 및 방 검출
            self.main_window.extract_walls.setChecked(params.get('extract_walls', False))
            self.main_window.min_line_length.setValue(params.get('min_line_length', 50))
            self.main_window.max_line_gap.setValue(params.get('max_line_gap', 5))
            self.main_window.extract_rooms.setChecked(params.get('extract_rooms', False))
            self.main_window.min_room_area.setValue(params.get('min_room_area', 1000))
            self.main_window.max_room_area.setValue(params.get('max_room_area', 100000))

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"파라미터 설정 중 오류: {str(e)}")

    def set_morphology_operations(self, morph_ops):
        """파라미터에서 모폴로지 연산 목록을 읽어 UI에 설정"""
        logging.info(f"Setting morphology operations from params: {morph_ops}")
        # 기존 UI 전부 지우고
        # 기존 UI를 제거할 때 self.morph_op_uis에서도 제거해야 함
        while self.morph_op_uis:
            ui_to_remove = self.morph_op_uis.pop() # 뒤에서부터 제거
            self.morph_ops_layout.removeWidget(ui_to_remove)
            ui_to_remove.deleteLater()
        # self.morph_op_uis = [] # 이미 위에서 비워짐

        logging.debug(f"Cleared existing morph UIs. Number of UIs to add: {len(morph_ops)}")

        for i, op in enumerate(morph_ops):
            logging.debug(f"Setting UI for operation {i}: {op}")
            self.add_morph_operation_ui() # 새 UI 그룹 추가 (이 안에서 self.morph_op_uis에 append됨)

            if not self.morph_op_uis or len(self.morph_op_uis) <= i: # 방어 코드
                logging.error(f"Failed to add UI group for operation {i}. self.morph_op_uis length: {len(self.morph_op_uis)}")
                continue

            morph_op_ui = self.morph_op_uis[i] # 방금 추가된 UI 그룹 (또는 순서대로 i번째)
                                            # add_morph_operation_ui가 self.morph_op_uis.append를 하므로,
                                            # self.morph_op_uis[-1]이 방금 추가된 것임.
                                            # 하지만 루프를 돌며 여러개를 생성하므로 인덱스 i로 접근하는 것이 더 명확할 수 있으나,
                                            # add_morph_operation_ui를 호출할 때마다 리스트가 늘어나므로,
                                            # morph_op_ui = self.morph_op_uis[-1] 가 더 안전함. 여기서는 -1을 사용.
            morph_op_ui = self.morph_op_uis[-1]
            grp_layout = morph_op_ui.layout()

            try:
                # 위젯 접근 (기존과 동일)
                type_layout = grp_layout.itemAt(0).layout()
                morph_type_combo = type_layout.itemAt(1).widget()

                kernel_layout = grp_layout.itemAt(1).layout()
                kernel_shape_combo = kernel_layout.itemAt(1).widget()

                size_layout = grp_layout.itemAt(2).layout()
                kernel_size_spinbox = size_layout.itemAt(1).widget()

                iter_layout = grp_layout.itemAt(3).layout()
                iterations_spinbox = iter_layout.itemAt(1).widget()

                # 값 설정
                op_type_str = MORPH_TYPE_MAP_REVERSE.get(op.get('morph_type', 0), "없음")
                op_shape_str = KERNEL_SHAPE_MAP_REVERSE.get(op.get('kernel_shape', 0), "사각형")
                op_size_val = op.get('morph_size', 3)
                op_iter_val = op.get('morph_iterations', 1)

                morph_type_combo.setCurrentText(op_type_str)
                kernel_shape_combo.setCurrentText(op_shape_str)
                kernel_size_spinbox.setValue(op_size_val)
                iterations_spinbox.setValue(op_iter_val)
                logging.debug(f"  Set UI values for op {i}: Type={op_type_str}, Shape={op_shape_str}, Size={op_size_val}, Iter={op_iter_val}")

                # visibility 콜백도 동일하게 연결 (add_morph_operation_ui 에서 이미 처리됨)
                # 이 부분은 add_morph_operation_ui 내부에서 처리되므로 중복 호출할 필요 없음
                # def update_kernel_visibility(): ...
                # morph_type_combo.currentIndexChanged.connect(update_kernel_visibility)
                # update_kernel_visibility() # 호출 필요 (add_morph_operation_ui에서 기본 상태 설정 후 여기서 다시 값에 맞게)

                # add_morph_operation_ui에서 연결된 update_kernel_visibility 함수가
                # 여기서 설정된 값에 따라 UI를 업데이트 하도록 수동으로 한번 더 호출해주는 것이 좋을 수 있습니다.
                # 또는 add_morph_operation_ui의 update_kernel_visibility 함수가 이미 연결되어 있으므로
                # morph_type_combo.setCurrentText 호출 시 자동으로 트리거될 수 있습니다.
                # 명시적으로 호출하려면 해당 콤보박스의 currentIndexChanged 시그널에 연결된 슬롯을 직접 호출하거나,
                # 슬롯 내용을 가져와 실행해야 합니다.
                # 가장 간단한 방법은 add_morph_operation_ui 내부의 update_kernel_visibility 함수를
                # morph_type_combo.currentIndexChanged.emit(morph_type_combo.currentIndex()) 와 같이 강제 실행하는 것.
                # 또는, 해당 함수를 직접 호출할 수 있도록 만들 수 있습니다.
                # 여기서는 add_morph_operation_ui에서 이미 update_kernel_visibility()를 호출하므로,
                # setCurrentText 이후 다시 호출할 필요는 없을 수 있습니다. (Qt 시그널/슬롯 동작 확인 필요)

            except AttributeError as e:
                logging.error(f"Error setting UI values for operation {i}: {e}", exc_info=True)
            except Exception as ex:
                logging.error(f"Generic error setting UI values for operation {i}: {ex}", exc_info=True)
        logging.info("Finished setting morphology operations to UI.")

    def apply_morphology_preset(self, index):
        """모폴로지 연산 프리셋을 적용"""
        presets = {
            "노이즈 제거": [
                {"morph_type": MORPH_TYPE_MAP["열림"], "kernel_shape": KERNEL_SHAPE_MAP["사각형"], "morph_size": 3, "morph_iterations": 1},
                {"morph_type": MORPH_TYPE_MAP["닫힘"], "kernel_shape": KERNEL_SHAPE_MAP["사각형"], "morph_size": 3, "morph_iterations": 1},
                {"morph_type": MORPH_TYPE_MAP["열림"], "kernel_shape": KERNEL_SHAPE_MAP["사각형"], "morph_size": 5, "morph_iterations": 1},
            ],
            "선 연결": [
                {"morph_type": MORPH_TYPE_MAP["닫힘"], "kernel_shape": KERNEL_SHAPE_MAP["사각형"], "morph_size": 5, "morph_iterations": 1},
                {"morph_type": MORPH_TYPE_MAP["팽창"], "kernel_shape": KERNEL_SHAPE_MAP["사각형"], "morph_size": 3, "morph_iterations": 1},
            ],
            "윤곽선 정제": [
                {"morph_type": MORPH_TYPE_MAP["닫힘"], "kernel_shape": KERNEL_SHAPE_MAP["타원"], "morph_size": 5, "morph_iterations": 1},
                {"morph_type": MORPH_TYPE_MAP["열림"], "kernel_shape": KERNEL_SHAPE_MAP["타원"], "morph_size": 3, "morph_iterations": 1},
                {"morph_type": MORPH_TYPE_MAP["팽창"], "kernel_shape": KERNEL_SHAPE_MAP["사각형"], "morph_size": 1, "morph_iterations": 2},
            ],
            "사용자 정의": [],
        }

        selected_preset = self.morph_preset_combo.currentText()
        self.set_morphology_operations(presets.get(selected_preset, []))

    def create_slider_control(self, label, min_val, max_val, default, layout, step=1):
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(label))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setSingleStep(step)
        if step > 1: 
             slider.setTickInterval(step)
             slider.setTickPosition(QSlider.TicksBelow)
        hbox.addWidget(slider)
        value_label = QLabel(str(default))
        slider.valueChanged.connect(lambda v, lbl=value_label, s=slider: self._update_slider_label(lbl, s, step))
        self._update_slider_label(value_label, slider, step)
        hbox.addWidget(value_label)
        layout.addLayout(hbox)
        return slider

    def _update_slider_label(self, label_widget, slider_widget, step):
        val = slider_widget.value()
        if step > 1 and val % step != 0 : 
            val = round(val / step) * step
        label_widget.setText(str(val))


    def create_combobox_control(self, label, items, layout):
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(label))
        combo = QComboBox()
        combo.addItems(items)
        hbox.addWidget(combo)
        layout.addLayout(hbox)
        return combo

    def create_checkbox_control(self, label, layout):
        checkbox = QCheckBox(label)
        layout.addWidget(checkbox)
        return checkbox