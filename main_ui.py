"""
건축 도면 외곽선 추출 도구 - 메인 UI 모듈
이 모듈은 애플리케이션의 메인 윈도우와 기본 UI 프레임워크를 정의합니다.
"""

import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QSlider,
                           QGroupBox, QComboBox, QSpinBox, QCheckBox,
                           QTabWidget, QScrollArea, QDoubleSpinBox,
                           QSplitter, QApplication, QMessageBox, QLineEdit)
from PySide6.QtGui import QPixmap, QImage, QAction, QTransform, QCursor
from PySide6.QtCore import Qt, Slot, Signal, QThread, QObject, QPointF

from ui_tabs import TabManager
from ui_handlers import ActionHandlers
from preprocess import Preprocessor
from functional import ContourExtractor

class OCRWorker(QObject):
    """OCR 처리를 위한 워커 스레드"""
    finished = Signal()
    results = Signal(list)

    def __init__(self, preprocessor, image, enhance=True, use_mser=True):
        super().__init__()
        self.preprocessor = preprocessor
        self.image = image
        self.enhance = enhance
        self.use_mser = use_mser

    def run(self):
        """OCR 실행"""
        try:
            results = self.preprocessor.detect_text(self.image, self.enhance, self.use_mser)
            self.results.emit(results)
        except Exception as e:
            print(f"OCR 오류: {e}")
        finally:
            self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("건축 도면 외곽선 추출 도구")
        self.setMinimumSize(1400, 900)

        # 초기 속성 설정
        self.pdf_path = None
        self.original_image = None
        self.processed_image = None
        self.text_overlay_image = None
        self.contour_image = None
        self.wall_image = None
        self.room_image = None
        self.current_page = 0
        self.total_pages = 0
        self.preprocessor = Preprocessor()
        self.extractor = ContourExtractor()
        self.text_regions = []
        self.processing_history = []

        self.image_scale = 1.0
        self.image_offset = QPointF(0, 0)
        self.is_dragging = False
        self.drag_start_pos = QPointF(0, 0)
        self.current_display_image = None # 현재 표시되는 이미지 저장
        self.optimized_room_image = None
        # 탭 관리자 초기화
        self.tab_manager = TabManager(self)

        # 이벤트 핸들러 초기화
        self.handlers = ActionHandlers(self)

        # UI 설정
        self.setup_ui()

    def setup_ui(self):
        # 중앙 위젯 및 메인 레이아웃
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 메뉴 바 생성
        self.create_menu_bar()

        # 상단 제어 바
        top_bar = QHBoxLayout()

        # 파일 선택
        self.load_btn = QPushButton("PDF 불러오기")
        self.load_btn.clicked.connect(self.handlers.load_pdf)
        top_bar.addWidget(self.load_btn)

        # 페이지 네비게이션
        self.prev_btn = QPushButton("이전 페이지")
        self.prev_btn.clicked.connect(self.handlers.prev_page)
        self.prev_btn.setEnabled(False)
        top_bar.addWidget(self.prev_btn)

        self.page_label = QLabel("페이지: 0/0")
        top_bar.addWidget(self.page_label)

        self.next_btn = QPushButton("다음 페이지")
        self.next_btn.clicked.connect(self.handlers.next_page)
        self.next_btn.setEnabled(False)
        top_bar.addWidget(self.next_btn)

        # 처리 버튼
        self.process_btn = QPushButton("처리하기")
        self.process_btn.clicked.connect(self.handlers.process_image)
        self.process_btn.setEnabled(False)
        top_bar.addWidget(self.process_btn)

        # OCR 버튼
        self.ocr_btn = QPushButton("텍스트 감지 (OCR)")
        self.ocr_btn.clicked.connect(self.handlers.detect_text)
        self.ocr_btn.setEnabled(False)
        top_bar.addWidget(self.ocr_btn)

        # 저장 버튼
        self.save_btn = QPushButton("결과 저장")
        self.save_btn.clicked.connect(self.handlers.save_result)
        self.save_btn.setEnabled(False)
        top_bar.addWidget(self.save_btn)

        # 실행 취소 버튼
        self.undo_btn = QPushButton("실행 취소")
        self.undo_btn.clicked.connect(self.handlers.undo_processing)
        self.undo_btn.setEnabled(False)
        top_bar.addWidget(self.undo_btn)

        main_layout.addLayout(top_bar)

        workflow_bar = QHBoxLayout()
        self.save_workflow_btn = QPushButton("워크플로우 저장")
        self.save_workflow_btn.clicked.connect(self.handlers.save_workflow)
        workflow_bar.addWidget(self.save_workflow_btn)

        self.load_workflow_btn = QPushButton("워크플로우 로드")
        self.load_workflow_btn.clicked.connect(self.handlers.load_workflow)
        workflow_bar.addWidget(self.load_workflow_btn)

        self.run_workflow_btn = QPushButton("워크플로우 실행")
        self.run_workflow_btn.clicked.connect(self.handlers.process_image)  # 기존 process_image 함수 재사용
        workflow_bar.addWidget(self.run_workflow_btn)

        main_layout.addLayout(workflow_bar)
        # 분할기로 콘텐츠 영역 구성
        self.splitter = QSplitter(Qt.Horizontal)

        # 이미지 뷰 영역
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)

        self.image_view = QLabel("이미지가 여기에 표시됩니다")
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setStyleSheet("border: 1px solid #ccc; background-color: #f5f5f5;")
        self.image_view.setScaledContents(False)  # 크기 자동 조정 비활성화

        # 마우스 이벤트 핸들러
        self.image_view.mousePressEvent = self.image_mouse_press
        self.image_view.mouseMoveEvent = self.image_mouse_move
        self.image_view.mouseReleaseEvent = self.image_mouse_release

        image_layout.addWidget(self.image_view)

        # 뷰 선택 및 확대/축소
        view_layout = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "원본 이미지", "전처리된 이미지", "텍스트 영역 표시",
            "외곽선 추출", "추출된 벽", "추출된 방", "최적화된 방"
        ])
        self.view_combo.currentIndexChanged.connect(self.handlers.update_view)
        view_layout.addWidget(QLabel("보기:"))
        view_layout.addWidget(self.view_combo)

        view_layout.addStretch()
        view_layout.addWidget(QLabel("확대/축소:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.handlers.zoom_image)
        view_layout.addWidget(self.zoom_slider)
        self.zoom_label = QLabel("100%")
        view_layout.addWidget(self.zoom_label)

        image_layout.addLayout(view_layout)
        self.splitter.addWidget(image_widget)

        # 파라미터 패널
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        self.param_tabs = QTabWidget()

        # 탭 생성
        self.tab_manager.create_all_tabs(self.param_tabs)

        param_layout.addWidget(self.param_tabs)
        self.splitter.addWidget(param_widget)
        self.splitter.setSizes([800, 400])

        main_layout.addWidget(self.splitter)
        self.statusBar().showMessage("준비 완료")

    def create_menu_bar(self):
        """메뉴 바 생성"""
        menu_bar = self.menuBar()

        # 파일 메뉴
        file_menu = menu_bar.addMenu("파일")
        open_action = QAction("PDF 불러오기", self)
        open_action.triggered.connect(self.handlers.load_pdf)
        file_menu.addAction(open_action)

        save_action = QAction("이미지 저장", self)
        save_action.triggered.connect(self.handlers.save_result)
        file_menu.addAction(save_action)

        export_dxf_action = QAction("DXF 내보내기", self)
        export_dxf_action.triggered.connect(self.handlers.export_to_dxf)
        file_menu.addAction(export_dxf_action)

        file_menu.addSeparator()

        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 편집 메뉴
        edit_menu = menu_bar.addMenu("편집")

        undo_action = QAction("실행 취소", self)
        undo_action.triggered.connect(self.handlers.undo_processing)
        edit_menu.addAction(undo_action)

        # 워크플로우 메뉴
        workflow_menu = menu_bar.addMenu("워크플로우")

        save_workflow_action = QAction("워크플로우 저장", self)
        save_workflow_action.triggered.connect(self.handlers.save_workflow)
        workflow_menu.addAction(save_workflow_action)

        load_workflow_action = QAction("워크플로우 로드", self)
        load_workflow_action.triggered.connect(self.handlers.load_workflow)
        workflow_menu.addAction(load_workflow_action)

        run_workflow_action = QAction("워크플로우 실행", self)
        run_workflow_action.triggered.connect(self.handlers.process_image)  # 기존 process_image 사용
        workflow_menu.addAction(run_workflow_action)

        # 보기 메뉴
        view_menu = menu_bar.addMenu("보기")

        zoom_in_action = QAction("확대", self)
        zoom_in_action.triggered.connect(lambda: self.zoom_slider.setValue(self.zoom_slider.value()+10))
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("축소", self)
        zoom_out_action.triggered.connect(lambda: self.zoom_slider.setValue(self.zoom_slider.value()-10))
        view_menu.addAction(zoom_out_action)

        reset_zoom_action = QAction("줌 초기화", self)
        reset_zoom_action.triggered.connect(lambda: self.zoom_slider.setValue(100))
        view_menu.addAction(reset_zoom_action)

        # 도구 메뉴
        tools_menu = menu_bar.addMenu("도구")

        ocr_action = QAction("텍스트 감지 (OCR)", self)
        ocr_action.triggered.connect(self.handlers.detect_text)
        tools_menu.addAction(ocr_action)

        # 도움말 메뉴
        help_menu = menu_bar.addMenu("도움말")

        about_action = QAction("정보", self)
        about_action.triggered.connect(self.handlers.show_about)
        help_menu.addAction(about_action)

    def image_mouse_press(self, event):
        if event.button() == Qt.LeftButton and self.image_scale > 1.0:
            self.is_dragging = True
            self.drag_start_pos = event.position()
            self.image_view.setCursor(Qt.ClosedHandCursor)

    def image_mouse_move(self, event):
        if self.is_dragging:
            new_pos = event.position()
            delta = new_pos - self.drag_start_pos
            self.image_offset += delta
            self.drag_start_pos = new_pos
            self.update_image_display()

    def image_mouse_release(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.image_view.setCursor(Qt.OpenHandCursor)

    def update_image_display(self):
        """이미지 뷰 업데이트 (줌 및 이동 적용)"""
        if self.original_image is None:
            return

        pixmap = self.convert_cv_to_pixmap(self.current_display_image)
        if pixmap is None:
            return

        transform = QTransform()
        transform.scale(self.image_scale, self.image_scale)
        transform.translate(self.image_offset.x(), self.image_offset.y())
        transformed_pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)
        self.image_view.setPixmap(transformed_pixmap)

    def display_image(self, img_cv):
        """OpenCV 이미지를 QLabel에 표시 (줌 및 이동)"""
        if img_cv is None:
            self.image_view.setText("표시할 이미지가 없습니다.")
            return

        self.current_display_image = img_cv.copy()  # 복사본 사용
        self.image_scale = self.zoom_slider.value() / 100.0
        self.image_offset = QPointF(0, 0)  # 오프셋 초기화
        self.update_image_display()
        self.image_view.setCursor(Qt.OpenHandCursor)  # 드래그 가능 표시

    def zoom_image(self):
        """줌 슬라이더 값 변경 시 호출되어 이미지 뷰 업데이트"""
        self.image_scale = self.zoom_slider.value() / 100.0
        self.zoom_label.setText(f"{int(self.image_scale * 100)}%")
        self.update_image_display()

    def convert_cv_to_pixmap(self, img_cv):
        """OpenCV 이미지를 QPixmap으로 변환"""
        try:
            if not isinstance(img_cv, np.ndarray):
                self.main_window.image_view.setText("잘못된 이미지 데이터입니다.")
                return None

            img_display = img_cv.copy()

            if img_display.dtype != np.uint8:
                if np.max(img_display) <= 1.0 and img_display.dtype != np.bool_:
                    img_display = (img_display * 255).astype(np.uint8)
                else:
                    img_display = np.clip(img_display, 0, 255).astype(np.uint8)

            if len(img_display.shape) == 2:
                img_format = QImage.Format_Grayscale8
                bytes_per_line = img_display.strides[0]
                q_image = QImage(img_display.data, img_display.shape[1], img_display.shape[0], bytes_per_line, img_format)

            elif len(img_display.shape) == 3 and img_display.shape[2] == 3:
                img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                h, w = img_display_rgb.shape[:2]
                bytes_per_line = img_display_rgb.strides[0]
                q_image = QImage(img_display_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            elif len(img_display.shape) == 3 and img_display.shape[2] == 4:
                img_display_rgba = cv2.cvtColor(img_display, cv2.COLOR_BGRA2RGBA)
                h, w = img_display_rgba.shape[:2]
                bytes_per_line = img_display_rgba.strides[0]
                q_image = QImage(img_display_rgba.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
            else:
                self.main_window.image_view.setText("지원하지 않는 이미지 형식입니다.")
                return None

            if q_image.isNull():
                self.main_window.image_view.setText("QImage 변환 실패")
                return None

            pixmap = QPixmap.fromImage(q_image)
            return pixmap

        except Exception as e:
            self.handlers.handle_exception("이미지 표시 중 예외 발생", e)
            return None