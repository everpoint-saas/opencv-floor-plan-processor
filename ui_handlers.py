# ui_handlers.py 수정 부분

import os
import cv2
import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QThread, Slot
from geometry_optimizer import GeometryOptimizer
import logging # 로깅 임포트 추가 (만약 없다면)

# main_ui에서 OCRWorker를 가져오기 위함 (현재 코드에서는 직접 사용하지 않음)
# from main_ui import OCRWorker

class ActionHandlers:
    """
    애플리케이션의 이벤트 핸들러 및 기능을 구현하는 클래스
    """
    def __init__(self, main_window):
        self.main_window = main_window
        self.main_window.ocr_thread = None
        self.main_window.ocr_worker = None

    def load_pdf(self):
        """PDF 파일 로드 및 첫 페이지 변환 (DPI 설정 로직 추가됨)"""
        try:
            # --- DPI 설정 로직 추가 ---
            # 1. UI의 DPI 콤보박스에서 현재 선택된 텍스트 값을 가져옵니다.
            selected_dpi_text = self.main_window.render_dpi.currentText()
            try:
                # 2. 가져온 텍스트 값을 정수로 변환합니다.
                selected_dpi = int(selected_dpi_text)
                # 3. Preprocessor의 set_resolution 메서드를 호출하여 DPI 값을 설정합니다.
                self.main_window.preprocessor.set_resolution(selected_dpi)
                logging.info(f"PDF rendering DPI set to: {selected_dpi}")
            except ValueError:
                # 만약 콤보박스의 값이 숫자가 아닌 경우 오류 처리
                logging.error(f"Invalid DPI value selected: {selected_dpi_text}. Using default 300 DPI.")
                # 오류 발생 시 기본값(예: 300)으로 설정
                self.main_window.preprocessor.set_resolution(300)
                # 사용자에게 알림 (선택 사항)
                QMessageBox.warning(self.main_window, "DPI 설정 오류",
                                     f"잘못된 DPI 값({selected_dpi_text})이 선택되어 기본값 300으로 설정합니다.")
            # --- DPI 설정 로직 추가 완료 ---

            # --- 기존 PDF 파일 선택 로직 ---
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window, "PDF 파일 선택", "", "PDF Files (*.pdf)"
            )
            if not file_path:
                return # 사용자가 파일 선택을 취소한 경우 함수 종료

            # --- 기존 PDF 로드 및 처리 로직 ---
            self.main_window.pdf_path = file_path
            self.main_window.statusBar().showMessage(f"PDF 파일 로드 중: {os.path.basename(file_path)}")
            QApplication.processEvents() # UI가 멈추지 않도록 이벤트 처리

            # PDF 로드 시도 (이제 설정된 DPI가 내부적으로 사용됨)
            success, self.main_window.total_pages = self.main_window.preprocessor.load_pdf(file_path)

            if success:
                self.main_window.current_page = 0
                self.update_page_navigation() # 페이지 네비게이션 UI 업데이트
                self.load_current_page()      # 첫 페이지 이미지 로드 (변경된 DPI 적용됨)
                # 버튼 활성화
                self.main_window.process_btn.setEnabled(True)
                self.main_window.ocr_btn.setEnabled(True)
                self.main_window.view_combo.setEnabled(True)
                # 워크플로우 버튼도 활성화 가능 (필요 시)
                self.main_window.save_workflow_btn.setEnabled(True)
                self.main_window.load_workflow_btn.setEnabled(True)
                self.main_window.run_workflow_btn.setEnabled(True)

                self.main_window.statusBar().showMessage(f"PDF 로드 완료: {os.path.basename(file_path)} ({self.main_window.total_pages} 페이지)")
            else:
                # PDF 로드 실패 시 처리
                self.main_window.image_view.setText("PDF 파일을 불러오는 데 실패했습니다.")
                QMessageBox.critical(self.main_window, "오류", "PDF 파일을 불러오는 데 실패했습니다.")
                self.main_window.statusBar().showMessage("PDF 로드 실패.")

        except Exception as e:
            # 예상치 못한 다른 오류 처리
            self.handle_exception("PDF 로드 중 예외 발생", e)

    # ... (ActionHandlers 클래스의 나머지 메서드들은 그대로 유지) ...

    def process_image(self):
        """이미지 처리 로직"""
        if self.main_window.original_image is None:
            QMessageBox.warning(self.main_window, "경고", "먼저 PDF 파일을 로드하세요.")
            return

        self.main_window.statusBar().showMessage("이미지 처리 중...")
        QApplication.processEvents()

        try:
            params = self.main_window.tab_manager.get_processing_parameters()

            # 실행 취소를 위한 히스토리 저장
            current_image_to_history = self.main_window.processed_image if self.main_window.processed_image is not None else self.main_window.original_image
            if current_image_to_history is not None:
                 # 히스토리에는 복사본 저장
                 self.main_window.processing_history.append(current_image_to_history.copy())

            # Preprocessor를 통해 이미지 처리 실행
            self.main_window.processed_image = self.main_window.preprocessor.process_image(
                self.main_window.original_image.copy(), params # 원본 이미지 복사본 전달
            )
            if self.main_window.processed_image is None:
                self.main_window.statusBar().showMessage("전처리 실패. 결과 이미지가 없습니다.")
                return

            # 윤곽선 및 기타 특징 추출
            self.process_contours_and_features(params)

            # 결과 뷰 업데이트 및 버튼 상태 변경
            self.update_view()
            self.main_window.save_btn.setEnabled(True)
            self.main_window.undo_btn.setEnabled(len(self.main_window.processing_history) > 0)
            self.main_window.statusBar().showMessage("이미지 처리 완료")

        except Exception as e:
            self.handle_exception("이미지 처리 중 예외 발생", e)

    @Slot()
    def _on_ocr_thread_finished(self):
        """OCR 스레드 종료 시 호출되는 슬롯"""
        if self.main_window.ocr_worker is not None:
            self.main_window.ocr_worker.deleteLater()
            self.main_window.ocr_worker = None

        if self.main_window.ocr_thread is not None:
            self.main_window.ocr_thread.wait()
            self.main_window.ocr_thread.deleteLater()
            self.main_window.ocr_thread = None

        if hasattr(self.main_window, 'ocr_btn') and self.main_window.ocr_btn is not None:
             self.main_window.ocr_btn.setEnabled(True)


    def detect_text(self):
        """OCR 텍스트 감지"""
        if self.main_window.original_image is None:
            QMessageBox.warning(self.main_window, "경고", "먼저 PDF 파일을 로드하세요.")
            return

        if self.main_window.ocr_thread is not None and self.main_window.ocr_thread.isRunning():
            self.main_window.statusBar().showMessage("OCR이 이미 실행 중입니다.")
            return

        if self.main_window.ocr_thread is not None:
            self._on_ocr_thread_finished()

        self.main_window.statusBar().showMessage("텍스트 감지 중... (시간이 소요될 수 있습니다)")
        QApplication.processEvents()

        try:
            from main_ui import OCRWorker # 이 부분 확인 필요

            ocr_langs = self.main_window.tab_manager.get_processing_parameters().get('ocr_langs', ['ko', 'en'])
            self.main_window.preprocessor.set_ocr_languages(ocr_langs)
            enhance = self.main_window.tab_manager.get_processing_parameters().get('enhance_ocr', True)
            use_mser = self.main_window.tab_manager.get_processing_parameters().get('use_mser', True)

            self.main_window.ocr_thread = QThread(self.main_window)
            self.main_window.ocr_worker = OCRWorker(
                self.main_window.preprocessor,
                self.main_window.original_image.copy(),
                enhance,
                use_mser
            )
            self.main_window.ocr_worker.moveToThread(self.main_window.ocr_thread)

            self.main_window.ocr_thread.started.connect(self.main_window.ocr_worker.run)
            self.main_window.ocr_worker.results.connect(self.handle_ocr_results)
            self.main_window.ocr_worker.finished.connect(self.main_window.ocr_thread.quit)
            self.main_window.ocr_thread.finished.connect(self._on_ocr_thread_finished)

            self.main_window.ocr_thread.start()
            self.main_window.ocr_btn.setEnabled(False)

        except RuntimeError as r_err:
            self.handle_exception(f"OCR 스레드 시작 중 런타임 오류: {r_err}", r_err)
            self._on_ocr_thread_finished()
        except Exception as e:
            self.handle_exception("OCR 시작 중 예외 발생", e)
            self._on_ocr_thread_finished()


    @Slot(list)
    def handle_ocr_results(self, results):
        """OCR 결과 처리"""
        try:
            confidence_threshold = self.main_window.tab_manager.get_processing_parameters().get('ocr_confidence', 0.5)
            # Preprocessor의 text_regions 업데이트
            self.main_window.preprocessor.text_regions = [r for r in results if r[2] >= confidence_threshold]
            # MainWindow의 text_regions도 업데이트 (필요하다면)
            self.main_window.text_regions = self.main_window.preprocessor.text_regions

            if self.main_window.original_image is not None:
                # 원본 이미지에 텍스트 영역을 그린 이미지를 생성
                self.main_window.text_overlay_image = self.main_window.preprocessor.draw_text_regions(
                    self.main_window.original_image.copy()
                )
                # 현재 뷰가 "텍스트 영역 표시"이면 즉시 업데이트
                if self.main_window.view_combo.currentIndex() == 2:
                    self.update_view()

            self.main_window.statusBar().showMessage(f"텍스트 감지 완료: {len(self.main_window.text_regions)}개 영역 발견")
        except Exception as e:
            self.handle_exception("OCR 결과 처리 중 예외 발생", e)


    def save_result(self):
        """현재 뷰의 결과 이미지 저장"""
        current_view_index = self.main_window.view_combo.currentIndex()
        image_to_save = None
        default_filename = "processed_image.png"

        # 각 뷰 인덱스에 해당하는 이미지가 있는지 확인하고 가져옴
        view_images = {
            0: self.main_window.original_image,
            1: self.main_window.processed_image,
            2: self.main_window.text_overlay_image,
            3: self.main_window.contour_image,
            4: self.main_window.wall_image,
            5: self.main_window.room_image,
        }
        view_filenames = {
            0: "original_image.png",
            1: "preprocessed_image.png",
            2: "text_overlay.png",
            3: "contours.png",
            4: "walls.png",
            5: "rooms.png",
        }

        if current_view_index in view_images and view_images[current_view_index] is not None:
            image_to_save = view_images[current_view_index]
            default_filename = view_filenames[current_view_index]
        else:
            QMessageBox.warning(self.main_window, "저장 오류", "현재 뷰에 저장할 이미지가 없습니다.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "이미지 저장", default_filename, "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)"
        )

        if file_path:
            try:
                # cv2.imwrite는 BGR 순서로 저장하므로, 필요시 변환
                # display_image에서 RGB로 변환했다면 다시 BGR로 바꿀 필요는 없을 수 있음
                # 하지만 안전하게 imwrite 사용
                cv2.imwrite(file_path, image_to_save)
                self.main_window.statusBar().showMessage(f"이미지 저장 완료: {file_path}")
            except Exception as e:
                self.handle_exception("이미지 저장 중 오류 발생", e)

    def save_workflow(self):
        """워크플로우 저장"""
        try:
            file_name, _ = QFileDialog.getSaveFileName(
                self.main_window, "워크플로우 저장", "workflow.json", "JSON Files (*.json)")
            if file_name:
                params = self.main_window.tab_manager.get_processing_parameters()
                import json
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=4, ensure_ascii=False)
                self.main_window.statusBar().showMessage(f"워크플로우 저장 완료: {file_name}")
        except Exception as e:
            self.handle_exception("워크플로우 저장 중 오류 발생", e)

    def load_workflow(self):
        """워크플로우 로드"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self.main_window, "워크플로우 로드", "", "JSON Files (*.json)")
            if file_name:
                import json
                with open(file_name, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                # 로드된 파라미터를 UI에 설정
                self.main_window.tab_manager.set_parameters(params)
                # DPI 설정도 반영되었는지 확인 (set_parameters에 포함됨)
                # 만약 즉시 DPI를 적용해야 한다면 preprocessor 호출 필요 (하지만 load_pdf 시점에 적용되므로 불필요)
                # if 'render_dpi' in params:
                #     self.main_window.preprocessor.set_resolution(params['render_dpi'])
                self.main_window.statusBar().showMessage(f"워크플로우 로드 완료: {file_name}")
        except Exception as e:
            self.handle_exception("워크플로우 로드 중 오류 발생", e)


    def export_to_dxf(self):
        """추출된 최종 외곽선을 DXF 파일로 내보내기"""
        # MainWindow 객체에 최종 외곽선 정보가 저장되어 있다고 가정 (예: self.main_window.final_outline_contour)
        if not hasattr(self.main_window, 'final_outline_contour') or self.main_window.final_outline_contour is None or len(self.main_window.final_outline_contour) == 0:
            QMessageBox.warning(self.main_window, "내보내기 오류", "추출된 최종 외곽선이 없습니다. 먼저 이미지를 처리하세요.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window, "DXF 내보내기", "outline.dxf", "DXF Files (*.dxf)")

        if file_path:
            try:
                # ContourExtractor의 메서드를 사용하여 저장
                success = self.main_window.extractor.save_contours_to_dxf(
                    self.main_window.final_outline_contour, file_path
                )
                if success:
                    self.main_window.statusBar().showMessage(f"DXF 내보내기 완료: {file_path}")
                else:
                    # save_contours_to_dxf 내부에서 오류 메시지를 출력했을 수 있음
                    QMessageBox.critical(self.main_window, "오류", "DXF 파일 저장에 실패했습니다. 콘솔 로그를 확인하세요.")
            except Exception as e:
                self.handle_exception("DXF 내보내기 중 오류 발생", e)


    def process_contours_and_features(self, params):
        """
        외곽선 추출, 필터링 및 기타 피처(벽, 방) 추출 및 최적화를 통합적으로 처리.
        기능 1 (빈 공간 채우기)은 functional.py에서 수행.
        기능 2 (작은 공간 병합)는 GeometryOptimizer에서 수행.
        """
        if self.main_window.processed_image is None:
            self.main_window.statusBar().showMessage("전처리된 이미지가 없습니다. PDF를 로드하고 처리하세요.")
            return

        self.main_window.statusBar().showMessage("외곽선 및 피처 추출 중...")
        QApplication.processEvents()

        try:
            # 1. 건물 외곽선 추출 (기존 로직과 유사)
            #    ContourExtractor의 extract_contours와 find_building_outline 사용
            _, all_raw_contours, hierarchy = self.main_window.extractor.extract_contours(
                self.main_window.processed_image.copy(), # 이진 이미지 전달
                params
            )

            if not all_raw_contours:
                self.main_window.statusBar().showMessage("건물 외곽선 추출을 위한 기본 윤곽선을 찾지 못했습니다.")
                self.main_window.contour_image = self.main_window.processed_image.copy()
                self.main_window.final_outline_contour = None
                # 방 추출 및 관련 이미지 초기화
                self.main_window.room_image = None
                self.main_window.optimized_room_image = None
                self.main_window.final_corrected_room_contours = []
                self.update_view()
                return

            self.main_window.contour_image, self.main_window.final_outline_contour = \
                self.main_window.extractor.find_building_outline(
                    self.main_window.processed_image.copy(), # 시각화 배경용
                    all_raw_contours,
                    hierarchy,
                    params
                )

            if self.main_window.final_outline_contour is None:
                self.main_window.statusBar().showMessage("최종 건물 외곽선을 선택하지 못했습니다.")
                # 방 추출 및 관련 이미지 초기화
                self.main_window.room_image = None
                self.main_window.optimized_room_image = None
                self.main_window.final_corrected_room_contours = []
                self.update_view()
                return

            # (선택적) 외곽선 병합 후처리 (기존 로직)
            if params.get('merge_overlapping', False) and self.main_window.final_outline_contour is not None:
                # ... (self.main_window.extractor.merge_overlapping_contours 호출) ...
                pass # 이 부분은 기존 코드 유지

            # --- 2. 방 추출 (functional.py에서 기능 1 포함) ---
            # 관련 이미지들 초기화
            self.main_window.room_image = None
            self.main_window.optimized_room_image = None
            self.main_window.final_corrected_room_contours = []

            if params.get('extract_rooms', False):
                if self.main_window.final_outline_contour is not None:
                    self.main_window.statusBar().showMessage("방 추출 진행 중 (기능 1,2 롤백 모드)...")
                    QApplication.processEvents()

                    # functional.py의 롤백된 extract_rooms_from_roi 호출
                    # 파라미터는 대표님의 원본 시그니처에 맞춤
                    raw_room_display_img, raw_room_contours_list = \
                        self.main_window.extractor.extract_rooms_from_roi(
                            self.main_window.processed_image.copy(),
                            self.main_window.final_outline_contour,
                            min_area=params.get('min_room_area', 10000), # 대표님 기본값 사용 또는 UI 값
                            max_area=params.get('max_room_area', 100000),# 대표님 기본값 사용 또는 UI 값
                            wall_closing_kernel_size=params.get('room_wall_closing_kernel', 1), # 대표님 기본값
                            space_opening_kernel_size=params.get('room_space_opening_kernel', 1),# 대표님 기본값
                            dist_transform_threshold_ratio=params.get('room_dist_transform_ratio', 0.1),
                            roi_margin=params.get('room_roi_margin', 0), # 대표님 기본값
                            room_simplify_epsilon_factor=params.get('room_simplify_epsilon', 0.02), # 대표님 기본값
                            min_area_for_missed_rooms_ratio=params.get('min_area_for_missed_rooms_ratio', 0.5) # 대표님 파라미터
                        )
                    
                    self.main_window.room_image = raw_room_display_img # functional.py의 시각화 결과

                    if raw_room_contours_list:
                        self.main_window.statusBar().showMessage("방 형상 최적화(스냅핑) 진행 중...")
                        QApplication.processEvents()

                        # GeometryOptimizer 롤백된 버전 사용
                        optimizer_settings = {
                            'snap_threshold_pixels': params.get('geom_snap_threshold', 5),
                            'min_vertices_for_polygon': params.get('geom_min_vertices', 3),
                            'min_edge_length_pixels': params.get('geom_min_edge_length', 1)
                        }
                        current_optimizer = GeometryOptimizer(**optimizer_settings)

                        # optimize_room_geometries 호출 시 roi_offset은 필요하지만,
                        # functional.py 롤백 버전은 roi_offset을 반환하지 않음.
                        # 시각화를 위해 (0,0) 또는 실제 ROI 오프셋을 어떻게든 구해야 함.
                        # 임시로 (0,0) 사용. 더 정확하려면 functional.py에서 받아와야 함.
                        # 또는, functional.py가 반환하는 raw_room_display_img의 원본 이미지(image)가
                        # 전체 페이지 이미지라면, raw_room_contours_list도 전체 좌표이므로 offset (0,0)이 맞음.
                        # extract_rooms_from_roi 내부에서 contour_orig_coords = ... + np.array([x_roi, y_roi]) 로직 확인 필요.
                        # 대표님 제공 코드에 해당 로직이 있으므로, raw_room_contours_list는 전체 이미지 좌표계임. 따라서 offset (0,0) 사용.
                        
                        optimized_final_contours, optimized_visualization_img = \
                            current_optimizer.optimize_room_geometries(
                                raw_room_contours_list, # 1차 추출된 방 목록
                                original_image_for_visualization=self.main_window.original_image.copy(),
                                roi_offset=(0,0) # raw_room_contours_list가 이미 전체 좌표이므로 (0,0)
                            )

                        self.main_window.final_corrected_room_contours = optimized_final_contours
                        self.main_window.optimized_room_image = optimized_visualization_img
                        
                        if optimized_visualization_img is not None:
                            self.main_window.statusBar().showMessage(f"방 형상 최적화(스냅핑) 완료. {len(optimized_final_contours)}개 방.")
                        else:
                             self.main_window.statusBar().showMessage(f"방 형상 최적화(스냅핑) 완료 (시각화 없음). {len(optimized_final_contours)}개 방.")
                    else: # raw_room_contours_incl_filled가 비어있는 경우
                        self.main_window.statusBar().showMessage("방 추출 결과, 유효한 방 윤곽선이 없습니다.")
                        # 관련 이미지 초기화는 이미 위에서 수행됨
                else: # self.main_window.final_outline_contour is None
                    self.main_window.statusBar().showMessage("방 추출을 위한 건물 외곽선이 선택되지 않았습니다.")
            else: # params.get('extract_rooms', False) is False
                self.main_window.statusBar().showMessage("방 추출 기능이 비활성화되어 있습니다.")
                # 관련 이미지 초기화는 이미 위에서 수행됨

            # --- 3. 벽 추출 (선택적, 기존 로직) ---
            self.main_window.wall_image = None # 초기화
            if params.get('extract_walls', False):
                self.main_window.statusBar().showMessage("벽 추출 중...")
                QApplication.processEvents()
                # extract_walls는 시각화 이미지와 선 리스트를 반환
                self.main_window.wall_image, _ = self.main_window.extractor.extract_walls(
                    self.main_window.processed_image.copy(), # 이진 이미지 전달
                    params.get('min_line_length', 50),
                    params.get('max_line_gap', 5)
                )

            self.main_window.statusBar().showMessage("모든 피처 처리 완료.")

        except Exception as e:
            self.handle_exception("외곽선 및 피처 처리 중 예외 발생", e)
            # 오류 발생 시 관련 이미지들 None으로 설정하여 UI에 문제 상황 반영
            self.main_window.contour_image = None
            self.main_window.final_outline_contour = None
            self.main_window.room_image = None
            self.main_window.optimized_room_image = None
            self.main_window.final_corrected_room_contours = []
            self.main_window.wall_image = None

        finally:
            self.update_view() # 모든 처리가 끝난 후 UI 뷰 업데이트
            self.main_window.save_btn.setEnabled(True) # 처리 후 저장 버튼 활성화
            self.main_window.undo_btn.setEnabled(len(self.main_window.processing_history) > 0)


    def prev_page(self):
        """이전 페이지 이동"""
        if self.main_window.pdf_path and self.main_window.current_page > 0:
            self.main_window.current_page -= 1
            self.load_current_page()
            self.update_page_navigation()
        else:
            self.main_window.statusBar().showMessage("첫 페이지입니다.")


    def next_page(self):
        """다음 페이지 이동"""
        if self.main_window.pdf_path and self.main_window.total_pages > 0 and self.main_window.current_page < self.main_window.total_pages - 1:
            self.main_window.current_page += 1
            self.load_current_page()
            self.update_page_navigation()
        else:
            self.main_window.statusBar().showMessage("마지막 페이지입니다.")


    def load_current_page(self):
        """현재 페이지 로드 및 관련 상태 초기화"""
        self.main_window.statusBar().showMessage(f"페이지 {self.main_window.current_page + 1} 로드 중...")
        QApplication.processEvents()
        try:
            # preprocessor.get_page_image는 설정된 DPI로 이미지를 렌더링
            img = self.main_window.preprocessor.get_page_image(self.main_window.current_page)
            if img is not None:
                # 새 페이지 로드 시 모든 처리 결과 초기화
                self.main_window.original_image = img
                self.main_window.processed_image = None
                self.main_window.text_overlay_image = None
                self.main_window.contour_image = None
                self.main_window.wall_image = None
                self.main_window.room_image = None
                self.main_window.final_outline_contour = None
                self.main_window.text_regions = []
                self.main_window.preprocessor.text_regions = [] # Preprocessor 내부 상태도 초기화
                self.main_window.processing_history = [] # 히스토리 초기화 (원본만 포함하도록 아래에서 추가)

                # 실행 취소를 위해 원본 이미지를 히스토리에 추가
                self.main_window.processing_history.append(self.main_window.original_image.copy())
                self.main_window.undo_btn.setEnabled(False) # 새 페이지 로드 시 실행 취소 비활성화

                self.update_view() # 원본 이미지 표시
                self.main_window.statusBar().showMessage(f"페이지 {self.main_window.current_page + 1}/{self.main_window.total_pages} 로드 완료")
            else:
                self.main_window.image_view.setText("현재 페이지 이미지를 로드할 수 없습니다.")
                self.main_window.statusBar().showMessage("페이지 로드 실패")
        except Exception as e:
            self.handle_exception(f"페이지 {self.main_window.current_page + 1} 로드 중 예외 발생", e)


    def update_page_navigation(self):
        """페이지 네비게이션 UI 업데이트"""
        if self.main_window.total_pages > 0 :
            self.main_window.page_label.setText(f"페이지: {self.main_window.current_page + 1}/{self.main_window.total_pages}")
            self.main_window.prev_btn.setEnabled(self.main_window.current_page > 0)
            self.main_window.next_btn.setEnabled(self.main_window.current_page < self.main_window.total_pages - 1)
        else:
            self.main_window.page_label.setText("페이지: -/-")
            self.main_window.prev_btn.setEnabled(False)
            self.main_window.next_btn.setEnabled(False)


    def update_view(self):
        """현재 선택된 뷰에 따라 이미지 표시 업데이트"""
        view_index = self.main_window.view_combo.currentIndex()
        img_to_display = None
        status_message = "뷰 업데이트됨"

        view_images = {
            0: self.main_window.original_image,
            1: self.main_window.processed_image,
            2: self.main_window.text_overlay_image,
            3: self.main_window.contour_image,
            4: self.main_window.wall_image,
            5: self.main_window.room_image,
            6: self.main_window.optimized_room_image
        }
        view_status_messages = {
            0: "원본 이미지 보기",
            1: "전처리된 이미지 보기",
            2: "텍스트 영역 표시 보기",
            3: "외곽선 추출 결과 보기",
            4: "추출된 벽 보기",
            5: "추출된 방 보기",
            6: "최적화된 방 보기"
        }

        if view_index in view_images:
            img_to_display = view_images[view_index]
            status_message = view_status_messages[view_index]
            if img_to_display is None:
                 status_message += " (결과 없음)"
                 # 결과 없을 시 원본 보여주기 (선택적)
                 # if view_index != 0: img_to_display = self.main_window.original_image

        if img_to_display is not None:
            self.display_image(img_to_display)
        else:
            # 표시할 이미지가 전혀 없는 경우 (초기 상태 등)
            self.main_window.image_view.setText("표시할 이미지가 없습니다. PDF를 로드하세요.")
            status_message = "표시할 이미지 없음"

        self.main_window.statusBar().showMessage(status_message)


    def zoom_image(self):
        """줌 슬라이더 값 변경 시 호출되어 이미지 뷰 업데이트"""
        zoom_percentage = self.main_window.zoom_slider.value()
        self.main_window.zoom_label.setText(f"{zoom_percentage}%")
        # 줌 값 변경 시 현재 보고 있는 뷰를 다시 그림
        self.update_view()


    def display_image(self, img_cv):
        """주어진 OpenCV 이미지를 QLabel에 표시 (줌 적용)"""
        if img_cv is None:
            self.main_window.image_view.setText("표시할 이미지가 없습니다.")
            return
        try:
            if not isinstance(img_cv, np.ndarray):
                self.main_window.image_view.setText("잘못된 이미지 데이터입니다.")
                return

            img_display = img_cv.copy()

            # 데이터 타입 및 채널 처리 (기존과 동일)
            if img_display.dtype != np.uint8:
                if np.max(img_display) <= 1.0 and img_display.dtype != np.bool_:
                    img_display = (img_display * 255).astype(np.uint8)
                else:
                    img_display = np.clip(img_display, 0, 255).astype(np.uint8)

            if len(img_display.shape) == 2:
                img_format = QImage.Format_Grayscale8 # 또는 RGB888로 변환
                bytes_per_line = img_display.strides[0]
                q_image = QImage(img_display.data, img_display.shape[1], img_display.shape[0], bytes_per_line, img_format)
                # QLabel은 Grayscale을 직접 잘 표시 못할 수 있으므로 RGB로 변환하는 것이 더 안전할 수 있음
                # img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
                # h, w = img_display_rgb.shape[:2]
                # bytes_per_line = img_display_rgb.strides[0]
                # q_image = QImage(img_display_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

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
                return

            if q_image.isNull():
                self.main_window.image_view.setText("QImage 변환 실패")
                return

            pixmap = QPixmap.fromImage(q_image)
            zoom_factor = self.main_window.zoom_slider.value() / 100.0

            # 원본 이미지 크기 기준으로 줌 적용
            original_width = pixmap.width()
            original_height = pixmap.height()
            target_width = int(original_width * zoom_factor)
            target_height = int(original_height * zoom_factor)

            # QLabel 크기에 맞춰 스케일링 (선택적, KeepAspectRatio 사용 시 불필요할 수 있음)
            # label_width = self.main_window.image_view.width()
            # label_height = self.main_window.image_view.height()
            # scaled_pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # 줌 비율에 따라 스케일링
            scaled_pixmap = pixmap.scaled(target_width, target_height,
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation) # SmoothTransformation 사용

            self.main_window.image_view.setPixmap(scaled_pixmap)

        except Exception as e:
            self.handle_exception("이미지 표시 중 예외 발생", e)


    def undo_processing(self):
        """실행 취소 (가장 최근 처리 상태로 복귀)"""
        if len(self.main_window.processing_history) > 1: # 원본 포함 최소 2개 이상 있어야 취소 가능
            # 현재 상태 (가장 마지막) 제거
            self.main_window.processing_history.pop()
            # 이전 상태 로드 (이것이 새로운 현재 상태가 됨)
            last_valid_state = self.main_window.processing_history[-1].copy()

            # 히스토리의 상태는 처리된 이미지일 수도 있고 원본 이미지일 수도 있음
            # 간단하게는 processed_image에 이전 상태를 넣고, 나머지는 None으로 초기화
            self.main_window.processed_image = last_valid_state
            self.main_window.contour_image = None
            self.main_window.wall_image = None
            self.main_window.room_image = None
            self.main_window.final_outline_contour = None
            # 텍스트 오버레이는 원본 기반이므로 유지하거나, 필요 시 재생성
            # 여기서는 일단 None으로 초기화
            self.main_window.text_overlay_image = None

            # 뷰 업데이트 (보통 processed_image나 original_image를 보여주게 됨)
            self.update_view()
            self.main_window.statusBar().showMessage("실행 취소 완료. 이전 처리 상태로 복원됨.")
        else:
            # 히스토리에 원본만 남은 경우
            self.main_window.statusBar().showMessage("더 이상 실행 취소할 내용이 없습니다.")

        # 실행 취소 버튼 활성화 여부 업데이트
        self.main_window.undo_btn.setEnabled(len(self.main_window.processing_history) > 1)


    def show_about(self):
        """정보 다이얼로그 표시"""
        QMessageBox.about(
            self.main_window,
            "정보",
            "건축 도면 처리 도구\n"
            "버전 1.1 (QThread 수정)\n\n"
            "이 프로그램은 건축 도면에서 외곽선을 추출하고 분석하는 도구입니다.\n"
            "PDF 도면을 로드하여 다양한 이미지 처리 기술을 적용하고,\n"
            "건물 외곽선, 벽, 방 등을 추출할 수 있습니다.\n\n"
            "© 2024-2025"
        )

    def handle_exception(self, message_prefix, exception_object):
        """ 공통 예외 처리기 """
        import traceback
        error_details = traceback.format_exc()
        full_message = f"{message_prefix}:\n{str(exception_object)}\n\nTraceback:\n{error_details}"
        print(full_message) # 콘솔에 전체 트레이스백 출력
        logging.error(full_message) # 로그 파일에도 기록
        QMessageBox.critical(self.main_window, "오류 발생", f"{message_prefix}:\n{str(exception_object)}")
        if hasattr(self.main_window, 'statusBar') and self.main_window.statusBar is not None:
            self.main_window.statusBar().showMessage(f"오류: {message_prefix}")