"""
건축 도면 외곽선 추출 도구 - 메인 엔트리 포인트
"""
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from main_ui import MainWindow
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app_debug.log'
)
logger = logging.getLogger(__name__)

# 예외 처리 훅 설정
def exception_hook(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = exception_hook

def main():
    """
    애플리케이션의 메인 함수
    """
    # 고해상도 화면 설정
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    # 앱 인스턴스 생성 및 실행
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()