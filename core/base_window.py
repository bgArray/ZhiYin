"""
基础窗口类
"""

from PySide6.QtWidgets import QMainWindow, QWidget
from PySide6.QtCore import Qt
from ui.styles.dark_theme import apply_dark_theme


class BaseWindow(QMainWindow):
    """所有功能窗口的基类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # 应用深色主题
        apply_dark_theme(self)
    
    def setup_window(self, title, width, height):
        """设置窗口基本属性"""
        self.setWindowTitle(title)
        self.resize(width, height)
    
    def center_on_screen(self):
        """将窗口居中显示在屏幕上"""
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        window = self.frameGeometry()
        center_point = screen.center()
        window.moveCenter(center_point)
        self.move(window.topLeft())