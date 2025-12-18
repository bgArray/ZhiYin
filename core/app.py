"""
知音应用程序核心类
"""

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Qt
from ui.main_window import MainWindow


class ZhiYinApp(MainWindow):
    """知音应用程序主类"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def closeEvent(self, event):
        """关闭事件处理"""
        # 确认退出
        reply = QMessageBox.question(
            self, 
            '确认退出',
            '确定要退出知音吗？',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 关闭所有子窗口
            for window in self.feature_windows.values():
                window.close()
            event.accept()
        else:
            event.ignore()