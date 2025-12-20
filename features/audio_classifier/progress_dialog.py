"""
进度条对话框窗口
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QPushButton, QWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont


class ProgressDialog(QDialog):
    """进度条对话框"""
    
    def __init__(self, parent=None, title="处理中", cancelable=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)  # 模态对话框
        self.setFixedSize(400, 150)
        self.cancelable = cancelable
        
        # 初始化UI
        self._init_ui()
        
        # 默认居中显示
        self._center_on_parent()
        
        # 取消标志
        self._cancelled = False
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题标签
        self.title_label = QLabel("正在处理...")
        self.title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(self.title_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("准备开始...")
        layout.addWidget(self.status_label)
        
        # 取消按钮（可选）
        if self.cancelable:
            button_layout = QHBoxLayout()
            button_layout.addStretch()
            
            self.cancel_button = QPushButton("取消")
            self.cancel_button.clicked.connect(self._on_cancel)
            button_layout.addWidget(self.cancel_button)
            
            layout.addLayout(button_layout)
    
    def _center_on_parent(self):
        """在父窗口中心显示"""
        if self.parent():
            parent_geometry = self.parent().geometry()
            x = parent_geometry.x() + (parent_geometry.width() - self.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - self.height()) // 2
            self.move(x, y)
    
    def _on_cancel(self):
        """取消操作"""
        self._cancelled = True
        self.close()
    
    def is_cancelled(self):
        """检查是否已取消"""
        return self._cancelled
    
    def update_progress(self, value, status=None):
        """更新进度"""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)
    
    def set_title(self, title):
        """设置标题"""
        self.setWindowTitle(title)
        self.title_label.setText(title)
    
    def close_after_delay(self, delay_ms=1000):
        """延迟关闭"""
        QTimer.singleShot(delay_ms, self.close)