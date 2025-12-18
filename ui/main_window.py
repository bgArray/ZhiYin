"""
主窗口
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QFrame, QSplitter
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap

from core.base_window import BaseWindow
from ui.components.navigation import NavigationWidget
from config.settings import APP_NAME, APP_VERSION, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT


class MainWindow(BaseWindow):
    """主窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.feature_windows = {}  # 存储已打开的功能窗口
        self._setup_ui()
        self._connect_signals()
        self.setup_window(APP_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        self.center_on_screen()
    
    def _setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧导航
        self.navigation = NavigationWidget()
        self.navigation.setMinimumWidth(300)
        self.navigation.setMaximumWidth(400)
        
        # 右侧欢迎区域
        self.welcome_widget = self._create_welcome_widget()
        
        splitter.addWidget(self.navigation)
        splitter.addWidget(self.welcome_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
    
    def _create_welcome_widget(self):
        """创建欢迎区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Logo图片
        logo_label = QLabel()
        import os
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logo_path = os.path.join(current_dir, "resources", "icons", "知音LOGO.png")
        logo_pixmap = QPixmap(logo_path)
        if not logo_pixmap.isNull():
            # 设置logo大小为200x200像素
            scaled_pixmap = logo_pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo_label)
        
        # 欢迎标题
        title = QLabel("欢迎使用知音")
        title.setFont(QFont("Arial", 32, QFont.Bold))
        title.setStyleSheet("color: #E0E0E0;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 版本信息
        version = QLabel(f"版本 {APP_VERSION}")
        version.setFont(QFont("Arial", 14))
        version.setStyleSheet("color: #A0A0A0;")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)
        
        # 说明文本
        description = QLabel(
            "知音是一款集成多种AI音频处理功能的软件，"
            "提供音频分类、音源分离等多种工具。"
            "请从左侧选择您需要的功能。"
        )
        description.setFont(QFont("Arial", 12))
        description.setStyleSheet("color: #A0A0A0;")
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        layout.addWidget(description)
        
        # 添加弹性空间
        layout.addStretch()
        
        return widget
    
    def _connect_signals(self):
        """连接信号"""
        self.navigation.feature_selected.connect(self.open_feature)
    
    def open_feature(self, feature_id):
        """打开功能窗口"""
        # 如果窗口已存在，则激活它
        if feature_id in self.feature_windows:
            self.feature_windows[feature_id].raise_()
            self.feature_windows[feature_id].activateWindow()
            return
        
        # 创建新的功能窗口
        if feature_id == "audio_classifier":
            from features.audio_classifier.classifier_window import AudioClassifierWindow
            window = AudioClassifierWindow(self)
        elif feature_id == "source_separation":
            from features.source_separation.demucs_window import DemucsWindow
            window = DemucsWindow(self)
        else:
            return
        
        # 存储窗口引用并显示
        self.feature_windows[feature_id] = window
        window.show()
        
        # 窗口关闭时移除引用
        window.destroyed.connect(lambda obj, fid=feature_id: self.feature_windows.pop(fid, None))