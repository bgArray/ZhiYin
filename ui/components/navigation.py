"""
导航组件
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap


class FeatureCard(QPushButton):
    """功能卡片组件"""
    
    def __init__(self, feature_id, feature_info, parent=None):
        super().__init__(parent)
        self.feature_id = feature_id
        self.feature_info = feature_info
        
        self.setFixedSize(240, 160)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2D2D30;
                border: 1px solid #3A3A3A;
                border-radius: 8px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #3A3A3A;
                border: 1px solid #5588EE;
            }
            QPushButton:pressed {
                background-color: #252526;
            }
        """)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """设置卡片UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # 图标占位
        icon_label = QLabel()
        icon_label.setFixedSize(48, 48)
        icon_label.setStyleSheet("""
            QLabel {
                background-color: #3A3A3A;
                border-radius: 24px;
            }
        """)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setText("🎵")  # 使用emoji作为临时图标
        if self.feature_info.get("icon") == "multi_audio":
            icon_label.setText("🎶")  # 多音频处理器使用不同的图标
        icon_label.setFont(QFont("Arial", 24))
        layout.addWidget(icon_label, 0, Qt.AlignCenter)
        
        # 标题
        title_label = QLabel(self.feature_info["name"])
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(title_label, 0, Qt.AlignCenter)
        
        # 描述
        desc_label = QLabel(self.feature_info["description"])
        desc_label.setFont(QFont("Arial", 10))
        desc_label.setStyleSheet("color: #A0A0A0;")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label, 1, Qt.AlignCenter)


class NavigationWidget(QWidget):
    """导航组件"""
    
    feature_selected = Signal(str)  # 功能选择信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """设置导航UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 标题
        title_label = QLabel("知音 - AI音频工具集")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel("选择您需要的音频处理工具")
        subtitle_label.setFont(QFont("Arial", 12))
        subtitle_label.setStyleSheet("color: #A0A0A0; margin-bottom: 20px;")
        layout.addWidget(subtitle_label)
        
        # 功能卡片区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2A2A2A;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #4A4A4A;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #5A5A5A;
            }
        """)
        
        cards_widget = QWidget()
        cards_layout = QVBoxLayout(cards_widget)
        cards_layout.setSpacing(20)
        
        # 创建功能卡片
        from config.settings import FEATURE_MODULES
        for feature_id, feature_info in FEATURE_MODULES.items():
            card = FeatureCard(feature_id, feature_info)
            card.clicked.connect(lambda checked, fid=feature_id: self.feature_selected.emit(fid))
            cards_layout.addWidget(card, 0, Qt.AlignCenter)
        
        scroll_area.setWidget(cards_widget)
        layout.addWidget(scroll_area)
        
        # 添加弹性空间
        layout.addStretch()