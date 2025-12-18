"""
深色主题样式
"""

def apply_dark_theme(widget):
    """应用深色主题到指定widget"""
    widget.setStyleSheet("""
        QMainWindow {
            background-color: #1E1E1E;
        }
        QWidget {
            color: #E0E0E0;
            background-color: #252526;
        }
        QPushButton {
            background-color: #3C3C3C;
            border: 1px solid #555555;
            border-radius: 4px;
            padding: 6px 12px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #4A4A4A;
        }
        QPushButton:pressed {
            background-color: #2D2D2D;
        }
        QSlider::groove:horizontal {
            border: 1px solid #3A3A3A;
            height: 8px;
            background: #2A2A2A;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #6699FF;
            border: 1px solid #5588EE;
            width: 16px;
            margin: -4px 0;
            border-radius: 8px;
        }
        QLabel {
            font-size: 12px;
        }
        QToolBar {
            background-color: #2D2D30;
            border: none;
        }
        QStatusBar {
            background-color: #2D2D30;
            border-top: 1px solid #3A3A3A;
        }
        QMenuBar {
            background-color: #2D2D30;
            border-bottom: 1px solid #3A3A3A;
        }
        QMenuBar::item {
            background-color: transparent;
            padding: 6px 12px;
        }
        QMenuBar::item:selected {
            background-color: #3A3A3A;
        }
        QMenu {
            background-color: #2D2D30;
            border: 1px solid #3A3A3A;
        }
        QMenu::item {
            padding: 6px 20px;
        }
        QMenu::item:selected {
            background-color: #3A3A3A;
        }
        QProgressBar {
            border: 1px solid #3A3A3A;
            border-radius: 4px;
            text-align: center;
            background-color: #2A2A2A;
        }
        QProgressBar::chunk {
            background-color: #6699FF;
            border-radius: 3px;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #3A3A3A;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QSpinBox, QDoubleSpinBox {
            background-color: #2A2A2A;
            border: 1px solid #3A3A3A;
            border-radius: 4px;
            padding: 4px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #6699FF;
        }
        QTabWidget::pane {
            border: 1px solid #3A3A3A;
            background-color: #252526;
        }
        QTabBar::tab {
            background-color: #2D2D30;
            border: 1px solid #3A3A3A;
            padding: 6px 12px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: #3A3A3A;
        }
        QListWidget {
            background-color: #2A2A2A;
            border: 1px solid #3A3A3A;
            border-radius: 4px;
        }
        QListWidget::item {
            padding: 6px;
        }
        QListWidget::item:selected {
            background-color: #3A3A3A;
        }
    """)