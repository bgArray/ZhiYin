#!/usr/bin/env python3
"""
测试录音机功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from features.realtime_recorder.recorder_window import RecorderWindow


def main():
    """测试函数"""
    app = QApplication(sys.argv)
    
    # 创建录音机窗口
    recorder = RecorderWindow()
    recorder.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()