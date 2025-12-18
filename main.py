#!/usr/bin/env python3
"""
知音 - AI音频听觉功能集成软件
主程序入口
"""

import sys
from PySide6.QtWidgets import QApplication
from core.app import ZhiYinApp


def main():
    """主函数"""
    app = QApplication(sys.argv)
    zhiyin_app = ZhiYinApp()
    zhiyin_app.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()