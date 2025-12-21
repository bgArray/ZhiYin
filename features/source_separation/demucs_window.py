"""
Demucs音源分离窗口
"""

import os
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QFrame,
    QProgressBar, QTextEdit, QGroupBox, QComboBox,
    QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices

from core.base_window import BaseWindow
from features.source_separation.demucs_processor import DemucsProcessorThread


class DemucsWindow(BaseWindow):
    """Demucs音源分离窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_window("音源分离 - Demucs", 1000, 700)
        self.center_on_screen()
        
        # 初始化变量
        self.current_file = None
        self.output_dir = None
        self.processing_thread = None
        
        # 创建UI
        self._create_ui()
    
    def _create_ui(self):
        """创建UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # 输入区域
        input_group = self._create_input_group()
        main_layout.addWidget(input_group)
        
        # 设置区域
        settings_group = self._create_settings_group()
        main_layout.addWidget(settings_group)
        
        # 进度区域
        progress_group = self._create_progress_group()
        main_layout.addWidget(progress_group)
        
        # 结果区域
        results_group = self._create_results_group()
        main_layout.addWidget(results_group)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
    
    def _create_input_group(self):
        """创建输入区域"""
        group = QGroupBox("输入文件")
        layout = QVBoxLayout(group)
        
        # 文件选择
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("音频文件:"))
        
        self.file_path_label = QLabel("未选择文件")
        self.file_path_label.setStyleSheet("color: #A0A0A0;")
        file_layout.addWidget(self.file_path_label, 1)
        
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self._browse_input_file)
        file_layout.addWidget(browse_button)
        
        layout.addLayout(file_layout)
        return group
    
    def _create_settings_group(self):
        """创建设置区域"""
        group = QGroupBox("分离设置")
        layout = QVBoxLayout(group)
        
        # 输出目录
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        
        self.output_dir_label = QLabel("默认: 与输入文件同目录")
        self.output_dir_label.setStyleSheet("color: #A0A0A0;")
        output_layout.addWidget(self.output_dir_label, 1)
        
        output_browse_button = QPushButton("浏览...")
        output_browse_button.clicked.connect(self._browse_output_dir)
        output_layout.addWidget(output_browse_button)
        
        layout.addLayout(output_layout)
        
        # 分离按钮
        self.separate_button = QPushButton("开始分离")
        self.separate_button.clicked.connect(self._start_separation)
        self.separate_button.setEnabled(False)
        layout.addWidget(self.separate_button)
        
        return group
    
    def _create_progress_group(self):
        """创建进度区域"""
        group = QGroupBox("处理进度")
        layout = QVBoxLayout(group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("等待开始...")
        layout.addWidget(self.status_label)
        
        return group
    
    def _create_results_group(self):
        """创建结果区域"""
        group = QGroupBox("分离结果")
        layout = QVBoxLayout(group)
        
        # 结果文本
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        layout.addWidget(self.results_text)
        
        # 按钮布局
        buttons_layout = QHBoxLayout()
        
        # 打开输出目录按钮
        self.open_output_button = QPushButton("打开输出目录")
        self.open_output_button.clicked.connect(self._open_output_directory)
        self.open_output_button.setEnabled(False)
        buttons_layout.addWidget(self.open_output_button)
        
        # 清空结果按钮
        clear_button = QPushButton("清空结果")
        clear_button.clicked.connect(self._clear_results)
        buttons_layout.addWidget(clear_button)
        
        layout.addLayout(buttons_layout)
        
        return group
    
    def _browse_input_file(self):
        """浏览输入文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a)"
        )
        
        if not file_path:
            return
        
        self.current_file = file_path
        self.file_path_label.setText(file_path)
        self.separate_button.setEnabled(True)
        
        # 设置默认输出目录
        if not self.output_dir:
            default_output_dir = os.path.dirname(file_path)
            self.output_dir = default_output_dir
            self.output_dir_label.setText(default_output_dir)
        
        self.statusBar().showMessage(f"已选择文件: {file_path}")
    
    def _browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            self.output_dir or ""
        )
        
        if not dir_path:
            return
        
        self.output_dir = dir_path
        self.output_dir_label.setText(dir_path)
    
    def _start_separation(self):
        """开始音源分离"""
        if not self.current_file:
            QMessageBox.warning(self, "警告", "请先选择音频文件")
            return
        
        if not self.output_dir:
            QMessageBox.warning(self, "警告", "请选择输出目录")
            return
        
        # 禁用按钮
        self.separate_button.setEnabled(False)
        
        # 重置进度和状态
        self.progress_bar.setValue(0)
        self.status_label.setText("正在初始化...")
        self.results_text.clear()
        
        # 创建处理线程，使用默认模型
        self.processing_thread = DemucsProcessorThread(
            self.current_file,
            self.output_dir
        )
        
        # 连接信号
        self.processing_thread.processing_progress.connect(self._on_progress)
        self.processing_thread.processing_complete.connect(self._on_complete)
        self.processing_thread.processing_error.connect(self._on_error)
        
        # 开始处理
        self.processing_thread.start()
        self.statusBar().showMessage("正在进行音源分离...")
    
    def _on_progress(self, value, status):
        """处理进度更新"""
        self.progress_bar.setValue(value)
        self.status_label.setText(status)
    
    def _on_complete(self, output_dir, result_info):
        """处理完成"""
        # 更新进度
        self.progress_bar.setValue(100)
        self.status_label.setText("分离完成")
        
        # 显示结果
        self.results_text.append(f"分离完成！输出目录: {output_dir}")
        self.results_text.append(f"采样率: {result_info['sample_rate']} Hz")
        self.results_text.append(f"时长: {result_info['duration']:.2f} 秒")
        self.results_text.append("\n分离的音源:")
        
        for source_name, file_path in result_info['output_files'].items():
            if source_name != "mix":  # 跳过混合音轨
                self.results_text.append(f"  {source_name}: {os.path.basename(file_path)}")
        
        # 启用打开输出目录按钮
        self.open_output_button.setEnabled(True)
        
        # 重新启用分离按钮
        self.separate_button.setEnabled(True)
        
        self.statusBar().showMessage("音源分离完成")
    
    def _on_error(self, error_msg):
        """处理错误"""
        self.status_label.setText(f"错误: {error_msg}")
        self.results_text.append(f"错误: {error_msg}")
        
        # 重新启用分离按钮
        self.separate_button.setEnabled(True)
        
        self.statusBar().showMessage("音源分离失败")
        QMessageBox.critical(self, "错误", f"音源分离失败: {error_msg}")
    
    def _open_output_directory(self):
        """打开输出目录"""
        if not self.output_dir:
            return
        
        # 使用系统默认文件管理器打开目录
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir))
    
    def _clear_results(self):
        """清空结果"""
        self.results_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("等待开始...")
        self.open_output_button.setEnabled(False)