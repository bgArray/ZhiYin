"""
多音频处理窗口
支持同时打开和处理多个音频文件
"""

import os
import json
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QFileDialog, QStatusBar,
    QMessageBox, QSplitter, QToolBar, QLabel
)
from PySide6.QtCore import Qt, Signal

from core.base_window import BaseWindow
from features.audio_classifier.classifier_window import AudioClassifierWindow


class AudioTabWidget(QWidget):
    """单个音频标签页内容"""
    
    # 定义信号
    analysis_completed = Signal(str, dict)  # 文件路径, 分析结果
    
    def __init__(self, file_path=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.audio_processor = None
        self.setup_ui()
        
        if file_path:
            self.load_audio_file(file_path)
    
    def setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建一个简单的容器来替代完整的AudioClassifierWindow
        # 这样可以避免在标签页中嵌入完整的窗口
        from features.audio_classifier.waveform_widget import WaveformWidget
        from features.audio_classifier.audio_processor import AudioProcessorThread
        from features.audio_classifier.lyrics_recognizer import LyricsRecognizerThread
        from features.audio_classifier.lyrics_tag_aligner import LyricsTagAligner
        from features.audio_classifier.progress_dialog import ProgressDialog
        from features.audio_classifier.lyrics_optimizer import LyricsOptimizerThread
        
        # 波形显示
        self.waveform_widget = WaveformWidget()
        layout.addWidget(self.waveform_widget)
        
        # 这里可以添加更多控件，但为了简化，我们只保留基本功能
        
        # 存储对齐结果
        self.aligned_data = None
    
    def load_audio_file(self, file_path):
        """加载音频文件"""
        self.file_path = file_path
        if self.waveform_widget:
            self.waveform_widget.load_audio(file_path)
    
    def get_aligned_data(self):
        """获取对齐后的数据"""
        return self.aligned_data
    
    def set_aligned_data(self, data):
        """设置对齐后的数据"""
        self.aligned_data = data


class MultiAudioProcessorWindow(BaseWindow):
    """多音频处理窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_window("多音频处理器", 1400, 900)
        self.center_on_screen()
        
        # 存储标签页
        self.audio_tabs = {}
        self.tab_counter = 0
        
        # 创建UI
        self.setup_ui()
        
        # 状态栏
        self.statusBar().showMessage("就绪")
    
    def setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 工具栏
        self.create_toolbar()
        
        # 使用说明
        info_label = QLabel("使用说明：1. 点击'加载对齐结果'加载已生成的JSON文件 2. 点击'AI声乐分析'进行专业分析")
        info_label.setStyleSheet("color: #A0A0A0; font-size: 12px; padding: 5px;")
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)
        
        # 标签页容器
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        
        main_layout.addWidget(self.tab_widget)
        
        # 底部按钮区域
        bottom_layout = QHBoxLayout()
        
        # 添加AI分析按钮
        self.ai_analysis_btn = QPushButton("AI声乐分析")
        self.ai_analysis_btn.clicked.connect(self.open_ai_analysis)
        self.ai_analysis_btn.setEnabled(False)  # 初始禁用
        
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.ai_analysis_btn)
        
        main_layout.addLayout(bottom_layout)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 打开音频文件按钮
        open_action = toolbar.addAction("打开音频文件")
        open_action.triggered.connect(self.open_audio_file)
        
        # 打开多个音频文件按钮
        open_multiple_action = toolbar.addAction("打开多个音频文件")
        open_multiple_action.triggered.connect(self.open_multiple_audio_files)
        
        toolbar.addSeparator()
        
        # 加载对齐结果按钮
        load_aligned_action = toolbar.addAction("加载对齐结果")
        load_aligned_action.triggered.connect(self.load_aligned_result)
        
        toolbar.addSeparator()
        
        # 关闭当前标签页
        close_tab_action = toolbar.addAction("关闭当前标签页")
        close_tab_action.triggered.connect(self.close_current_tab)
        
        # 关闭所有标签页
        close_all_action = toolbar.addAction("关闭所有标签页")
        close_all_action.triggered.connect(self.close_all_tabs)
    
    def open_audio_file(self):
        """打开单个音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", 
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*)"
        )
        
        if file_path:
            self.add_audio_tab(file_path)
    
    def open_multiple_audio_files(self):
        """打开多个音频文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个音频文件", "", 
            "音频文件 (*.wav *.mp3 *.flac *.ogg *.m4a);;所有文件 (*)"
        )
        
        for file_path in file_paths:
            self.add_audio_tab(file_path)
    
    def load_aligned_result(self):
        """加载对齐结果JSON文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择对齐结果文件", "", 
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                aligned_data = json.load(f)
            
            # 获取文件名作为标签标题
            file_name = os.path.basename(file_path)
            
            # 创建新的标签页
            self.tab_counter += 1
            tab_id = f"tab_{self.tab_counter}"
            
            # 创建标签页内容
            tab_widget = AudioTabWidget(file_path, self)
            
            # 设置对齐数据
            tab_widget.set_aligned_data(aligned_data)
            
            # 添加到标签页控件
            index = self.tab_widget.addTab(tab_widget, file_name)
            self.tab_widget.setCurrentIndex(index)
            
            # 存储标签页信息
            self.audio_tabs[tab_id] = {
                'widget': tab_widget,
                'file_path': file_path,
                'file_name': file_name
            }
            
            # 更新AI分析按钮状态
            self.update_ai_analysis_button()
            
            self.statusBar().showMessage(f"已加载对齐结果: {file_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载对齐结果失败: {str(e)}")
    
    def add_audio_tab(self, file_path):
        """添加音频标签页"""
        # 检查文件是否已经打开
        for tab_id, tab_data in self.audio_tabs.items():
            if tab_data['file_path'] == file_path:
                # 激活已存在的标签页
                self.tab_widget.setCurrentWidget(tab_data['widget'])
                return
        
        # 创建新的标签页
        self.tab_counter += 1
        tab_id = f"tab_{self.tab_counter}"
        
        # 获取文件名作为标签标题
        file_name = os.path.basename(file_path)
        
        # 创建标签页内容
        tab_widget = AudioTabWidget(file_path, self)
        
        # 连接信号
        tab_widget.analysis_completed.connect(self.on_analysis_completed)
        
        # 添加到标签页控件
        index = self.tab_widget.addTab(tab_widget, file_name)
        self.tab_widget.setCurrentIndex(index)
        
        # 存储标签页信息
        self.audio_tabs[tab_id] = {
            'widget': tab_widget,
            'file_path': file_path,
            'file_name': file_name
        }
        
        # 更新AI分析按钮状态
        self.update_ai_analysis_button()
        
        self.statusBar().showMessage(f"已加载: {file_name}")
    
    def close_tab(self, index):
        """关闭指定索引的标签页"""
        if index < 0:
            return
        
        widget = self.tab_widget.widget(index)
        if widget:
            # 找到对应的tab_id并删除
            tab_id_to_remove = None
            for tab_id, tab_data in self.audio_tabs.items():
                if tab_data['widget'] == widget:
                    tab_id_to_remove = tab_id
                    break
            
            if tab_id_to_remove:
                del self.audio_tabs[tab_id_to_remove]
            
            # 移除标签页
            self.tab_widget.removeTab(index)
            
            # 更新AI分析按钮状态
            self.update_ai_analysis_button()
    
    def close_current_tab(self):
        """关闭当前标签页"""
        current_index = self.tab_widget.currentIndex()
        if current_index >= 0:
            self.close_tab(current_index)
    
    def close_all_tabs(self):
        """关闭所有标签页"""
        while self.tab_widget.count() > 0:
            self.close_tab(0)
    
    def update_ai_analysis_button(self):
        """更新AI分析按钮状态"""
        # 如果有至少一个标签页，则启用AI分析按钮
        has_tabs = self.tab_widget.count() > 0
        self.ai_analysis_btn.setEnabled(has_tabs)
    
    def on_analysis_completed(self, file_path, analysis_data):
        """当分析完成时的处理"""
        # 这里可以处理分析完成的逻辑
        self.statusBar().showMessage(f"分析完成: {os.path.basename(file_path)}")
    
    def open_ai_analysis(self):
        """打开AI分析窗口"""
        # 获取当前活动的标签页
        current_widget = self.tab_widget.currentWidget()
        if not current_widget:
            return
        
        # 获取对齐后的数据
        aligned_data = current_widget.get_aligned_data()
        if not aligned_data:
            QMessageBox.warning(self, "警告", "请先完成歌词与标签的对齐操作")
            return
        
        # 打开AI分析窗口
        from features.ai_analysis.ai_analysis_window import AIAnalysisWindow
        ai_window = AIAnalysisWindow(self)
        ai_window.load_analysis_data(aligned_data, current_widget.file_path)
        ai_window.show()