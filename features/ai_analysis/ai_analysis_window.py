"""
AI分析窗口
用于与大语言模型交互，分析声乐技术表现
"""

import json
import os
import requests
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QLineEdit, QSplitter,
    QGroupBox, QProgressBar, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QTextCursor

from core.base_window import BaseWindow
from config.config_manager import config_manager


class AIAnalysisThread(QThread):
    """AI分析线程"""
    
    # 定义信号
    response_received = Signal(str)  # AI响应
    error_occurred = Signal(str)  # 错误信息
    progress_updated = Signal(int, str)  # 进度更新
    
    def __init__(self, api_key, prompt, data=None, model_name="doubao-seed-1-6-lite-251015"):
        super().__init__()
        self.api_key = api_key
        self.prompt = prompt
        self.data = data
        self.model_name = model_name
        self.is_cancelled = False
    
    def run(self):
        """执行AI分析"""
        try:
            self.progress_updated.emit(10, "正在准备请求数据...")
            
            # 构建请求
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 构建请求体
            request_data = {
                "model": self.model_name,  # 使用指定的模型名称
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位专业的声乐技术分析师，能够分析歌手的演唱技巧、发声方法、呼吸控制等方面的表现。"
                                   "在客观分析的基础上可以多一些欣赏和鼓励。"
                                   "注意标签对应"
                                   "{说话: -1,真声: 0,混声: 1,假声: 2,气声: 3,咽音: 4,颤音: 5,滑音: 6}"
                    },
                    {
                        "role": "user",
                        "content": self.prompt
                    }
                ],
                "thinking": {
                    "type": "disabled"
                },
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            # 如果有数据，添加到请求中
            if self.data:
                self.progress_updated.emit(30, "正在处理声乐数据...")
                # 将数据转换为JSON字符串并添加到提示中
                data_str = json.dumps(self.data, ensure_ascii=False, indent=2)
                request_data["messages"][1]["content"] = f"{self.prompt}\n\n以下是声乐技术分析数据：\n```json\n{data_str}\n```"
            
            self.progress_updated.emit(50, "正在发送请求...")
            
            # 发送请求
            response = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                headers=headers,
                json=request_data,
                timeout=60
            )
            
            if self.is_cancelled:
                return
            
            self.progress_updated.emit(80, "正在处理响应...")
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    ai_response = result["choices"][0]["message"]["content"]
                    self.progress_updated.emit(100, "分析完成")
                    self.response_received.emit(ai_response)
                else:
                    self.error_occurred.emit("API响应格式错误")
            else:
                error_msg = f"API请求失败: {response.status_code} - {response.text}"
                self.error_occurred.emit(error_msg)
                
        except Exception as e:
            if self.is_cancelled:
                return
            error_msg = f"分析过程中发生错误: {str(e)}"
            self.error_occurred.emit(error_msg)
    
    def cancel(self):
        """取消分析"""
        self.is_cancelled = True


class AIAnalysisWindow(BaseWindow):
    """AI分析窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_window("AI声乐分析", 1000, 700)
        self.center_on_screen()
        
        # 初始化变量
        self.analysis_data = None
        self.audio_file_path = None
        self.api_key = self._load_api_key()
        self.analysis_thread = None
        
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
        
        # API密钥设置区域
        api_group = self.create_api_group()
        main_layout.addWidget(api_group)
        
        # 主分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 上部分：对话区域
        chat_widget = self.create_chat_widget()
        splitter.addWidget(chat_widget)
        
        # 下部分：输入区域
        input_widget = self.create_input_widget()
        splitter.addWidget(input_widget)
        
        # 设置分割比例
        splitter.setSizes([500, 200])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # 底部按钮区域
        buttons_widget = self.create_buttons_widget()
        main_layout.addWidget(buttons_widget)
    
    def create_api_group(self):
        """创建API密钥设置区域"""
        group = QGroupBox("API设置")
        layout = QVBoxLayout(group)
        
        # API密钥输入
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("豆包API密钥:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setText(self.api_key)
        self.api_key_edit.setPlaceholderText("请输入豆包API密钥")
        key_layout.addWidget(self.api_key_edit)
        
        # 保存API密钥按钮
        save_key_btn = QPushButton("保存密钥")
        save_key_btn.clicked.connect(self.save_api_key)
        key_layout.addWidget(save_key_btn)
        
        layout.addLayout(key_layout)
        
        # 模型名称输入
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型名称:"))
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setText("doubao-seed-1-6-lite-251015")
        self.model_name_edit.setPlaceholderText("请输入模型名称，如: doubao-seed-1-6-lite-251015")
        model_layout.addWidget(self.model_name_edit)
        
        layout.addLayout(model_layout)
        
        return group
    
    def create_chat_widget(self):
        """创建对话区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 对话历史显示
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 10))
        layout.addWidget(self.chat_display)
        
        return widget
    
    def create_input_widget(self):
        """创建输入区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 用户输入区域
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入:"))
        
        self.user_input = QTextEdit()
        self.user_input.setMaximumHeight(100)
        self.user_input.setPlaceholderText("请输入您的问题或分析要求...")
        input_layout.addWidget(self.user_input)
        
        layout.addLayout(input_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return widget
    
    def create_buttons_widget(self):
        """创建按钮区域"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # 发送按钮
        self.send_btn = QPushButton("发送")
        self.send_btn.clicked.connect(self.send_request)
        self.send_btn.setEnabled(False)  # 初始禁用
        
        # 快速分析按钮
        self.quick_analysis_btn = QPushButton("快速声乐分析")
        self.quick_analysis_btn.clicked.connect(self.quick_analysis)
        self.quick_analysis_btn.setEnabled(False)  # 初始禁用
        
        # 保存对话按钮
        self.save_chat_btn = QPushButton("保存对话")
        self.save_chat_btn.clicked.connect(self.save_chat)
        
        # 清空对话按钮
        self.clear_chat_btn = QPushButton("清空对话")
        self.clear_chat_btn.clicked.connect(self.clear_chat)
        
        layout.addStretch()
        layout.addWidget(self.send_btn)
        layout.addWidget(self.quick_analysis_btn)
        layout.addWidget(self.save_chat_btn)
        layout.addWidget(self.clear_chat_btn)
        
        return widget
    
    def _load_api_key(self):
        """加载API密钥"""
        # 从配置管理器加载API密钥
        return config_manager.get_api_key("doubao")
    
    def save_api_key(self):
        """保存API密钥"""
        self.api_key = self.api_key_edit.text().strip()
        if not self.api_key:
            QMessageBox.warning(self, "警告", "API密钥不能为空")
            return
        
        # 保存到配置管理器
        config_manager.set_api_key("doubao", self.api_key)
        config_manager.save_config()
        
        self.statusBar().showMessage("API密钥已保存")
        
        # 启用按钮
        self.send_btn.setEnabled(True)
        if self.analysis_data:
            self.quick_analysis_btn.setEnabled(True)
    
    def load_analysis_data(self, aligned_data, audio_file_path):
        """加载分析数据"""
        self.analysis_data = aligned_data
        self.audio_file_path = audio_file_path
        
        # 如果有API密钥，启用快速分析按钮
        if self.api_key:
            self.quick_analysis_btn.setEnabled(True)
        
        # 显示文件信息
        file_name = os.path.basename(audio_file_path)
        self.add_message("系统", f"已加载音频文件: {file_name}")
    
    def send_request(self):
        """发送用户请求"""
        user_text = self.user_input.toPlainText().strip()
        if not user_text:
            return
        
        if not self.api_key:
            QMessageBox.warning(self, "警告", "请先设置API密钥")
            return
        
        # 添加用户消息到对话历史
        self.add_message("用户", user_text)
        
        # 清空输入框
        self.user_input.clear()
        
        # 创建分析线程
        model_name = self.model_name_edit.text().strip()
        self.analysis_thread = AIAnalysisThread(self.api_key, user_text, self.analysis_data, model_name)
        
        # 连接信号
        self.analysis_thread.response_received.connect(self.on_response_received)
        self.analysis_thread.error_occurred.connect(self.on_error_occurred)
        self.analysis_thread.progress_updated.connect(self.on_progress_updated)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 禁用发送按钮
        self.send_btn.setEnabled(False)
        self.quick_analysis_btn.setEnabled(False)
        
        # 启动线程
        self.analysis_thread.start()
    
    def quick_analysis(self):
        """快速声乐分析"""
        if not self.analysis_data:
            QMessageBox.warning(self, "警告", "没有可分析的数据")
            return
        
        if not self.api_key:
            QMessageBox.warning(self, "警告", "请先设置API密钥")
            return
        
        # 构建快速分析提示
        prompt = """请分析以下音频的声乐技术表现，包括：
1. 歌手的发声技巧和特点
2. 呼吸控制情况
3. 音准和节奏感
4. 情感表达能力
5. 整体演唱水平评估
6. 改进建议

请根据提供的声乐技术标签数据，给出专业、详细的分析。"""
        
        # 添加用户消息到对话历史
        self.add_message("系统", "开始快速声乐分析...")
        
        # 创建分析线程
        model_name = self.model_name_edit.text().strip()
        self.analysis_thread = AIAnalysisThread(self.api_key, prompt, self.analysis_data, model_name)
        
        # 连接信号
        self.analysis_thread.response_received.connect(self.on_response_received)
        self.analysis_thread.error_occurred.connect(self.on_error_occurred)
        self.analysis_thread.progress_updated.connect(self.on_progress_updated)
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 禁用按钮
        self.send_btn.setEnabled(False)
        self.quick_analysis_btn.setEnabled(False)
        
        # 启动线程
        self.analysis_thread.start()
    
    def on_response_received(self, response):
        """处理AI响应"""
        # 添加AI响应到对话历史
        self.add_message("AI分析师", response)
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        
        # 启用按钮
        self.send_btn.setEnabled(True)
        if self.analysis_data:
            self.quick_analysis_btn.setEnabled(True)
        
        self.statusBar().showMessage("分析完成")
    
    def on_error_occurred(self, error_msg):
        """处理错误"""
        # 添加错误消息到对话历史
        self.add_message("系统", f"错误: {error_msg}")
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        
        # 启用按钮
        self.send_btn.setEnabled(True)
        if self.analysis_data:
            self.quick_analysis_btn.setEnabled(True)
        
        self.statusBar().showMessage("分析失败")
    
    def on_progress_updated(self, value, message):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def add_message(self, sender, message):
        """添加消息到对话历史"""
        # 获取当前时间
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 格式化消息
        formatted_message = f"[{timestamp}] {sender}:\n{message}\n\n"
        
        # 添加到对话历史
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(formatted_message)
        
        # 滚动到底部
        self.chat_display.ensureCursorVisible()
    
    def save_chat(self):
        """保存对话历史"""
        if not self.chat_display.toPlainText():
            QMessageBox.information(self, "提示", "没有可保存的对话内容")
            return
        
        # 获取保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存对话", "ai_analysis_chat.txt", "文本文件 (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.chat_display.toPlainText())
                self.statusBar().showMessage(f"对话已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def clear_chat(self):
        """清空对话历史"""
        reply = QMessageBox.question(
            self, "确认", "确定要清空对话历史吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.chat_display.clear()
            self.statusBar().showMessage("对话历史已清空")