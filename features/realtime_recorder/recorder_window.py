"""
实时录音机窗口
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QGroupBox, QGridLayout, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush
from PySide6.QtMultimedia import QAudioInput, QAudioFormat, QAudioDevice, QMediaDevices
import pyaudio

from core.base_window import BaseWindow
from .waveform_display import WaveformDisplay
from .pitch_display import PitchDisplay
from .audio_recorder import AudioRecorder
from .realtime_classifier import RealtimeAudioClassifier


class RecorderWindow(BaseWindow):
    """实时录音机窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实时录音机")
        self.setMinimumSize(1000, 800)
        
        # 音频录制器
        self.recorder = AudioRecorder()
        self.recording = False
        
        # 实时分类器
        self.classifier = RealtimeAudioClassifier()
        
        # 设置UI
        self._setup_ui()
        self._connect_signals()
        
        # 定时器用于更新显示
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        
    def _setup_ui(self):
        """设置UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 标题
        title = QLabel("实时录音机")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #E0E0E0; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 控制区域
        control_group = QGroupBox("录音控制")
        control_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        control_layout = QHBoxLayout(control_group)
        
        # 录音按钮
        self.record_button = QPushButton("开始录音")
        self.record_button.setFont(QFont("Arial", 12))
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        control_layout.addWidget(self.record_button)
        
        # 状态标签
        self.status_label = QLabel("状态: 未录音")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: #A0A0A0;")
        control_layout.addWidget(self.status_label)
        
        control_layout.addStretch()
        
        layout.addWidget(control_group)
        
        # 声音分类显示区域
        classification_group = QGroupBox("声音分类")
        classification_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        classification_layout = QVBoxLayout(classification_group)
        
        # 分类结果显示
        self.classification_label = QLabel("当前声音类型: 未检测")
        self.classification_label.setFont(QFont("Arial", 12))
        self.classification_label.setStyleSheet("color: #E0E0E0; padding: 10px; background-color: #333; border-radius: 5px;")
        classification_layout.addWidget(self.classification_label)
        
        layout.addWidget(classification_group)
        
        # 显示区域
        display_group = QGroupBox("实时显示")
        display_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        display_layout = QVBoxLayout(display_group)
        
        # 波形显示
        waveform_label = QLabel("音频波形")
        waveform_label.setFont(QFont("Arial", 12, QFont.Bold))
        waveform_label.setStyleSheet("color: #E0E0E0;")
        display_layout.addWidget(waveform_label)
        
        self.waveform_display = WaveformDisplay()
        self.waveform_display.setMinimumHeight(200)
        display_layout.addWidget(self.waveform_display)
        
        # 基频显示
        pitch_label = QLabel("基频 (F0) 分析")
        pitch_label.setFont(QFont("Arial", 12, QFont.Bold))
        pitch_label.setStyleSheet("color: #E0E0E0;")
        display_layout.addWidget(pitch_label)
        
        self.pitch_display = PitchDisplay()
        self.pitch_display.setMinimumHeight(200)
        display_layout.addWidget(self.pitch_display)
        
        layout.addWidget(display_group)
        
        # 设置区域
        settings_group = QGroupBox("录音设置")
        settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        settings_layout = QGridLayout(settings_group)
        
        # 采样率设置
        settings_layout.addWidget(QLabel("采样率:"), 0, 0)
        self.sample_rate_label = QLabel("16000 Hz")
        self.sample_rate_label.setStyleSheet("color: #A0A0A0;")
        settings_layout.addWidget(self.sample_rate_label, 0, 1)
        
        # 缓冲区大小设置
        settings_layout.addWidget(QLabel("缓冲区大小:"), 1, 0)
        self.buffer_size_label = QLabel("1024 样本")
        self.buffer_size_label.setStyleSheet("color: #A0A0A0;")
        settings_layout.addWidget(self.buffer_size_label, 1, 1)
        
        # FFT窗口大小
        settings_layout.addWidget(QLabel("FFT窗口大小:"), 2, 0)
        self.fft_size_label = QLabel("2048 样本")
        self.fft_size_label.setStyleSheet("color: #A0A0A0;")
        settings_layout.addWidget(self.fft_size_label, 2, 1)
        
        # 降噪选项
        self.noise_reduction_checkbox = QCheckBox("启用降噪")
        self.noise_reduction_checkbox.setFont(QFont("Arial", 10))
        self.noise_reduction_checkbox.setStyleSheet("color: #E0E0E0;")
        settings_layout.addWidget(self.noise_reduction_checkbox, 3, 0, 1, 2)
        
        # 降噪强度滑块
        settings_layout.addWidget(QLabel("降噪强度:"), 4, 0)
        self.noise_reduction_slider = QSlider(Qt.Horizontal)
        self.noise_reduction_slider.setMinimum(0)
        self.noise_reduction_slider.setMaximum(100)
        self.noise_reduction_slider.setValue(30)
        self.noise_reduction_slider.setEnabled(False)
        settings_layout.addWidget(self.noise_reduction_slider, 4, 1)
        
        self.noise_reduction_label = QLabel("30%")
        self.noise_reduction_label.setStyleSheet("color: #A0A0A0;")
        settings_layout.addWidget(self.noise_reduction_label, 4, 2)
        
        # 分类间隔设置
        settings_layout.addWidget(QLabel("分类间隔:"), 5, 0)
        self.classification_interval_slider = QSlider(Qt.Horizontal)
        self.classification_interval_slider.setMinimum(3)  # 0.3秒
        self.classification_interval_slider.setMaximum(20)  # 2.0秒
        self.classification_interval_slider.setValue(10)  # 1.0秒
        settings_layout.addWidget(self.classification_interval_slider, 5, 1)
        
        self.classification_interval_label = QLabel("1.0秒")
        self.classification_interval_label.setStyleSheet("color: #A0A0A0;")
        settings_layout.addWidget(self.classification_interval_label, 5, 2)
        
        layout.addWidget(settings_group)
        
    def _connect_signals(self):
        """连接信号"""
        self.record_button.clicked.connect(self.toggle_recording)
        self.noise_reduction_checkbox.toggled.connect(self.toggle_noise_reduction)
        self.noise_reduction_slider.valueChanged.connect(self.update_noise_reduction_strength)
        self.classification_interval_slider.valueChanged.connect(self.update_classification_interval)
        
    def toggle_recording(self):
        """切换录音状态"""
        if not self.recording:
            # 开始录音
            if self.recorder.start_recording():
                self.recording = True
                self.record_button.setText("停止录音")
                self.record_button.setStyleSheet("""
                    QPushButton {
                        background-color: #F44336;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #d32f2f;
                    }
                    QPushButton:pressed {
                        background-color: #b71c1c;
                    }
                """)
                self.status_label.setText("状态: 录音中...")
                self.status_label.setStyleSheet("color: #4CAF50;")
                
                # 启动更新定时器
                self.update_timer.start(100)  # 降低到10 FPS以提高性能
            else:
                self.status_label.setText("状态: 录音启动失败")
                self.status_label.setStyleSheet("color: #F44336;")
        else:
            # 停止录音
            self.recorder.stop_recording()
            self.recording = False
            self.record_button.setText("开始录音")
            self.record_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            self.status_label.setText("状态: 录音已停止")
            self.status_label.setStyleSheet("color: #A0A0A0;")
            
            # 停止更新定时器
            self.update_timer.stop()
            
    def toggle_noise_reduction(self, checked):
        """切换降噪状态"""
        self.noise_reduction_slider.setEnabled(checked)
        strength = self.noise_reduction_slider.value() / 100.0
        self.classifier.enable_noise_reduction(checked, strength)
        
        if checked:
            # 开始录音时捕获噪声 profile
            if self.recording:
                self.classifier.capture_noise_profile()
                
    def update_noise_reduction_strength(self, value):
        """更新降噪强度"""
        strength = value / 100.0
        self.noise_reduction_label.setText(f"{value}%")
        self.classifier.enable_noise_reduction(self.noise_reduction_checkbox.isChecked(), strength)
        
    def update_classification_interval(self, value):
        """更新分类间隔"""
        interval = value / 10.0  # 将滑块值转换为秒
        self.classification_interval_label.setText(f"{interval:.1f}秒")
        self.classifier.classification_interval = interval
            
    def _update_displays(self):
        """更新显示"""
        if self.recording:
            # 获取最新的音频数据
            audio_data = self.recorder.get_latest_audio()
            if audio_data is not None and len(audio_data) > 0:
                # 更新波形显示
                self.waveform_display.update_waveform(audio_data)
                
                # 更新基频显示
                pitch = self.recorder.get_latest_pitch()
                if pitch is not None:
                    self.pitch_display.update_pitch(pitch)
                
                # 添加音频数据到分类器
                self.classifier.add_audio_data(audio_data)
                
                # 获取分类结果
                result = self.classifier.classify()
                labels = result.get("labels", [])
                probabilities = result.get("probabilities", [])
                
                # 更新分类显示
                if labels:
                    # 格式化显示文本
                    text_parts = []
                    for i, (label, prob) in enumerate(zip(labels, probabilities)):
                        text_parts.append(f"{label} ({prob*100:.1f}%)")
                        if i >= 2:  # 最多显示3个标签
                            break
                    
                    self.classification_label.setText(f"当前声音类型: {', '.join(text_parts)}")
                else:
                    self.classification_label.setText("当前声音类型: 未检测到明确类型")
                    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.recording:
            self.recorder.stop_recording()
            self.update_timer.stop()
        event.accept()