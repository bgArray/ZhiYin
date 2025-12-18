# 音频功能大差不差


import sys
import math
import wave
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QSizePolicy,
    QSplitter, QFrame, QToolBar, QStatusBar, QDoubleSpinBox, QSpinBox, QMessageBox
)
from PySide6.QtCore import (
    Qt, QRectF, QPointF, QSize, Signal, Slot, QTimer,
    QMargins, QPoint, QIODevice, QThread
)
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont,
    QLinearGradient, QPixmap, QPainterPath
)
from PySide6.QtMultimedia import QAudioSink, QAudioFormat
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

# 确保matplotlib能够显示中文
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


# 多标签分类模型
class LightweightMultiLabelCNNLSTM(torch.nn.Module):
    """与M7训练脚本保持一致的模型定义"""

    def __init__(self, input_dim, num_labels, max_length):
        super().__init__()
        self.max_length = max_length
        self.num_labels = num_labels

        self.conv1 = torch.nn.Conv1d(
            in_channels=input_dim, out_channels=32, kernel_size=5, padding=2
        )
        self.bn1 = torch.nn.BatchNorm1d(32)

        self.depthwise_conv = torch.nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            groups=32,
        )
        self.pointwise_conv = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=128,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.lstm_dropout = torch.nn.Dropout(0.3)

        self.shared_fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
        )
        self.label_classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(64, 1) for _ in range(num_labels)]
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.pointwise_conv(self.depthwise_conv(x))))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        shared_features = self.shared_fc(lstm_out)

        outputs = []
        for classifier in self.label_classifiers:
            outputs.append(classifier(shared_features))
        multi_label_output = torch.cat(outputs, dim=-1)
        return multi_label_output

# 后台音频处理线程
class AudioProcessorThread(QThread):
    """后台音频处理线程，避免主线程阻塞"""

    processing_complete = Signal(object, object, object, object, object, float)
    processing_error = Signal(str)

    def __init__(
        self,
        audio_path,
        model,
        device,
        max_length,
        target_dim,
    ):
        super().__init__()
        self.audio_path = audio_path
        self.model = model
        self.device = device
        self.max_length = max_length
        self.target_dim = target_dim

    def run(self):
        try:
            y, sr = librosa.load(self.audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            n_fft = 2048
            hop_length = 512
            n_mels = 128

            mel_spectrogram = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_sequence = mel_spectrogram_db.T

            if mel_sequence.shape[1] >= self.target_dim:
                mel_sequence = mel_sequence[:, : self.target_dim]
            else:
                pad_width = self.target_dim - mel_sequence.shape[1]
                mel_sequence = np.pad(
                    mel_sequence,
                    pad_width=((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=0.0,
                )

            total_frames = mel_sequence.shape[0]
            if total_frames == 0:
                raise ValueError("音频帧数为0，无法进行推理")

            probabilities = []
            self.model.eval()

            with torch.no_grad():
                for start in range(0, total_frames, self.max_length):
                    end = min(start + self.max_length, total_frames)
                    chunk = mel_sequence[start:end]
                    current_len = len(chunk)
                    if current_len < self.max_length:
                        pad_len = self.max_length - current_len
                        chunk = np.pad(
                            chunk,
                            pad_width=((0, pad_len), (0, 0)),
                            mode="constant",
                            constant_values=0.0,
                        )

                    chunk_tensor = (
                        torch.from_numpy(chunk.astype(np.float32))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    logits = self.model(chunk_tensor).squeeze(0).cpu().numpy()
                    probs = 1.0 / (1.0 + np.exp(-logits))
                    probabilities.append(probs[:current_len])

            prob_matrix = np.vstack(probabilities)

            frame_duration = hop_length / sr
            time_stamps = np.arange(total_frames) * frame_duration
            time_stamps = np.minimum(time_stamps, duration)

            self.processing_complete.emit(
                y,
                sr,
                mel_spectrogram_db,
                prob_matrix,
                time_stamps,
                duration,
            )
        except Exception as exc:
            self.processing_error.emit(str(exc))

# 音频波形绘制组件
class WaveformWidget(QWidget):
    # 自定义信号
    mouseMoved = Signal(int)  # 鼠标移动时发送时间位置
    zoomChanged = Signal(float)  # 缩放变化信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 200)
        self.setMouseTracking(True)

        # 音频数据
        self.audio_data = None
        self.sample_rate = 44100
        self.channels = 1
        self.total_samples = 0
        self.total_seconds = 0
        self.sampwidth = 2  # 默认2字节（16位）

        # 视图参数
        self.zoom_level = 100.0  # 像素/秒
        self.offset_x = 0  # 水平偏移
        self.drag_start_x = 0
        self.is_dragging = False

        # 样式参数
        self.waveform_color = QColor(66, 135, 245)
        self.background_color = QColor(24, 24, 24)
        self.grid_color = QColor(40, 40, 40)
        self.playhead_color = QColor(255, 70, 70)
        self.time_ruler_color = QColor(180, 180, 180)

        # 播放头位置（秒）
        self.playhead_position = 0

        # 音频输出设备
        self.audio_sink = None
        self.audio_stream = None
        self.audio_format = None
        self.current_playback_sample = 0
        self.playback_buffer_size = 4096  # 播放缓冲区大小

    def load_audio(self, file_path):
        """加载音频文件并解析波形数据"""
        try:
            with wave.open(file_path, 'rb') as wf:
                self.channels = wf.getnchannels()
                self.sample_rate = wf.getframerate()
                self.sampwidth = wf.getsampwidth()
                self.total_samples = wf.getnframes()
                self.total_seconds = self.total_samples / self.sample_rate

                # 读取音频数据
                frames = wf.readframes(self.total_samples)
                dtype = np.int16 if self.sampwidth == 2 else np.int8
                self.audio_data = np.frombuffer(frames, dtype=dtype)

                # 如果是立体声，取单声道
                if self.channels == 2:
                    self.audio_data = self.audio_data[::2]

                # 归一化到[-1, 1]
                self.audio_data = self.audio_data / np.iinfo(dtype).max

                # 配置音频格式和输出设备
                self._setup_audio_output()

                # 重绘
                self.update()
                return True
        except Exception as e:
            print(f"加载音频失败: {e}")
            return False

    def _setup_audio_output(self):
        """配置音频输出格式和设备"""
        # 创建音频格式
        self.audio_format = QAudioFormat()
        self.audio_format.setSampleRate(self.sample_rate)
        self.audio_format.setChannelCount(1)  # 单声道输出
        if self.sampwidth == 2:
            self.audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        else:
            self.audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int8)

        # 初始化音频输出设备
        self.audio_sink = QAudioSink(self.audio_format)

    def start_playback(self):
        """开始播放音频"""
        if self.audio_data is None or len(self.audio_data) == 0 or not self.audio_sink:
            return

        # 将播放头位置转换为样本索引
        self.current_playback_sample = int(self.playhead_position * self.sample_rate)

        # 打开音频输出设备
        self.audio_stream = self.audio_sink.start()

    def stop_playback(self):
        """停止播放音频"""
        if self.audio_sink:
            self.audio_sink.stop()
        self.audio_stream = None

    def pause_playback(self):
        """暂停播放音频"""
        if self.audio_sink:
            self.audio_sink.suspend()

    def resume_playback(self):
        """恢复播放音频"""
        if self.audio_sink:
            self.audio_sink.resume()

    def get_next_audio_buffer(self, buffer_size):
        """获取下一段音频数据缓冲区"""
        if self.audio_data is None or len(self.audio_data) == 0 or self.current_playback_sample >= len(self.audio_data):
            return None

        # 计算要读取的样本范围
        end_sample = min(self.current_playback_sample + buffer_size, len(self.audio_data))
        samples = self.audio_data[self.current_playback_sample:end_sample]

        # 将归一化的音频数据转换回原始格式
        if self.sampwidth == 2:
            samples = (samples * 32767).astype(np.int16)
        else:
            samples = (samples * 127).astype(np.int8)

        # 更新当前播放位置
        self.current_playback_sample = end_sample

        # 将numpy数组转换为字节数据
        return samples.tobytes()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), self.background_color)

        # 绘制网格
        self._draw_grid(painter)

        # 绘制时间标尺
        self._draw_time_ruler(painter)

        # 绘制波形
        if self.audio_data is not None:
            self._draw_waveform(painter)

        # 绘制播放头
        self._draw_playhead(painter)

    def _draw_grid(self, painter):
        pen = QPen(self.grid_color, 1)
        painter.setPen(pen)

        # 水平网格线
        height = self.height()
        mid_y = height // 2
        painter.drawLine(0, mid_y, self.width(), mid_y)

        # 垂直网格线（每秒）
        sec_width = self.zoom_level
        x = -self.offset_x % sec_width
        while x < self.width():
            painter.drawLine(int(x), 0, int(x), height)
            x += sec_width

    def _draw_time_ruler(self, painter):
        pen = QPen(self.time_ruler_color, 1)
        painter.setPen(pen)

        font = QFont("Arial", 8)
        painter.setFont(font)

        # 绘制时间标签
        sec_width = self.zoom_level
        start_sec = int(-self.offset_x / sec_width)
        x = -self.offset_x % sec_width

        for i in range(start_sec, start_sec + int(self.width() / sec_width) + 1):
            if x > 0 and x < self.width():
                time_text = f"{i // 60}:{i % 60:02d}"
                painter.drawText(int(x) + 5, 15, time_text)
            x += sec_width

    def _draw_waveform(self, painter):
        if self.audio_data is None:
            return

        # 创建渐变笔刷
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, self.waveform_color.lighter(120))
        gradient.setColorAt(0.5, self.waveform_color)
        gradient.setColorAt(1, self.waveform_color.darker(120))

        pen = QPen(QBrush(gradient), 1)
        painter.setPen(pen)

        # 计算可见区域的样本范围
        start_sec = -self.offset_x / self.zoom_level
        end_sec = start_sec + self.width() / self.zoom_level

        start_sample = max(0, int(start_sec * self.sample_rate))
        end_sample = min(len(self.audio_data), int(end_sec * self.sample_rate))

        if start_sample >= end_sample:
            return

        # 抽取样本以适应显示宽度
        samples = self.audio_data[start_sample:end_sample]
        num_pixels = self.width()
        samples_per_pixel = max(1, len(samples) // num_pixels)

        # 绘制波形
        height = self.height()
        mid_y = height // 2
        scale = mid_y * 0.8  # 波形高度比例

        for x in range(num_pixels):
            sample_start = x * samples_per_pixel
            sample_end = min((x + 1) * samples_per_pixel, len(samples))

            if sample_start >= len(samples):
                break

            # 计算该像素位置的最大/最小值
            pixel_samples = samples[sample_start:sample_end]
            if len(pixel_samples) == 0:
                continue

            max_val = np.max(pixel_samples)
            min_val = np.min(pixel_samples)

            # 绘制垂直线
            y1 = mid_y - max_val * scale
            y2 = mid_y - min_val * scale
            painter.drawLine(x, int(y1), x, int(y2))

    def _draw_playhead(self, painter):
        pen = QPen(self.playhead_color, 2)
        painter.setPen(pen)

        # 计算播放头位置
        playhead_x = self.playhead_position * self.zoom_level + self.offset_x

        if 0 <= playhead_x <= self.width():
            painter.drawLine(int(playhead_x), 0, int(playhead_x), self.height())

            # 绘制播放头三角形
            path = QPainterPath()
            triangle_size = 8
            path.moveTo(int(playhead_x), 0)
            path.lineTo(int(playhead_x) - triangle_size, triangle_size * 2)
            path.lineTo(int(playhead_x) + triangle_size, triangle_size * 2)
            path.closeSubpath()

            painter.fillPath(path, self.playhead_color)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_start_x = event.position().x() - self.offset_x
        elif event.button() == Qt.RightButton:
            # 右键点击设置播放头位置
            self.playhead_position = max(0, min(
                self.total_seconds,
                (event.position().x() - self.offset_x) / self.zoom_level
            ))
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            self.offset_x = event.position().x() - self.drag_start_x
            self.update()

        # 发送鼠标位置信号
        mouse_sec = max(0, min(
            self.total_seconds,
            (event.position().x() - self.offset_x) / self.zoom_level
        ))
        self.mouseMoved.emit(int(mouse_sec))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False

    def wheelEvent(self, event):
        # 滚轮缩放
        delta = event.angleDelta().y()
        mouse_x = event.position().x()

        # 计算缩放中心的时间位置
        center_sec = (mouse_x - self.offset_x) / self.zoom_level

        # 调整缩放级别
        if delta > 0:
            self.zoom_level = min(self.zoom_level * 1.2, 1000.0)  # 最大缩放
        else:
            self.zoom_level = max(self.zoom_level / 1.2, 10.0)  # 最小缩放

        # 调整偏移以保持缩放中心位置不变
        self.offset_x = mouse_x - center_sec * self.zoom_level
        self.zoomChanged.emit(self.zoom_level)
        self.update()

    def set_zoom(self, zoom):
        """设置缩放级别"""
        # 计算当前视图中心的时间位置
        center_x = self.width() // 2
        center_sec = (center_x - self.offset_x) / self.zoom_level
        
        # 设置新的缩放级别
        self.zoom_level = max(10.0, min(zoom, 1000.0))
        
        # 调整偏移以保持视图中心位置不变
        self.offset_x = center_x - center_sec * self.zoom_level
        
        self.zoomChanged.emit(self.zoom_level)
        self.update()

    def set_playhead(self, position):
        """设置播放头位置（秒）"""
        self.playhead_position = max(0, min(self.total_seconds, position))
        self.update()
    
    def _get_visible_time_range(self):
        """获取当前可见的时间范围（开始和结束时间）"""
        start_sec = -self.offset_x / self.zoom_level
        end_sec = start_sec + self.width() / self.zoom_level
        return start_sec, end_sec


# 主窗口
class AudioEditorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音频编辑工具 - WaveEditor")
        self.setMinimumSize(1000, 600)

        # 设置样式
        self._setup_styles()

        # 初始化变量
        self.current_file = None
        self.is_playing = False
        self.playback_timer = QTimer()
        self.playback_timer.setInterval(10)  # 100fps for better buffer management
        self.playback_timer.timeout.connect(self._update_playback)

        # 音频播放缓冲区大小（样本数）
        self.playback_buffer_size = 4096
        
        # 进度条相关变量
        self.progress_slider = None
        self.is_dragging_progress = False

        # 音频分类相关变量
        self.model = None
        self.label_names = []
        self.label_to_idx = {}
        self.no_label_idx = None
        self.max_length = 512
        self.target_dim = 74
        self.last_visual_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.5

        # 创建UI
        self._create_ui()

    def _setup_styles(self):
        """设置应用程序样式"""
        self.setStyleSheet("""
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
            QSplitter::handle {
                background-color: #3A3A3A;
            }
            QSplitter::handle:horizontal {
                width: 1px;
            }
            QSplitter::handle:vertical {
                height: 1px;
            }
        """)

    def _create_ui(self):
        """创建用户界面"""
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建工具栏
        self._create_toolbar()

        # 创建波形显示区域
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.zoomChanged.connect(self._on_zoom_changed)
        self.waveform_widget.mouseMoved.connect(self._on_mouse_moved)

        # 创建控制区域
        control_layout = QHBoxLayout()

        # 缩放滑块
        zoom_label = QLabel("缩放:")
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.waveform_widget.set_zoom)

        # 播放控制按钮
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self._toggle_playback)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self._stop_playback)

        # 进度条
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setValue(0)
        self.progress_slider.setToolTip("音频播放进度")
        
        # 连接进度条信号
        self.progress_slider.sliderPressed.connect(self._on_progress_slider_pressed)
        self.progress_slider.valueChanged.connect(self._on_progress_slider_changed)
        self.progress_slider.sliderReleased.connect(self._on_progress_slider_released)

        # 时间显示
        self.time_label = QLabel("00:00 / 00:00")

        # 添加到控制布局
        control_layout.addWidget(zoom_label)
        control_layout.addWidget(self.zoom_slider)
        control_layout.addStretch()
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addStretch()
        control_layout.addWidget(self.progress_slider)
        control_layout.addWidget(self.time_label)
        control_layout.setContentsMargins(10, 5, 10, 5)
        control_layout.setStretchFactor(self.progress_slider, 3)

        # 添加波形部件到主布局
        main_layout.addWidget(self.waveform_widget)
        main_layout.addLayout(control_layout)

        # 创建分类结果展示区域
        self._create_classification_display(main_layout)

        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请加载音频文件")

    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        # 打开文件按钮
        open_btn = QPushButton("打开音频")
        open_btn.clicked.connect(self._open_audio_file)
        toolbar.addWidget(open_btn)

        toolbar.addSeparator()

        # 保存按钮
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self._save_audio_file)
        toolbar.addWidget(save_btn)

        toolbar.addSeparator()

        # 模型加载按钮
        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self._load_model_files)
        toolbar.addWidget(self.load_model_btn)

        # 阈值调节
        toolbar.addSeparator()
        threshold_label = QLabel("阈值:")
        toolbar.addWidget(threshold_label)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 0.95)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self.threshold_spin.setFixedWidth(60)
        toolbar.addWidget(self.threshold_spin)

        toolbar.addSeparator()

        # 缩放控制
        zoom_in_btn = QPushButton("放大")
        zoom_in_btn.clicked.connect(lambda: self._adjust_zoom(1.2))
        toolbar.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("缩小")
        zoom_out_btn.clicked.connect(lambda: self._adjust_zoom(0.8))
        toolbar.addWidget(zoom_out_btn)

    def _save_audio_file(self):
        """保存音频文件（占位）"""
        if self.current_file:
            self.status_bar.showMessage("保存功能 - 开发中")
        else:
            self.status_bar.showMessage("请先加载音频文件")

    def _adjust_zoom(self, factor):
        """调整缩放级别"""
        current_zoom = self.zoom_slider.value()
        new_zoom = current_zoom * factor
        self.zoom_slider.setValue(int(new_zoom))

    def _toggle_playback(self):
        """切换播放/暂停"""
        if self.is_playing:
            self.playback_timer.stop()
            self.waveform_widget.pause_playback()
            self.play_btn.setText("播放")
        else:
            if self.waveform_widget.audio_data is not None:
                self.waveform_widget.start_playback()
                self.playback_timer.start()
                self.play_btn.setText("暂停")

        self.is_playing = not self.is_playing

    def _stop_playback(self):
        """停止播放"""
        self.playback_timer.stop()
        self.waveform_widget.stop_playback()
        self.is_playing = False
        self.play_btn.setText("播放")
        self.waveform_widget.set_playhead(0)

        # 更新时间显示
        total_sec = self.waveform_widget.total_seconds
        total_time = f"{int(total_sec) // 60}:{int(total_sec) % 60:02d}"
        self.time_label.setText(f"00:00 / {total_time}")

    def _update_playback(self):
        """更新播放位置"""
        # 检查是否到达末尾
        if self.waveform_widget.current_playback_sample >= len(self.waveform_widget.audio_data):
            self._stop_playback()
            return

        # 发送音频数据到输出设备
        if self.is_playing and self.waveform_widget.audio_stream:
            # 获取音频设备的当前缓冲区状态
            bytes_free = self.waveform_widget.audio_sink.bytesFree()

            # 计算可以写入的最大样本数
            # 根据样本宽度计算每个样本占用的字节数
            bytes_per_sample = self.waveform_widget.sampwidth
            max_samples = bytes_free // bytes_per_sample

            if max_samples > 0:
                # 发送不超过可用缓冲区的样本数
                # 同时保持合理的缓冲区大小
                buffer_size = min(max_samples, int(0.1 * self.waveform_widget.sample_rate))
                audio_buffer = self.waveform_widget.get_next_audio_buffer(buffer_size)
                if audio_buffer:
                    self.waveform_widget.audio_stream.write(audio_buffer)

        # 根据实际播放的样本数更新播放头位置
        new_pos = self.waveform_widget.current_playback_sample / self.waveform_widget.sample_rate
        self.waveform_widget.set_playhead(new_pos)

        # 更新时间显示
        current_min = int(new_pos) // 60
        current_sec = int(new_pos) % 60
        total_sec = self.waveform_widget.total_seconds
        total_min = int(total_sec) // 60
        total_sec_display = int(total_sec) % 60

        time_text = f"{current_min}:{current_sec:02d} / {total_min}:{total_sec_display:02d}"
        self.time_label.setText(time_text)
        
        # 更新进度条位置
        if not self.is_dragging_progress and self.progress_slider is not None:
            # 计算进度条值（0-1000）
            progress_value = int((new_pos / total_sec) * 1000) if total_sec > 0 else 0
            self.progress_slider.blockSignals(True)
            self.progress_slider.setValue(progress_value)
            self.progress_slider.blockSignals(False)

    @Slot(float)
    def _on_zoom_changed(self, zoom_level):
        """缩放变化响应"""
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(zoom_level))
        self.zoom_slider.blockSignals(False)

    @Slot(int)
    def _on_mouse_moved(self, position_sec):
        """鼠标移动响应"""
        min_val = position_sec // 60
        sec_val = position_sec % 60
        self.status_bar.showMessage(
            f"位置: {min_val}:{sec_val:02d} 秒 | 缩放: {int(self.waveform_widget.zoom_level)} 像素/秒")
    
    def _on_progress_slider_pressed(self):
        """进度条按下事件"""
        self.is_dragging_progress = True
        if self.is_playing:
            # 暂停播放
            self.playback_timer.stop()
            self.waveform_widget.pause_playback()
    
    def _on_progress_slider_changed(self, value):
        """进度条值改变事件"""
        if self.is_dragging_progress:
            # 根据滑块位置计算新的播放时间
            if self.waveform_widget.total_seconds > 0:
                new_pos = (value / 1000.0) * self.waveform_widget.total_seconds
                self.waveform_widget.set_playhead(new_pos)
                
                # 更新时间显示
                current_min = int(new_pos) // 60
                current_sec = int(new_pos) % 60
                total_sec = self.waveform_widget.total_seconds
                total_min = int(total_sec) // 60
                total_sec_display = int(total_sec) % 60
                
                time_text = f"{current_min}:{current_sec:02d} / {total_min}:{total_sec_display:02d}"
                self.time_label.setText(time_text)
    
    def _on_progress_slider_released(self):
        """进度条释放事件"""
        self.is_dragging_progress = False
        if self.is_playing:
            # 更新当前播放位置并恢复播放
            new_pos = self.waveform_widget.playhead_position
            self.waveform_widget.current_playback_sample = int(new_pos * self.waveform_widget.sample_rate)
            self.waveform_widget.resume_playback()
            self.playback_timer.start()

    def _load_model_files(self):
        """加载模型和标签文件"""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import os
        import json
        import numpy as np
        
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型权重",
            "",
            "PyTorch模型 (*.pth *.pt);;所有文件 (*)",
        )
        if not model_path:
            return

        label_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标签映射文件",
            "",
            "JSON文件 (*.json);;Numpy文件 (*.npy);;所有文件 (*)",
        )
        if not label_path:
            return

        try:
            label_map = self._load_label_mapping(label_path)
            if not label_map:
                raise ValueError("标签映射文件为空或格式错误")

            self.label_to_idx = {str(k): int(v) for k, v in label_map.items()}
            idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
            self.label_names = [idx_to_label[idx] for idx in sorted(idx_to_label)]
            self.no_label_idx = self.label_to_idx.get("-1")

            num_labels = len(self.label_names)
            if num_labels == 0:
                raise ValueError("未能解析任何标签")

            self.model = LightweightMultiLabelCNNLSTM(
                input_dim=self.target_dim,
                num_labels=num_labels,
                max_length=self.max_length,
            )
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.status_bar.showMessage(
                f"模型加载成功，标签：{', '.join(self.label_names)}"
            )
            
            # 如果已经加载了音频文件，自动开始分类
            if self.current_file:
                self._classify_audio()
                
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", f"模型或标签加载错误：{exc}")
            self.status_bar.showMessage("模型加载失败")

    def _load_label_mapping(self, path):
        """加载标签映射文件"""
        import json
        import numpy as np
        import os
        
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("JSON标签文件必须是字典格式")
                return data
        if ext == ".npy":
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
            elif isinstance(data, np.ndarray) and len(data) == 1:
                data = data[0].item()
            if not isinstance(data, dict):
                raise ValueError("Numpy标签文件必须包含字典对象")
            return data
        raise ValueError("暂不支持的标签文件格式")

    def _classify_audio(self):
        """对当前加载的音频进行分类"""
        if not self.current_file:
            self.status_bar.showMessage("请先加载音频文件")
            return

        if not self.model:
            self.status_bar.showMessage("请先加载模型")
            return

        self.status_bar.showMessage("正在处理音频...")
        self.load_model_btn.setEnabled(False)
        
        # 创建音频处理线程
        self.processor = AudioProcessorThread(
            audio_path=self.current_file,
            model=self.model,
            device=self.device,
            max_length=self.max_length,
            target_dim=self.target_dim,
        )
        self.processor.processing_complete.connect(self._on_processing_complete)
        self.processor.processing_error.connect(self._on_processing_error)
        self.processor.start()

    def _on_processing_complete(self, waveform, sample_rate, mel_spectrogram_db, prob_matrix, time_stamps, duration):
        """音频处理完成回调"""
        self.last_visual_data = (waveform, sample_rate, mel_spectrogram_db, prob_matrix, time_stamps, duration)
        self._visualize_classification_results(*self.last_visual_data)
        self.status_bar.showMessage("音频分类完成")
        self.load_model_btn.setEnabled(True)

    def _on_processing_error(self, error_msg):
        """音频处理错误回调"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "处理错误", f"音频处理失败：{error_msg}")
        self.status_bar.showMessage("音频分类失败")
        self.load_model_btn.setEnabled(True)

    def _on_threshold_changed(self, value):
        """阈值变化响应"""
        self.threshold = value
        if self.last_visual_data:
            self._visualize_classification_results(*self.last_visual_data)

    def _visualize_classification_results(self, waveform, sample_rate, mel_spectrogram_db, prob_matrix, time_stamps, duration):
        """可视化分类结果"""
        self.classification_ax.clear()
        
        # 确保深色背景样式
        self.classification_fig.patch.set_facecolor('#252526')
        self.classification_ax.set_facecolor('#252526')
        
        # 使用阈值获取二分类结果
        threshold = self.threshold
        binary_preds = (prob_matrix > threshold).astype(bool)

        # 设置颜色映射
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.label_names)))
        color_map = {idx: colors[idx] for idx in range(len(self.label_names))}

        # 绘制每个标签的概率曲线
        for idx, label in enumerate(self.label_names):
            if label == "-1":
                continue
            
            # 绘制概率曲线
            self.classification_ax.plot(
                time_stamps,
                prob_matrix[:, idx],
                label=label,
                color=color_map[idx],
                alpha=0.7,
                linewidth=1.5
            )
            
            # 绘制阈值线
            self.classification_ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, linewidth=1)

        # 设置图表属性
        self.classification_ax.set_title("音频分类结果 - 标签概率随时间变化", color='#E0E0E0')
        self.classification_ax.set_xlabel("时间 (秒)", color='#E0E0E0')
        self.classification_ax.set_ylabel("概率", color='#E0E0E0')
        self.classification_ax.set_xlim(0, duration)
        self.classification_ax.set_ylim(0, 1.0)
        
        # 设置轴和文本颜色
        self.classification_ax.tick_params(colors='#E0E0E0')
        self.classification_ax.spines['bottom'].set_color('#E0E0E0')
        self.classification_ax.spines['left'].set_color('#E0E0E0')
        self.classification_ax.spines['top'].set_color('#252526')
        self.classification_ax.spines['right'].set_color('#252526')
        
        # 调整时间轴样式，使其与波形显示区域一致
        self.classification_ax.xaxis.set_tick_params(labelsize=8, pad=5)
        
        # 添加网格线
        self.classification_ax.grid(True, alpha=0.3, color='#444444')
        
        # 清除旧的图例项
        for i in reversed(range(self.legend_content_layout.count())): 
            self.legend_content_layout.itemAt(i).widget().setParent(None)
        
        # 在右侧图例容器中创建图例项
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.label_names)))
        for idx, label in enumerate(self.label_names):
            if label == "-1":
                continue
                
            # 创建图例项容器
            legend_item = QWidget()
            legend_item_layout = QHBoxLayout(legend_item)
            legend_item_layout.setContentsMargins(5, 2, 5, 2)
            
            # 创建颜色标记
            color_label = QLabel("■")
            color_label.setStyleSheet(f"color: rgb({colors[idx][0]*255:.0f}, {colors[idx][1]*255:.0f}, {colors[idx][2]*255:.0f}); font-size: 12px;")
            legend_item_layout.addWidget(color_label)
            
            # 创建标签文本
            text_label = QLabel(label)
            text_label.setStyleSheet("color: #E0E0E0; font-size: 10px;")
            legend_item_layout.addWidget(text_label)
            
            # 添加到图例内容布局
            self.legend_content_layout.addWidget(legend_item)
        
        # 添加弹性空间，使图例项顶部对齐
        self.legend_content_layout.addStretch()
        
        # 恢复原始布局
        self.classification_fig.tight_layout()
        self.classification_canvas.draw()

    def _open_audio_file(self):
        """打开音频文件对话框"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开音频文件",
            "",
            "Wave文件 (*.wav);;所有文件 (*.*)"
        )

        if file_path:
            if self.waveform_widget.load_audio(file_path):
                self.current_file = file_path
                self.status_bar.showMessage(f"已加载: {file_path}")

                # 更新时间显示
                total_sec = self.waveform_widget.total_seconds
                total_time = f"{int(total_sec) // 60}:{int(total_sec) % 60:02d}"
                self.time_label.setText(f"00:00 / {total_time}")
                
                # 初始化进度条
                if self.progress_slider is not None:
                    self.progress_slider.blockSignals(True)
                    self.progress_slider.setValue(0)
                    self.progress_slider.blockSignals(False)
                    
                # 如果已经加载了模型，自动开始分类
                if self.model:
                    self._classify_audio()

    def _create_classification_display(self, parent_layout):
        """创建分类结果展示区域"""
        # 创建分类显示框架
        self.classification_frame = QFrame()
        # 使用水平布局，将图表和图例放在同一行
        self.classification_layout = QHBoxLayout(self.classification_frame)
        self.classification_frame.setMinimumHeight(250)

        # 创建图表容器
        self.chart_container = QFrame()
        self.chart_layout = QVBoxLayout(self.chart_container)
        
        # 创建Matplotlib图形和画布，不再需要为图例预留空间
        self.classification_fig, self.classification_ax = plt.subplots(figsize=(10, 3))
        # 恢复原始布局，不需要为图例预留空间
        self.classification_fig.tight_layout()
        
        # 设置深色背景样式
        self.classification_fig.patch.set_facecolor('#252526')
        self.classification_ax.set_facecolor('#252526')
        
        # 设置轴和文本颜色
        self.classification_ax.tick_params(colors='#E0E0E0')
        self.classification_ax.spines['bottom'].set_color('#E0E0E0')
        self.classification_ax.spines['left'].set_color('#E0E0E0')
        self.classification_ax.spines['top'].set_color('#252526')
        self.classification_ax.spines['right'].set_color('#252526')
        
        self.classification_canvas = FigureCanvas(self.classification_fig)
        
        # 添加导航工具栏
        self.classification_toolbar = NavigationToolbar(self.classification_canvas, self.chart_container)
        # 设置工具栏样式为深色主题
        self.classification_toolbar.setStyleSheet("""
            QToolBar {
                background-color: #3C3C3C;
                border: 1px solid #555555;
            }
            QToolButton {
                background-color: #3C3C3C;
                color: #E0E0E0;
                border: 1px solid #555555;
            }
            QToolButton:hover {
                background-color: #4A4A4A;
            }
            QToolButton:pressed {
                background-color: #2D2D2D;
            }
            QLineEdit {
                background-color: #1E1E1E;
                color: #E0E0E0;
                border: 1px solid #555555;
            }
        """)
        
        # 将工具栏和画布添加到图表容器
        self.chart_layout.addWidget(self.classification_toolbar)
        self.chart_layout.addWidget(self.classification_canvas)
        
        # 创建图例容器
        self.legend_container = QFrame()
        self.legend_container.setFixedWidth(150)  # 设置图例容器宽度
        self.legend_layout = QVBoxLayout(self.legend_container)
        self.legend_container.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border: 1px solid #555555;
                border-radius: 5px;
            }
        """)
        
        # 创建图例标题
        self.legend_title = QLabel("标签图例")
        self.legend_title.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.legend_layout.addWidget(self.legend_title)
        
        # 创建图例内容区域
        self.legend_content = QWidget()
        self.legend_content_layout = QVBoxLayout(self.legend_content)
        self.legend_layout.addWidget(self.legend_content)
        
        # 添加默认提示文本
        default_label = QLabel("请加载模型查看标签")
        default_label.setStyleSheet("color: #888888; font-size: 10px; padding: 10px;")
        self.legend_content_layout.addWidget(default_label)
        self.legend_content_layout.addStretch()
        
        # 将图表容器和图例容器添加到水平布局
        self.classification_layout.addWidget(self.chart_container, 7)  # 图表占70%空间
        self.classification_layout.addWidget(self.legend_container, 3)  # 图例占30%空间

        # 添加到父布局
        parent_layout.addWidget(self.classification_frame)

        # 初始状态显示提示信息
        self.classification_ax.clear()
        self.classification_ax.text(0.5, 0.5, "请先加载模型和音频文件进行分类", 
                                  ha='center', va='center', fontsize=12, color='#E0E0E0')
        self.classification_ax.set_axis_off()
        # 确保深色主题一致性
        self.classification_fig.patch.set_facecolor('#252526')
        self.classification_ax.set_facecolor('#252526')
        # 恢复原始布局
        self.classification_fig.tight_layout()
        self.classification_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置高DPI支持
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    window = AudioEditorWindow()
    window.show()

    sys.exit(app.exec())
