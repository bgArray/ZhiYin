"""
音频分类器窗口
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QSlider,
    QLabel,
    QSizePolicy,
    QSplitter,
    QFrame,
    QToolBar,
    QStatusBar,
    QDoubleSpinBox,
    QSpinBox,
    QMessageBox,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QTextEdit,
    QLineEdit,
)
from PySide6.QtCore import Qt, QTimer


def convert_numpy_types(obj):
    """
    递归转换numpy类型为Python原生类型，以便JSON序列化

    Args:
        obj: 包含numpy类型的对象

    Returns:
        转换后的对象，所有numpy类型都转换为Python原生类型
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


from core.base_window import BaseWindow
from features.audio_classifier.waveform_widget import WaveformWidget
from features.audio_classifier.model import AudioClassifierModel
from features.audio_classifier.audio_processor import AudioProcessorThread
from features.audio_classifier.lyrics_recognizer import LyricsRecognizerThread
from features.audio_classifier.lyrics_tag_aligner import LyricsTagAligner
from features.audio_classifier.progress_dialog import ProgressDialog

# 确保matplotlib能够显示中文
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class AudioClassifierWindow(BaseWindow):
    """音频分类器窗口"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_window("音频分类器", 1200, 800)
        self.center_on_screen()

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

        # 进度对话框
        self.progress_dialog = None

        # 音频分类相关变量
        self.model = None
        self.label_names = []
        self.label_to_idx = {}
        self.no_label_idx = None
        self.max_length = 512
        self.target_dim = 74
        self.last_visual_data = None
        self.device = None
        self.threshold = 0.5

        # 创建UI
        self._create_ui()

        # 加载模型
        self._load_model()

    def _create_ui(self):
        """创建UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 设置窗口最小尺寸
        self.setMinimumSize(1000, 700)

        # 工具栏
        self._create_toolbar()

        # 主分割器
        main_splitter = QSplitter(Qt.Vertical)

        # 上部分：波形显示
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.mouseMoved.connect(self._on_mouse_moved)
        self.waveform_widget.zoomChanged.connect(self._on_zoom_changed)
        self.waveform_widget.setMinimumHeight(200)  # 设置波形区域最小高度
        self.waveform_widget.setMaximumHeight(250)  # 设置波形区域最大高度

        # 下部分：分类结果和歌词识别
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)

        # 分类结果 - 增加高度比例，使其更突出
        results_widget = self._create_results_widget()
        results_widget.setMinimumHeight(300)  # 设置最小高度
        results_layout.addWidget(results_widget, 3)  # 拉伸因子为3

        # 歌词识别区域
        lyrics_group = self._create_lyrics_group()
        lyrics_group.setMaximumHeight(250)  # 限制最大高度
        results_layout.addWidget(lyrics_group, 1)  # 拉伸因子为1

        # 结果显示区域
        main_splitter.addWidget(self.waveform_widget)
        main_splitter.addWidget(results_container)
        main_splitter.setSizes([300, 500])  # 调整比例，给分类结果更多空间
        main_splitter.setStretchFactor(0, 1)  # 波形区域拉伸因子
        main_splitter.setStretchFactor(1, 2)  # 分类结果区域拉伸因子

        main_layout.addWidget(main_splitter)

        # 底部控制栏
        controls_widget = self._create_controls_widget()
        main_layout.addWidget(controls_widget)

        # 状态栏
        self.statusBar().showMessage("就绪")

    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # 打开文件按钮
        open_action = toolbar.addAction("打开音频文件")
        open_action.triggered.connect(self._open_audio_file)

        toolbar.addSeparator()

        # 播放控制按钮
        self.play_action = toolbar.addAction("播放")
        self.play_action.triggered.connect(self._toggle_playback)
        self.play_action.setEnabled(False)

        self.stop_action = toolbar.addAction("停止")
        self.stop_action.triggered.connect(self._stop_playback)
        self.stop_action.setEnabled(False)

        toolbar.addSeparator()

        # 模型状态显示
        self.model_status_action = toolbar.addAction("模型状态: 未加载")
        self.model_status_action.setEnabled(False)  # 禁用点击

        toolbar.addSeparator()

        # 加载模型按钮
        load_model_action = toolbar.addAction("加载模型")
        load_model_action.triggered.connect(self._load_model)

    def _create_model_progress_group(self):
        """创建模型加载进度区域"""
        group = QGroupBox("模型状态")
        layout = QVBoxLayout(group)

        # 模型状态显示
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("模型状态:"))
        self.model_status_label = QLabel("未加载")
        self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addWidget(self.model_status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # 模型加载进度条
        self.model_progress_bar = QProgressBar()
        self.model_progress_bar.setVisible(False)  # 初始隐藏
        layout.addWidget(self.model_progress_bar)

        # 模型信息显示
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("模型信息:"))
        self.model_info_label = QLabel("点击'加载模型'按钮开始")
        info_layout.addWidget(self.model_info_label)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        return group

    def _create_results_widget(self):
        """创建分类结果显示区域"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)  # 添加间距

        # 标题
        title_label = QLabel("分类结果")
        title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; margin-bottom: 10px;"
        )
        title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # 固定高度
        layout.addWidget(title_label)

        # 创建主容器 - 使用水平布局，将图表和图例放在同一行
        main_container = QWidget()
        main_layout = QHBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # 创建图表容器（占70%空间）
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)

        # 创建matplotlib图形
        self.figure = plt.figure(figsize=(10, 4))
        self.figure.patch.set_facecolor("#3C3C3C")  # 设置初始背景色为灰色
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)  # 设置画布最小高度
        self.canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )  # 允许扩展
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # 固定高度

        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(self.canvas, 1)  # 给画布更高的拉伸因子

        # 初始化一个空的图形，确保背景色正确
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#3C3C3C")
        ax.set_xlabel("时间 (秒)", color="#E0E0E0")
        ax.set_ylabel("概率", color="#E0E0E0")
        ax.set_title("音频分类结果", color="#E0E0E0")
        ax.grid(True, alpha=0.3, color="#444444")

        # 设置轴和文本颜色
        ax.tick_params(colors="#E0E0E0")
        for spine in ax.spines.values():
            spine.set_color("#E0E0E0")

        # 调整子图布局，确保底部标签有足够空间显示
        self.figure.subplots_adjust(
            bottom=0.15
        )  # 增加底部边距，确保"时间(秒)"标签不被遮挡

        self.canvas.draw()

        # 创建右侧容器（占30%空间），包含图例和标签选择
        right_container = QWidget()
        right_container.setMaximumWidth(250)  # 限制右侧容器的最大宽度
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)

        # 创建水平布局容器，用于放置图例和标签选择
        horizontal_container = QWidget()
        horizontal_layout = QHBoxLayout(horizontal_container)
        horizontal_layout.setContentsMargins(0, 0, 0, 0)
        horizontal_layout.setSpacing(10)

        # 创建图例部分
        legend_container = QWidget()
        legend_layout = QVBoxLayout(legend_container)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(5)

        # 创建图例标题
        legend_title = QLabel("图例")
        legend_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        legend_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # 固定高度
        legend_layout.addWidget(legend_title)

        self.legend_content = QWidget()
        self.legend_content.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )  # 允许扩展
        self.legend_content_layout = QVBoxLayout(self.legend_content)
        self.legend_content_layout.setContentsMargins(0, 0, 0, 0)
        self.legend_content_layout.setSpacing(2)
        legend_layout.addWidget(self.legend_content)

        horizontal_layout.addWidget(legend_container)

        # 创建标签选择部分
        label_container = QWidget()
        label_layout = QVBoxLayout(label_container)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(5)

        # 创建标签选择标题
        label_title = QLabel("显示标签")
        label_title.setStyleSheet("font-size: 12px; font-weight: bold;")
        label_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # 固定高度
        label_layout.addWidget(label_title)

        # 创建标签选择复选框容器
        self.label_checkboxes = {}
        self.label_checkboxes_widget = QWidget()
        self.label_checkboxes_layout = QVBoxLayout(self.label_checkboxes_widget)
        self.label_checkboxes_layout.setContentsMargins(0, 0, 0, 0)
        self.label_checkboxes_layout.setSpacing(5)

        # 初始化标签复选框（将在模型加载后更新）
        self._init_label_checkboxes()

        label_layout.addWidget(self.label_checkboxes_widget)
        horizontal_layout.addWidget(label_container)

        right_layout.addWidget(horizontal_container)

        # 将图表容器和右侧容器添加到水平布局
        main_layout.addWidget(chart_container, 7)  # 图表占70%空间
        main_layout.addWidget(right_container, 3)  # 右侧容器占30%空间

        layout.addWidget(main_container)

        return widget

    def _init_label_checkboxes(self):
        """初始化标签复选框"""
        # 清除现有复选框
        for checkbox in self.label_checkboxes.values():
            checkbox.setParent(None)
        self.label_checkboxes.clear()

        # 如果标签名称尚未加载，则使用默认值
        if not self.label_names:
            default_labels = ["真声", "混声", "假声", "气声", "咽音", "颤音", "滑音"]
            for label in default_labels:
                checkbox = QCheckBox(label)
                checkbox.setChecked(True)  # 默认全部选中
                checkbox.stateChanged.connect(self._update_visualization)
                self.label_checkboxes[label] = checkbox
                self.label_checkboxes_layout.addWidget(checkbox)
        else:
            # 使用实际标签名称
            for i, label_name in enumerate(self.label_names):
                if i == self.no_label_idx:
                    continue  # 跳过"无标签"

                checkbox = QCheckBox(label_name)
                checkbox.setChecked(True)  # 默认全部选中
                checkbox.stateChanged.connect(self._update_visualization)
                self.label_checkboxes[label_name] = checkbox
                self.label_checkboxes_layout.addWidget(checkbox)

    def _create_lyrics_group(self):
        """创建歌词识别区域"""
        group = QGroupBox("歌词识别")
        layout = QVBoxLayout(group)

        # 歌词识别状态
        self.lyrics_status_label = QLabel("等待音频处理完成后自动识别歌词...")
        self.lyrics_status_label.setStyleSheet("color: #666666; font-style: italic;")
        layout.addWidget(self.lyrics_status_label)

        # 歌词文本显示区域
        self.lyrics_text = QTextEdit()
        self.lyrics_text.setReadOnly(True)
        self.lyrics_text.setMaximumHeight(150)
        self.lyrics_text.setPlaceholderText("识别的歌词将显示在这里...")
        layout.addWidget(self.lyrics_text)

        return group

    def _create_controls_widget(self):
        """创建底部控制区域"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)

        # 进度滑块
        layout.addWidget(QLabel("进度:"))
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 1000)  # 使用0-1000范围，更精确
        self.progress_slider.setValue(0)
        self.progress_slider.sliderPressed.connect(self._on_progress_pressed)
        self.progress_slider.sliderReleased.connect(self._on_progress_released)
        self.progress_slider.valueChanged.connect(self._on_progress_changed)
        layout.addWidget(self.progress_slider)

        # 时间显示
        self.time_label = QLabel("00:00 / 00:00")
        layout.addWidget(self.time_label)

        # 缩放控制
        layout.addWidget(QLabel("缩放:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_slider_changed)
        layout.addWidget(self.zoom_slider)

        self.zoom_label = QLabel("100%")
        layout.addWidget(self.zoom_label)

        return widget

    def _load_model(self):
        """加载分类模型"""
        try:
            # 更新工具栏中的模型状态
            self.model_status_action.setText("模型状态: 加载中...")

            # 创建进度对话框
            self.progress_dialog = ProgressDialog(self, "加载模型", cancelable=False)
            self.progress_dialog.show()

            # 更新进度
            self.progress_dialog.update_progress(5, "正在初始化模型...")
            self.statusBar().showMessage("正在加载模型...")

            # 模拟初始化步骤
            import time

            time.sleep(0.2)  # 短暂延迟以显示初始化状态

            # 更新进度
            self.progress_dialog.update_progress(15, "正在检查设备支持...")

            # 更新进度
            self.progress_dialog.update_progress(25, "正在加载模型文件...")

            self.model = AudioClassifierModel()

            # 更新进度
            self.progress_dialog.update_progress(50, "正在加载标签映射...")

            self.label_names = self.model.label_names
            self.label_to_idx = self.model.label_to_idx
            self.no_label_idx = self.model.no_label_idx
            self.device = self.model.device

            # 更新进度
            self.progress_dialog.update_progress(75, "正在初始化UI组件...")

            # 模型加载后更新标签复选框
            self._init_label_checkboxes()

            # 更新进度
            self.progress_dialog.update_progress(90, "正在准备可视化组件...")

            # 初始化图表
            self._init_visualization()

            # 完成加载
            self.progress_dialog.update_progress(100, "模型加载完成")

            # 更新工具栏中的模型状态
            device_str = "GPU" if str(self.device) == "cuda" else "CPU"
            self.model_status_action.setText(
                f"模型状态: 已加载 ({device_str}, {len(self.label_names)}标签)"
            )

            # 延迟关闭进度对话框
            self.progress_dialog.close_after_delay(500)

            self.statusBar().showMessage("模型加载完成")
        except Exception as e:
            # 更新工具栏中的模型状态 - 错误状态
            self.model_status_action.setText("模型状态: 加载失败")

            if self.progress_dialog:
                self.progress_dialog.close()
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.statusBar().showMessage("模型加载失败")

    def _open_audio_file(self):
        """打开音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频文件 (*.wav *.mp3 *.flac *.ogg)"
        )

        if not file_path:
            return

        self._open_audio_file_path(file_path)

    def _open_audio_file_path(self, file_path):
        """通过文件路径直接打开音频文件"""
        self.current_file = file_path
        self.statusBar().showMessage(f"正在加载音频文件: {file_path}")

        # 加载音频
        if self.waveform_widget.load_audio(file_path):
            # 更新UI
            self.play_action.setEnabled(True)
            self.stop_action.setEnabled(True)

            # 更新进度条范围
            self.progress_slider.setRange(0, 1000)  # 使用0-1000范围，更精确
            self.progress_slider.setValue(0)

            # 更新时间显示
            self._update_time_display()

            # 开始处理音频
            self._process_audio()

            self.statusBar().showMessage(f"音频加载完成: {file_path}")
        else:
            QMessageBox.critical(self, "错误", "音频文件加载失败")
            self.statusBar().showMessage("音频文件加载失败")

    def get_aligned_data(self):
        """获取对齐后的数据"""
        if hasattr(self, "aligned_result") and self.aligned_result:
            return self.aligned_result
        return None

    def _process_audio(self):
        """处理音频进行分类"""
        if not self.model or not self.current_file:
            return

        # 创建音频处理进度对话框
        self.progress_dialog = ProgressDialog(self, "处理音频", cancelable=True)
        self.progress_dialog.show()

        self.progress_dialog.update_progress(10, "正在初始化音频处理...")
        self.statusBar().showMessage("正在处理音频...")

        # 创建处理线程
        self.processing_thread = AudioProcessorThread(
            self.current_file, self.model, self.max_length, self.target_dim
        )

        # 连接信号
        self.processing_thread.processing_complete.connect(self._on_processing_complete)
        self.processing_thread.processing_error.connect(self._on_processing_error)
        self.processing_thread.processing_progress.connect(self._on_processing_progress)

        # 开始处理
        self.progress_dialog.update_progress(20, "正在加载音频文件...")
        self.processing_thread.start()

    def _on_processing_complete(
        self, y, sr, mel_spectrogram_db, prob_matrix, time_stamps, duration
    ):
        """处理完成回调"""
        if self.progress_dialog:
            self.progress_dialog.update_progress(90, "正在更新可视化...")

        self.last_visual_data = {
            "y": y,
            "sr": sr,
            "mel_spectrogram_db": mel_spectrogram_db,
            "prob_matrix": prob_matrix,
            "time_stamps": time_stamps,
            "duration": duration,
        }

        # 更新可视化
        self._update_visualization()

        # 完成处理
        if self.progress_dialog:
            self.progress_dialog.update_progress(100, "音频处理完成")
            self.progress_dialog.close_after_delay(1000)

        # 自动开始歌词识别
        self._auto_recognize_lyrics()

        self.statusBar().showMessage("处理完成，正在自动识别歌词...")

    def _on_processing_progress(self, value, status):
        """处理进度更新"""
        if self.progress_dialog:
            self.progress_dialog.update_progress(value, status)

    def _on_processing_error(self, error_msg):
        """处理错误回调"""
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "处理错误", f"音频处理失败: {error_msg}")
        self.statusBar().showMessage("处理失败")

    def _auto_recognize_lyrics(self):
        """自动识别歌词"""
        if not self.current_file:
            return

        # 更新状态标签
        self.lyrics_status_label.setText("正在自动识别歌词...")
        self.lyrics_status_label.setStyleSheet("color: #0066CC; font-style: italic;")

        # 创建歌词识别线程
        self.lyrics_thread = LyricsRecognizerThread(
            self.current_file, preserve_pronunciation=False  # 默认不保留咬字信息
        )

        # 连接信号
        self.lyrics_thread.recognition_progress.connect(self._on_auto_lyrics_progress)
        self.lyrics_thread.recognition_complete.connect(self._on_auto_lyrics_complete)
        self.lyrics_thread.recognition_error.connect(self._on_auto_lyrics_error)

        # 开始识别
        self.lyrics_thread.start()

    def _on_auto_lyrics_progress(self, value, status):
        """自动歌词识别进度更新"""
        self.lyrics_status_label.setText(f"正在识别歌词... {value}%")

    def _on_auto_lyrics_complete(self, lyrics_data):
        """自动歌词识别完成回调"""
        # 保存歌词数据
        self.lyrics_data = lyrics_data

        # 更新状态标签
        self.lyrics_status_label.setText("歌词识别完成，正在转换为简体中文...")
        self.lyrics_status_label.setStyleSheet("color: #009900; font-style: italic;")

        # 显示歌词文本
        if "text" in lyrics_data:
            self.lyrics_text.setText(lyrics_data["text"])

        # 自动对齐歌词与标签
        self._auto_align_lyrics_with_tags()

        # 更新状态栏
        self.statusBar().showMessage("歌词识别与对齐完成")

    def _on_auto_lyrics_error(self, error_msg):
        """自动歌词识别错误"""
        self.lyrics_status_label.setText(f"歌词识别失败: {error_msg}")
        self.lyrics_status_label.setStyleSheet("color: #CC0000; font-style: italic;")

    def _auto_align_lyrics_with_tags(self):
        """自动对齐歌词与标签"""
        if not hasattr(self, "lyrics_data") or not self.lyrics_data:
            return

        if not hasattr(self, "last_visual_data") or not self.last_visual_data:
            return

        try:
            # 更新状态标签
            self.lyrics_status_label.setText("正在对齐歌词与标签...")
            self.lyrics_status_label.setStyleSheet(
                "color: #0066CC; font-style: italic;"
            )

            # 创建歌词标签对齐器
            aligner = LyricsTagAligner()

            # 执行对齐
            self.aligned_result = aligner.align_lyrics_with_tags(
                self.lyrics_data,
                self.last_visual_data["prob_matrix"],
                self.last_visual_data["time_stamps"],
                self.label_names,
            )

            # 自动保存对齐结果到JSON文件
            self._save_aligned_result_to_json()

            # 更新状态标签
            self.lyrics_status_label.setText("歌词识别与对齐完成，已自动保存JSON文件")
            self.lyrics_status_label.setStyleSheet(
                "color: #009900; font-style: italic;"
            )

        except Exception as e:
            print(f"歌词对齐失败: {e}")
            # 更新状态标签
            self.lyrics_status_label.setText(f"歌词对齐失败: {e}")
            self.lyrics_status_label.setStyleSheet(
                "color: #CC0000; font-style: italic;"
            )

    def _save_aligned_result_to_json(self):
        """自动保存对齐结果到JSON文件"""
        if not hasattr(self, "aligned_result") or not self.aligned_result:
            return

        try:
            # 获取音频文件路径和基本信息
            audio_file = self.current_file
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_dir = os.path.dirname(audio_file)

            # 创建输出文件路径
            output_file = os.path.join(output_dir, f"{base_name}_lyrics_alignment.json")

            # 准备保存的数据
            save_data = {
                "audio_file": audio_file,
                "alignment_result": convert_numpy_types(self.aligned_result),
                "label_names": self.label_names,
                "label_to_idx": self.label_to_idx,
            }

            # 保存到JSON文件
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            print(f"歌词对齐结果已保存到: {output_file}")

        except Exception as e:
            print(f"保存歌词对齐结果失败: {e}")

    def _recognize_lyrics(self):
        """识别歌词"""
        if not self.current_file:
            QMessageBox.warning(self, "警告", "请先加载音频文件")
            return

        # 获取用户选择的选项
        preserve_pronunciation = self.preserve_pronunciation_checkbox.isChecked()

        # 创建歌词识别线程
        self.lyrics_thread = LyricsRecognizerThread(
            self.current_file, preserve_pronunciation=preserve_pronunciation
        )

        # 创建歌词识别进度对话框
        self.progress_dialog = ProgressDialog(self, "识别歌词", cancelable=True)
        self.progress_dialog.show()

        # 连接信号
        self.lyrics_thread.recognition_progress.connect(self._on_lyrics_progress)
        self.lyrics_thread.recognition_complete.connect(self._on_lyrics_complete)
        self.lyrics_thread.recognition_error.connect(self._on_lyrics_error)

        # 重置进度和状态
        self.progress_dialog.update_progress(0, "正在识别歌词...")

        # 开始识别
        self.lyrics_thread.start()

        # 禁用识别按钮，启用取消按钮
        self.recognize_lyrics_btn.setEnabled(False)
        self.align_lyrics_btn.setEnabled(False)

    def _on_lyrics_progress(self, value, status):
        """歌词识别进度更新"""
        if self.progress_dialog:
            self.progress_dialog.update_progress(value, status)

    def _on_lyrics_complete(self, lyrics_data):
        """歌词识别完成回调"""
        if self.progress_dialog:
            self.progress_dialog.update_progress(100, "歌词识别完成")
            self.progress_dialog.close_after_delay(1000)

        # 保存歌词数据
        self.lyrics_data = lyrics_data

        # 显示歌词文本
        if "text" in lyrics_data:
            display_text = lyrics_data["text"]

            # 如果启用了保留咬字信息，同时显示原始识别结果
            if (
                lyrics_data.get("preserve_pronunciation", False)
                and "original_text" in lyrics_data
            ):
                display_text = f"识别结果（含咬字信息）:\n{display_text}\n\n原始识别结果:\n{lyrics_data['original_text']}"

            self.lyrics_text.setText(display_text)

        # 启用对齐按钮
        self.align_lyrics_btn.setEnabled(True)

        # 重新启用识别按钮
        self.recognize_lyrics_btn.setEnabled(True)

        self.statusBar().showMessage("歌词识别完成")

    def _on_lyrics_error(self, error_msg):
        """歌词识别错误回调"""
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "识别错误", f"歌词识别失败: {error_msg}")

        # 重新启用识别按钮
        self.recognize_lyrics_btn.setEnabled(True)

        self.statusBar().showMessage("歌词识别失败")

    def _align_lyrics_with_tags(self):
        """对齐歌词与标签"""
        if not hasattr(self, "lyrics_data") or not self.lyrics_data:
            QMessageBox.warning(self, "警告", "请先识别歌词")
            return

        if not hasattr(self, "last_visual_data") or not self.last_visual_data:
            QMessageBox.warning(self, "警告", "请先处理音频")
            return

        # 获取必要的数据
        prob_matrix = self.last_visual_data["prob_matrix"]
        time_stamps = self.last_visual_data["time_stamps"]
        label_names = self.model.label_names

        # 创建对齐器
        aligner = LyricsTagAligner()

        # 执行对齐
        aligned_result = aligner.align_lyrics_with_tags(
            self.lyrics_data, prob_matrix, time_stamps, label_names
        )

        # 保存对齐结果
        self.aligned_result = aligned_result

        # 询问用户是否保存结果
        reply = QMessageBox.question(
            self,
            "保存结果",
            "是否保存对齐结果到JSON文件？",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self._save_aligned_result()

    def _save_aligned_result(self):
        """保存对齐结果到JSON文件"""
        if not hasattr(self, "aligned_result"):
            QMessageBox.warning(self, "警告", "没有可保存的对齐结果")
            return

        # 获取保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存对齐结果",
            f"{os.path.splitext(os.path.basename(self.current_file))[0]}_aligned.json",
            "JSON文件 (*.json)",
        )

        if file_path:
            try:
                import json

                print(f"准备保存对齐结果到: {file_path}")
                print(f"对齐结果类型: {type(self.aligned_result)}")

                # 验证对齐结果的结构
                if isinstance(self.aligned_result, dict):
                    print(f"对齐结果键: {list(self.aligned_result.keys())}")
                    if "aligned_words" in self.aligned_result:
                        print(
                            f"对齐词数量: {len(self.aligned_result['aligned_words'])}"
                        )

                # 尝试序列化以提前发现问题
                # 转换numpy类型为Python原生类型
                serializable_result = convert_numpy_types(self.aligned_result)
                json_str = json.dumps(serializable_result, ensure_ascii=False, indent=2)
                print(f"JSON序列化成功，长度: {len(json_str)}")

                # 写入文件
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json_str)

                print(f"文件保存成功，大小: {os.path.getsize(file_path)} 字节")
                QMessageBox.information(
                    self, "保存成功", f"对齐结果已保存到:\n{file_path}"
                )
                self.statusBar().showMessage("对齐结果已保存")
            except Exception as e:
                import traceback

                error_msg = (
                    f"保存文件时出错:\n{str(e)}\n\n详细信息:\n{traceback.format_exc()}"
                )
                print(error_msg)
                QMessageBox.critical(self, "保存失败", error_msg)
                self.statusBar().showMessage("保存失败")

    def _update_visualization(self):
        """更新分类结果可视化"""
        # 清除当前图形
        self.figure.clear()

        # 创建子图
        ax = self.figure.add_subplot(111)

        # 设置灰色背景（即使没有音频也保持灰色）
        self.figure.patch.set_facecolor("#3C3C3C")
        ax.set_facecolor("#3C3C3C")

        if not self.last_visual_data:
            # 如果没有数据，只显示空白图形
            ax.set_xlabel("时间 (秒)", color="#E0E0E0")
            ax.set_ylabel("概率", color="#E0E0E0")
            ax.set_title("音频分类结果", color="#E0E0E0")
            ax.grid(True, alpha=0.3, color="#444444")

            # 禁用matplotlib默认图例
            ax.legend().set_visible(False) if ax.get_legend() else None

            # 设置轴和文本颜色
            ax.tick_params(colors="#E0E0E0")
            for spine in ax.spines.values():
                spine.set_color("#E0E0E0")

            # 调整子图布局，确保底部标签有足够空间显示
            self.figure.subplots_adjust(
                bottom=0.15
            )  # 增加底部边距，确保"时间(秒)"标签不被遮挡

            # 刷新画布
            self.canvas.draw()
            # 清空外部图例
            self._update_external_legend([])
            return

        data = self.last_visual_data
        prob_matrix = data["prob_matrix"]
        time_stamps = data["time_stamps"]
        duration = data["duration"]

        # 使用更鲜艳的颜色
        colors = [
            "#FF6B6B",  # 红色
            "#4ECDC4",  # 青色
            "#45B7D1",  # 蓝色
            "#96CEB4",  # 绿色
            "#FFEAA7",  # 黄色
            "#DDA0DD",  # 紫色
            "#FFA07A",  # 浅橙色
            "#98D8C8",  # 薄荷绿
            "#F7DC6F",  # 金黄色
            "#BB8FCE",  # 淡紫色
        ]

        # 存储显示的标签信息，用于外部图例
        displayed_labels = []

        # 绘制每个标签的概率（只绘制选中的标签）
        for i, label_name in enumerate(self.label_names):
            if i == self.no_label_idx:
                continue  # 跳过"无标签"

            # 检查该标签是否被选中
            if (
                label_name not in self.label_checkboxes
                or not self.label_checkboxes[label_name].isChecked()
            ):
                continue  # 跳过未选中的标签

            # 应用阈值
            probs = prob_matrix[:, i]
            color = colors[i % len(colors)]

            # 绘制概率曲线
            ax.plot(
                time_stamps,
                probs,
                label=label_name,
                color=color,
                linewidth=2,
                alpha=0.9,
            )

            # 绘制阈值线下方的区域
            ax.fill_between(
                time_stamps,
                0,
                probs,
                where=(probs >= self.threshold),
                color=color,
                alpha=0.3,
                interpolate=True,
            )

            # 添加到显示的标签列表
            displayed_labels.append((label_name, color))

        # 绘制阈值线
        ax.axhline(
            y=self.threshold,
            color="r",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"阈值: {self.threshold:.2f}",
        )

        # 设置图形属性
        ax.set_xlabel("时间 (秒)", color="#E0E0E0")
        ax.set_ylabel("概率", color="#E0E0E0")
        ax.set_title("音频分类结果", color="#E0E0E0")
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, color="#444444")

        # 禁用matplotlib默认图例
        ax.legend().set_visible(False) if ax.get_legend() else None

        # 设置轴和文本颜色
        ax.tick_params(colors="#E0E0E0")
        for spine in ax.spines.values():
            spine.set_color("#E0E0E0")

        # 调整子图布局，确保底部标签有足够空间显示
        self.figure.subplots_adjust(
            bottom=0.15
        )  # 增加底部边距，确保"时间(秒)"标签不被遮挡

        # 不在图中显示图例，而是使用外部图例
        # 刷新画布
        self.canvas.draw()

        # 更新外部图例
        self._update_external_legend(displayed_labels)

    def _update_external_legend(self, displayed_labels):
        """更新外部图例"""
        # 清除现有图例内容
        for i in reversed(range(self.legend_content_layout.count())):
            self.legend_content_layout.itemAt(i).widget().setParent(None)

        # 添加阈值线图例
        threshold_widget = QWidget()
        threshold_layout = QHBoxLayout(threshold_widget)
        threshold_layout.setContentsMargins(5, 2, 5, 2)

        # 创建阈值线示例
        threshold_line = QLabel("──")
        threshold_line.setStyleSheet(f"color: red; font-weight: bold;")
        threshold_layout.addWidget(threshold_line)

        threshold_text = QLabel(f"阈值: {self.threshold:.2f}")
        threshold_text.setStyleSheet("color: #E0E0E0;")
        threshold_layout.addWidget(threshold_text)

        threshold_layout.addStretch()
        self.legend_content_layout.addWidget(threshold_widget)

        # 添加每个标签的图例
        for label_name, color in displayed_labels:
            label_widget = QWidget()
            label_layout = QHBoxLayout(label_widget)
            label_layout.setContentsMargins(5, 2, 5, 2)

            # 创建颜色示例
            color_box = QLabel("████")
            color_box.setStyleSheet(f"color: {color}; font-weight: bold;")
            label_layout.addWidget(color_box)

            label_text = QLabel(label_name)
            label_text.setStyleSheet("color: #E0E0E0;")
            label_layout.addWidget(label_text)

            label_layout.addStretch()
            self.legend_content_layout.addWidget(label_widget)

    def _init_visualization(self):
        """初始化可视化组件"""
        # 确保图形已初始化
        if not hasattr(self, "figure"):
            return

        # 清除当前图形
        self.figure.clear()

        # 创建子图
        ax = self.figure.add_subplot(111)

        # 设置灰色背景
        self.figure.patch.set_facecolor("#3C3C3C")
        ax.set_facecolor("#3C3C3C")

        # 设置基本属性
        ax.set_xlabel("时间 (秒)", color="#E0E0E0")
        ax.set_ylabel("概率", color="#E0E0E0")
        ax.set_title("音频分类结果", color="#E0E0E0")
        ax.set_xlim(0, 10)  # 默认显示10秒
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, color="#444444")

        # 禁用matplotlib默认图例
        ax.legend().set_visible(False) if ax.get_legend() else None

        # 设置轴和文本颜色
        ax.tick_params(colors="#E0E0E0")
        for spine in ax.spines.values():
            spine.set_color("#E0E0E0")

        # 调整子图布局
        self.figure.subplots_adjust(bottom=0.15)

        # 刷新画布
        self.canvas.draw()

        # 初始化外部图例
        self._update_external_legend([])

    def _update_threshold(self, value):
        """更新分类阈值"""
        self.threshold = value
        self._update_visualization()

    def _toggle_playback(self):
        """切换播放状态"""
        if self.is_playing:
            self.waveform_widget.pause_playback()
            self.play_action.setText("播放")
            self.playback_timer.stop()
        else:
            self.waveform_widget.start_playback()
            self.play_action.setText("暂停")
            self.playback_timer.start()

        self.is_playing = not self.is_playing

        # 如果开始播放，使用统一的时间轴管理函数更新播放状态
        if self.is_playing:
            self._update_timeline_manager(from_progress_bar=False)

    def _stop_playback(self):
        """停止播放"""
        self.waveform_widget.stop_playback()
        self.play_action.setText("播放")
        self.playback_timer.stop()
        self.is_playing = False

        # 重置播放位置
        self.waveform_widget.set_playhead(0)
        # 使用统一的时间轴管理函数更新播放状态
        self._update_timeline_manager(0, from_progress_bar=False)

    def _update_playback(self):
        """更新播放状态"""
        if not self.is_playing:
            return

        # 写入音频缓冲区
        self.waveform_widget._write_audio_buffer()

        # 获取当前播放位置
        current_pos = (
            self.waveform_widget.current_playback_sample
            / self.waveform_widget.sample_rate
        )

        # 使用统一的时间轴管理函数更新播放状态
        self._update_timeline_manager(current_pos, from_progress_bar=False)

        # 检查是否播放完成
        if current_pos >= self.waveform_widget.total_seconds:
            self._stop_playback()

    def _on_mouse_moved(self, position):
        """鼠标移动事件（仅在按住鼠标右键时触发）"""
        if not self.is_dragging_progress:
            self.waveform_widget.set_playhead(position)
            self._update_progress_bar_for_zoom()
            self._update_time_display(position)

    def _on_zoom_changed(self, zoom_level):
        """缩放变化事件"""
        self.zoom_slider.setValue(int(zoom_level))
        self.zoom_label.setText(f"{int(zoom_level)}%")

        # 更新进度条范围以反映当前可见区域
        if self.waveform_widget.total_seconds > 0:
            start_sec, end_sec = self.waveform_widget._get_visible_time_range()
            # 确保进度条范围与当前可见范围对齐
            self._update_progress_bar_for_zoom()

    def _on_zoom_slider_changed(self, value):
        """缩放滑块变化事件"""
        self.waveform_widget.set_zoom(value)
        self.zoom_label.setText(f"{value}%")

    def _on_progress_pressed(self):
        """进度条按下事件"""
        self.is_dragging_progress = True

    def _on_progress_released(self):
        """进度条释放事件"""
        self.is_dragging_progress = False
        # 位置已经在_on_progress_changed中更新，无需重复处理

    def _update_timeline_manager(self, new_position=None, from_progress_bar=False):
        """统一的时间轴管理函数，处理波形窗口和进度条同步

        Args:
            new_position: 新的播放位置（秒），如果为None则使用当前播放位置
            from_progress_bar: 是否来自进度条的操作
        """
        if self.waveform_widget.total_seconds <= 0:
            return

        # 获取当前可见时间范围
        start_sec, end_sec = self.waveform_widget._get_visible_time_range()

        # 确保范围在有效区间内
        start_sec = max(0, start_sec)
        end_sec = min(self.waveform_widget.total_seconds, end_sec)

        # 如果没有提供新位置，使用当前播放位置
        if new_position is None:
            new_position = self.waveform_widget.playhead_position

        # 确保位置在有效范围内
        new_position = max(0, min(self.waveform_widget.total_seconds, new_position))

        # 更新播放头位置
        self.waveform_widget.set_playhead(new_position)

        # 检查播放头是否在可见范围内
        is_playhead_visible = start_sec <= new_position <= end_sec

        # 如果不是来自进度条的操作且播放头不在可见范围内，自动调整视图
        if not from_progress_bar and not is_playhead_visible and self.is_playing:
            # 如果播放头在可见范围左侧，调整视图使播放头在左侧20%位置
            if new_position < start_sec:
                visible_width = end_sec - start_sec
                new_start = max(0, new_position - visible_width * 0.2)
                new_end = new_start + visible_width
                if new_end > self.waveform_widget.total_seconds:
                    new_end = self.waveform_widget.total_seconds
                    new_start = max(0, new_end - visible_width)

                # 计算新的偏移量
                self.waveform_widget.offset_x = (
                    -new_start * self.waveform_widget.zoom_level
                )
                self.waveform_widget.update()

            # 如果播放头在可见范围右侧，调整视图使播放头在右侧80%位置
            elif new_position > end_sec:
                visible_width = end_sec - start_sec
                new_start = max(0, new_position - visible_width * 0.8)
                new_end = new_start + visible_width
                if new_end > self.waveform_widget.total_seconds:
                    new_end = self.waveform_widget.total_seconds
                    new_start = max(0, new_end - visible_width)

                # 计算新的偏移量
                self.waveform_widget.offset_x = (
                    -new_start * self.waveform_widget.zoom_level
                )
                self.waveform_widget.update()

        # 更新进度条（使用绝对时间进度）
        if not from_progress_bar:
            # 计算绝对时间进度（0-1000）
            if self.waveform_widget.total_seconds > 0:
                absolute_progress = (
                    new_position / self.waveform_widget.total_seconds
                ) * 1000
                self.progress_slider.setValue(int(absolute_progress))

        # 更新时间显示
        self._update_time_display(new_position)

    def _update_progress_bar_for_zoom(self):
        """更新进度条以反映缩放后的可见区域"""
        if self.waveform_widget.total_seconds <= 0:
            return

        # 获取当前可见时间范围
        start_sec, end_sec = self.waveform_widget._get_visible_time_range()

        # 计算当前播放头在可见区域中的相对位置
        if start_sec < 0:
            start_sec = 0
        if end_sec > self.waveform_widget.total_seconds:
            end_sec = self.waveform_widget.total_seconds

        visible_range = end_sec - start_sec
        if visible_range <= 0:
            return

        # 计算播放头在可见区域中的相对位置
        if self.waveform_widget.playhead_position < start_sec:
            relative_pos = 0
        elif self.waveform_widget.playhead_position > end_sec:
            relative_pos = 1000
        else:
            relative_pos = (
                (self.waveform_widget.playhead_position - start_sec) / visible_range
            ) * 1000

        # 更新进度条位置
        if not self.is_dragging_progress:
            self.progress_slider.setValue(int(relative_pos))

    def _on_progress_changed(self, value):
        """进度条变化事件"""
        if self.is_dragging_progress and self.waveform_widget.total_seconds > 0:
            # 获取当前可见时间范围
            start_sec, end_sec = self.waveform_widget._get_visible_time_range()

            # 确保范围在有效区间内
            start_sec = max(0, start_sec)
            end_sec = min(self.waveform_widget.total_seconds, end_sec)

            visible_range = end_sec - start_sec
            if visible_range <= 0:
                return

            # 计算在可见范围内的相对位置
            relative_pos = (value / 1000.0) * visible_range
            position = start_sec + relative_pos

            # 使用统一的时间轴管理函数更新播放状态
            self._update_timeline_manager(position, from_progress_bar=True)

    def _update_time_display(self, current_position=None):
        """更新时间显示"""
        if current_position is None:
            current_position = self.waveform_widget.playhead_position

        total_seconds = self.waveform_widget.total_seconds

        # 格式化时间
        current_min = int(current_position // 60)
        current_sec = int(current_position % 60)
        total_min = int(total_seconds // 60)
        total_sec = int(total_seconds % 60)

        time_text = (
            f"{current_min:02d}:{current_sec:02d} / {total_min:02d}:{total_sec:02d}"
        )
        self.time_label.setText(time_text)
