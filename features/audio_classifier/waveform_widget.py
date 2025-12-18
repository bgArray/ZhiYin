"""
音频波形显示组件
"""

import numpy as np
import wave
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF, QPointF, QSize, Signal, QPoint, QIODevice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QLinearGradient, QPixmap, QPainterPath
from PySide6.QtMultimedia import QAudioSink, QAudioFormat


class WaveformWidget(QWidget):
    """音频波形显示组件"""
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
        
        # 初始化音频缓冲区
        self._write_audio_buffer()

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
    
    def _write_audio_buffer(self):
        """写入音频缓冲区"""
        if not self.audio_stream:
            return
            
        # 获取音频设备的当前缓冲区状态
        bytes_free = self.audio_sink.bytesFree()
        
        # 计算可以写入的最大样本数
        bytes_per_sample = self.sampwidth
        max_samples = bytes_free // bytes_per_sample
        
        if max_samples > 0:
            # 发送不超过可用缓冲区的样本数
            buffer_size = min(max_samples, self.playback_buffer_size)
            audio_buffer = self.get_next_audio_buffer(buffer_size)
            if audio_buffer:
                self.audio_stream.write(audio_buffer)

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

        # 垂直网格线（根据缩放级别调整间隔）
        if self.zoom_level < 50:
            # 低缩放级别，每5秒一条网格线
            grid_interval = 5
        elif self.zoom_level < 100:
            # 中等缩放级别，每2秒一条网格线
            grid_interval = 2
        elif self.zoom_level < 200:
            # 较高缩放级别，每1秒一条网格线
            grid_interval = 1
        else:
            # 高缩放级别，每0.5秒一条网格线
            grid_interval = 0.5
            
        # 计算网格间隔的像素宽度
        grid_width = grid_interval * self.zoom_level
        
        # 计算起始时间点和x位置
        start_time = -self.offset_x / self.zoom_level
        start_grid = int(start_time / grid_interval) * grid_interval
        x = (start_grid - start_time) * self.zoom_level
        
        # 计算音频结束位置的x坐标
        audio_end_x = self.total_seconds * self.zoom_level + self.offset_x
        
        # 绘制垂直网格线，确保不超过音频总长度
        current_time = start_grid
        while x < self.width():
            if x >= 0 and current_time <= self.total_seconds and x <= audio_end_x:
                painter.drawLine(int(x), 0, int(x), height)
            x += grid_width
            current_time += grid_interval

    def _draw_time_ruler(self, painter):
        pen = QPen(self.time_ruler_color, 1)
        painter.setPen(pen)

        font = QFont("Arial", 8)
        painter.setFont(font)

        # 根据缩放级别动态调整时间间隔
        # 缩放级别越高，显示的时间间隔越小
        if self.zoom_level < 50:
            # 低缩放级别，每5秒显示一个标记
            time_interval = 5
        elif self.zoom_level < 100:
            # 中等缩放级别，每2秒显示一个标记
            time_interval = 2
        elif self.zoom_level < 200:
            # 较高缩放级别，每1秒显示一个标记
            time_interval = 1
        else:
            # 高缩放级别，每0.5秒显示一个标记
            time_interval = 0.5
            
        # 计算每个时间间隔的像素宽度
        interval_width = time_interval * self.zoom_level
        
        # 计算起始时间点
        start_time = -self.offset_x / self.zoom_level
        start_interval = int(start_time / time_interval) * time_interval
        
        # 计算起始x位置
        x = (start_interval - start_time) * self.zoom_level
        
        # 计算音频结束位置的x坐标
        audio_end_x = self.total_seconds * self.zoom_level + self.offset_x
        
        # 绘制时间标记
        current_time = start_interval
        while x < self.width():
            if x >= 0:
                # 确保不超过音频总长度
                if current_time <= self.total_seconds and x <= audio_end_x:
                    # 格式化时间显示
                    if time_interval < 1:
                        # 如果间隔小于1秒，显示毫秒
                        time_text = f"{int(current_time // 60)}:{int(current_time % 60):02d}.{int((current_time % 1) * 10)}"
                    else:
                        # 否则只显示分钟和秒
                        time_text = f"{int(current_time // 60)}:{int(current_time % 60):02d}"
                    
                    painter.drawText(int(x) + 5, 15, time_text)
                    
                    # 绘制刻度线
                    painter.drawLine(int(x), 20, int(x), 25)
            
            x += interval_width
            current_time += time_interval

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
        
        # 计算音频结束位置的x坐标
        audio_end_x = self.total_seconds * self.zoom_level + self.offset_x
        
        # 如果音频结束位置在可见区域内，绘制黑底区域
        if 0 <= audio_end_x <= self.width():
            # 设置黑底画笔
            black_pen = QPen(Qt.black, 1)
            painter.setPen(black_pen)
            
            # 绘制黑底矩形
            painter.fillRect(int(audio_end_x), 0, self.width() - int(audio_end_x), self.height(), Qt.black)
            
            # 恢复波形画笔
            painter.setPen(pen)
        
        start_sample = max(0, int(start_sec * self.sample_rate))
        end_sample = min(len(self.audio_data), int(end_sec * self.sample_rate))

        if start_sample >= end_sample:
            return

        # 抽取样本以适应显示宽度
        samples = self.audio_data[start_sample:end_sample]
        num_pixels = min(self.width(), int(audio_end_x))  # 只绘制到音频结束位置
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
        # 计算播放头位置
        playhead_x = self.playhead_position * self.zoom_level + self.offset_x

        # 只有当播放头在可见范围内时才绘制
        if 0 <= playhead_x <= self.width():
            pen = QPen(self.playhead_color, 2)
            painter.setPen(pen)
            
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

        # 只在按住鼠标右键时发送鼠标位置信号
        if event.buttons() & Qt.RightButton:
            mouse_sec = max(0, min(
                self.total_seconds,
                (event.position().x() - self.offset_x) / self.zoom_level
            ))
            self.mouseMoved.emit(int(mouse_sec))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            
            # 确保拖动后不会显示超出音频长度的区域
            if self.total_seconds > 0:
                # 计算当前可见的时间范围
                start_sec = -self.offset_x / self.zoom_level
                end_sec = start_sec + self.width() / self.zoom_level
                
                # 如果开始时间小于0，调整偏移
                if start_sec < 0:
                    self.offset_x = 0
                
                # 如果结束时间超过音频总长度，调整偏移
                if end_sec > self.total_seconds:
                    max_offset = self.total_seconds * self.zoom_level - self.width()
                    self.offset_x = min(0, max_offset)
                
                self.update()

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
        
        # 确保缩放后不会显示超出音频长度的区域
        if self.total_seconds > 0:
            # 计算当前可见的时间范围
            start_sec = -self.offset_x / self.zoom_level
            end_sec = start_sec + self.width() / self.zoom_level
            
            # 如果开始时间小于0，调整偏移
            if start_sec < 0:
                self.offset_x = 0
            
            # 如果结束时间超过音频总长度，调整偏移
            if end_sec > self.total_seconds:
                max_offset = self.total_seconds * self.zoom_level - self.width()
                self.offset_x = min(0, max_offset)
        
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
        
        # 确保缩放后不会显示超出音频长度的区域
        if self.total_seconds > 0:
            # 计算当前可见的时间范围
            start_sec = -self.offset_x / self.zoom_level
            end_sec = start_sec + self.width() / self.zoom_level
            
            # 如果开始时间小于0，调整偏移
            if start_sec < 0:
                self.offset_x = 0
            
            # 如果结束时间超过音频总长度，调整偏移
            if end_sec > self.total_seconds:
                max_offset = self.total_seconds * self.zoom_level - self.width()
                self.offset_x = min(0, max_offset)
        
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