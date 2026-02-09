"""
波形显示组件 - 优化版本
"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath


class WaveformDisplay(QWidget):
    """波形显示组件 - 优化版本"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 样式设置
        self.background_color = QColor(24, 24, 24)
        self.waveform_color = QColor(66, 135, 245)
        self.grid_color = QColor(40, 40, 40)
        self.center_line_color = QColor(80, 80, 80)
        
        # 音频数据
        self.audio_data = np.zeros(10240)  # 初始空数据
        
        # 性能优化 - 缓存绘制元素
        self._grid_path = None
        self._center_line = None
        self._last_size = None
        
        # 波形绘制优化
        self._waveform_path = QPainterPath()
        self._waveform_needs_update = True
        
    def update_waveform(self, audio_data):
        """更新波形数据"""
        self.audio_data = audio_data
        self._waveform_needs_update = True
        self.update()
        
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 窗口大小改变时，需要重新计算缓存
        self._grid_path = None
        self._center_line = None
        self._waveform_needs_update = True
        
    def paintEvent(self, event):
        """绘制事件 - 优化版本"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), self.background_color)
        
        # 绘制网格（使用缓存）
        self._draw_grid_cached(painter)
        
        # 绘制中心线（使用缓存）
        self._draw_center_line_cached(painter)
        
        # 绘制波形（使用优化的绘制方法）
        self._draw_waveform_optimized(painter)
        
    def _draw_grid_cached(self, painter):
        """绘制网格 - 使用缓存"""
        current_size = (self.width(), self.height())
        
        # 如果缓存不存在或窗口大小改变，重新计算网格
        if self._grid_path is None or self._last_size != current_size:
            self._grid_path = QPainterPath()
            
            # 垂直网格线
            width = self.width()
            height = self.height()
            grid_spacing = 50
            
            for x in range(0, width, grid_spacing):
                self._grid_path.moveTo(x, 0)
                self._grid_path.lineTo(x, height)
                
            # 水平网格线
            for y in range(0, height, grid_spacing):
                self._grid_path.moveTo(0, y)
                self._grid_path.lineTo(width, y)
                
            self._last_size = current_size
            
        # 绘制网格
        pen = QPen(self.grid_color, 1)
        painter.setPen(pen)
        painter.drawPath(self._grid_path)
        
    def _draw_center_line_cached(self, painter):
        """绘制中心线 - 使用缓存"""
        current_size = (self.width(), self.height())
        
        # 如果缓存不存在或窗口大小改变，重新计算中心线
        if self._center_line is None or self._last_size != current_size:
            self._center_line = QPainterPath()
            
            center_y = self.height() // 2
            self._center_line.moveTo(0, center_y)
            self._center_line.lineTo(self.width(), center_y)
            
        # 绘制中心线
        pen = QPen(self.center_line_color, 2)
        painter.setPen(pen)
        painter.drawPath(self._center_line)
        
    def _draw_waveform_optimized(self, painter):
        """绘制波形 - 优化版本"""
        if len(self.audio_data) == 0:
            return
            
        # 设置波形画笔
        pen = QPen(self.waveform_color, 2)
        painter.setPen(pen)
        
        # 计算绘制参数
        width = self.width()
        height = self.height()
        center_y = height // 2
        
        # 使用QPainterPath优化绘制
        if self._waveform_needs_update:
            self._waveform_path = QPainterPath()
            
            # 计算每像素对应的样本数
            samples_per_pixel = max(1, len(self.audio_data) // width)
            
            # 优化：减少绘制点数，每隔几个像素绘制一次
            step = max(1, width // 500)  # 最多绘制500个点
            
            # 绘制波形
            for x in range(0, width, step):
                # 计算当前像素对应的样本范围
                start_idx = x * samples_per_pixel
                end_idx = min((x + 1) * samples_per_pixel, len(self.audio_data))
                
                if start_idx >= len(self.audio_data):
                    break
                    
                # 获取当前像素范围内的样本
                samples = self.audio_data[start_idx:end_idx]
                
                if len(samples) > 0:
                    # 计算最大和最小值
                    max_val = np.max(samples)
                    min_val = np.min(samples)
                    
                    # 转换为屏幕坐标
                    y1 = center_y - int(max_val * center_y * 0.8)
                    y2 = center_y - int(min_val * center_y * 0.8)
                    
                    # 添加到路径
                    if x == 0:
                        self._waveform_path.moveTo(x, y1)
                    else:
                        self._waveform_path.lineTo(x, y1)
                        
            self._waveform_needs_update = False
            
        # 绘制波形路径
        painter.drawPath(self._waveform_path)