"""
基频显示组件 - 优化版本
"""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QFont


class PitchDisplay(QWidget):
    """基频显示组件 - 优化版本"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 样式设置
        self.background_color = QColor(24, 24, 24)
        self.pitch_color = QColor(255, 100, 100)
        self.grid_color = QColor(40, 40, 40)
        self.text_color = QColor(200, 200, 200)
        
        # 基频数据
        self.pitch_data = np.zeros(100)  # 初始空数据
        
        # 显示范围
        self.min_pitch = 50
        self.max_pitch = 500
        
        # 性能优化 - 缓存绘制元素
        self._grid_path = None
        self._last_size = None
        
        # 基频绘制优化
        self._pitch_path = QPainterPath()
        self._pitch_needs_update = True  # 最大显示频率
        
    def update_pitch(self, pitch_data):
        """更新基频数据"""
        self.pitch_data = pitch_data
        self._pitch_needs_update = True
        self.update()
        
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 窗口大小改变时，需要重新计算缓存
        self._grid_path = None
        self._pitch_needs_update = True
        
    def paintEvent(self, event):
        """绘制事件 - 优化版本"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), self.background_color)
        
        # 绘制网格（使用缓存）
        self._draw_grid_cached(painter)
        
        # 绘制基频曲线（使用优化的绘制方法）
        self._draw_pitch_curve_optimized(painter)
        
        # 绘制当前基频值
        self._draw_current_pitch(painter)
        
    def _draw_grid_cached(self, painter):
        """绘制网格 - 使用缓存"""
        current_size = (self.width(), self.height())
        
        # 如果缓存不存在或窗口大小改变，重新计算网格
        if self._grid_path is None or self._last_size != current_size:
            self._grid_path = QPainterPath()
            
            width = self.width()
            height = self.height()
            
            # 垂直网格线
            grid_spacing_x = width // 10
            for x in range(0, width, grid_spacing_x):
                self._grid_path.moveTo(x, 0)
                self._grid_path.lineTo(x, height)
                
            # 水平网格线
            grid_spacing_y = height // 5
            for i in range(6):
                y = i * grid_spacing_y
                self._grid_path.moveTo(0, y)
                self._grid_path.lineTo(width, y)
                
            self._last_size = current_size
            
        # 绘制网格
        pen = QPen(self.grid_color, 1)
        painter.setPen(pen)
        painter.drawPath(self._grid_path)
        
        # 绘制频率标记（每次都绘制，因为文本不缓存）
        font = QFont("Arial", 8)
        painter.setFont(font)
        
        height = self.height()
        grid_spacing_y = height // 5
        
        for i in range(6):
            y = i * grid_spacing_y
            
            # 计算对应的频率值
            freq = self.max_pitch - (i / 5) * (self.max_pitch - self.min_pitch)
            
            # 绘制频率标记
            painter.setPen(self.text_color)
            painter.drawText(5, y - 2, f"{freq:.0f} Hz")
            
            # 恢复网格线颜色
            painter.setPen(self.grid_color)
            
    def _draw_pitch_curve_optimized(self, painter):
        """绘制基频曲线 - 优化版本"""
        if len(self.pitch_data) == 0:
            return
            
        # 设置基频曲线画笔
        pen = QPen(self.pitch_color, 2)
        painter.setPen(pen)
        
        # 计算绘制参数
        width = self.width()
        height = self.height()
        
        # 使用QPainterPath优化绘制
        if self._pitch_needs_update:
            self._pitch_path = QPainterPath()
            
            # 计算每像素对应的数据点
            points_per_pixel = max(1, len(self.pitch_data) // width)
            
            # 优化：减少绘制点数，每隔几个像素绘制一次
            step = max(1, width // 500)  # 最多绘制500个点
            
            # 绘制基频曲线
            prev_x = 0
            prev_y = height
            path_started = False
            
            for x in range(0, width, step):
                # 计算当前像素对应的数据范围
                start_idx = x * points_per_pixel
                end_idx = min((x + 1) * points_per_pixel, len(self.pitch_data))
                
                if start_idx >= len(self.pitch_data):
                    break
                    
                # 获取当前像素范围内的基频值
                pitch_values = self.pitch_data[start_idx:end_idx]
                
                if len(pitch_values) > 0:
                    # 计算平均基频值
                    avg_pitch = np.mean(pitch_values)
                    
                    # 跳过无效值（0或NaN）
                    if avg_pitch <= 0 or np.isnan(avg_pitch):
                        path_started = False
                        continue
                        
                    # 将频率值转换为屏幕坐标
                    normalized_freq = (avg_pitch - self.min_pitch) / (self.max_pitch - self.min_pitch)
                    normalized_freq = max(0, min(1, normalized_freq))  # 限制在[0,1]范围内
                    
                    y = height - int(normalized_freq * height)
                    
                    # 添加到路径
                    if not path_started:
                        self._pitch_path.moveTo(x, y)
                        path_started = True
                    else:
                        self._pitch_path.lineTo(x, y)
                        
            self._pitch_needs_update = False
            
        # 绘制基频路径
        painter.drawPath(self._pitch_path)
                
    def _draw_current_pitch(self, painter):
        """绘制当前基频值"""
        if len(self.pitch_data) == 0:
            return
            
        # 获取最新的有效基频值
        latest_pitch = 0.0
        for i in range(len(self.pitch_data) - 1, -1, -1):
            if self.pitch_data[i] > 0 and not np.isnan(self.pitch_data[i]):
                latest_pitch = self.pitch_data[i]
                break
                
        if latest_pitch <= 0:
            return
            
        # 设置文本样式
        font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(font)
        painter.setPen(self.pitch_color)
        
        # 绘制当前基频值
        text = f"当前基频: {latest_pitch:.1f} Hz"
        painter.drawText(self.width() - 180, 25, text)
        
        # 绘制基频范围标记
        if latest_pitch < self.min_pitch:
            range_text = "(低于检测范围)"
            painter.setPen(QColor(255, 100, 100))
        elif latest_pitch > self.max_pitch:
            range_text = "(高于检测范围)"
            painter.setPen(QColor(255, 100, 100))
        else:
            range_text = ""
            
        if range_text:
            painter.drawText(self.width() - 180, 45, range_text)