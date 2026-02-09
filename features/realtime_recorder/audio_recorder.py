"""
音频录制器 - 处理实时音频录制和基频提取
"""

import numpy as np
import pyaudio
import librosa
from threading import Thread, Lock
import queue
import time


class AudioRecorder:
    """音频录制器类"""
    
    def __init__(self, sample_rate=22050, buffer_size=256, fft_size=2048):
        """
        初始化音频录制器
        
        Args:
            sample_rate: 采样率 (提高到22050以获得更好的音质)
            buffer_size: 缓冲区大小 (减小到256以进一步减少延迟)
            fft_size: FFT窗口大小
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.fft_size = fft_size
        self.hop_length = fft_size // 4  # STFT跳跃长度
        
        # 录音状态
        self.is_recording = False
        self.stream = None
        self.pa = None
        
        # 数据存储 - 增加缓冲区大小以存储更多历史数据
        self.audio_buffer = np.zeros(buffer_size * 20)  # 增加到20倍的缓冲区
        self.audio_queue = queue.Queue(maxsize=10)  # 限制队列大小防止内存溢出
        self.pitch_buffer = np.zeros(100)  # 存储最近的基频数据
        self.buffer_lock = Lock()
        
        # 基频提取参数
        self.f0_min = 50.0  # 最小基频
        self.f0_max = 500.0  # 最大基频
        
        # 性能优化参数
        self.last_process_time = 0
        self.min_process_interval = 0.02  # 增加最小处理间隔到20ms，减少计算频率
        
    def start_recording(self):
        """开始录音"""
        try:
            self.pa = pyaudio.PyAudio()
            
            # 打开音频流
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            # 启动处理线程
            self.process_thread = Thread(target=self._process_audio)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            return True
        except Exception as e:
            print(f"录音启动失败: {e}")
            return False
    
    def stop_recording(self):
        """停止录音"""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.pa:
            self.pa.terminate()
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if self.is_recording:
            # 将字节数据转换为numpy数组
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # 将数据放入队列，如果队列满了则丢弃最旧的数据
            try:
                self.audio_queue.put_nowait(audio_data)
            except queue.Full:
                # 队列已满，丢弃最旧的数据
                try:
                    self.audio_queue.get_nowait()
                    self.audio_queue.put_nowait(audio_data)
                except queue.Empty:
                    pass
            
        return (None, pyaudio.paContinue)
    
    def _process_audio(self):
        """处理音频数据"""
        window = np.hanning(self.fft_size)
        
        while self.is_recording:
            try:
                # 从队列获取音频数据
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # 更新音频缓冲区
                with self.buffer_lock:
                    # 添加新数据到缓冲区
                    self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                    self.audio_buffer[-len(audio_data):] = audio_data
                    
                    # 控制处理频率，避免过度计算
                    current_time = time.time()
                    if current_time - self.last_process_time < self.min_process_interval:
                        continue
                        
                    self.last_process_time = current_time
                    
                    # 提取基频
                    if len(self.audio_buffer) >= self.fft_size:
                        # 取最新的fft_size长度的数据
                        frame = self.audio_buffer[-self.fft_size:].copy()
                        
                        # 应用窗函数
                        windowed_frame = frame * window
                        
                        # 使用librosa提取基频 - 优化参数
                        f0, voiced_flag, voiced_probs = librosa.pyin(
                            windowed_frame,
                            sr=self.sample_rate,
                            fmin=self.f0_min,
                            fmax=self.f0_max,
                            frame_length=self.fft_size,
                            hop_length=self.hop_length,
                            fill_na=0.0,  # 填充NaN值为0
                            center=False   # 不居中，提高实时性
                        )
                        
                        # 获取有效的基频值
                        if voiced_flag[0] and not np.isnan(f0[0]):
                            pitch = f0[0]
                        else:
                            pitch = 0.0
                            
                        # 更新基频缓冲区
                        self.pitch_buffer = np.roll(self.pitch_buffer, -1)
                        self.pitch_buffer[-1] = pitch
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"音频处理错误: {e}")
                
    def get_latest_audio(self):
        """获取最新的音频数据"""
        with self.buffer_lock:
            return self.audio_buffer.copy()
            
    def get_latest_pitch(self):
        """获取最新的基频数据"""
        with self.buffer_lock:
            return self.pitch_buffer.copy()