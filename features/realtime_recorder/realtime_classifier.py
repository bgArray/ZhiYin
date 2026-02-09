"""
实时声音分类处理器
"""

import numpy as np
import librosa
import torch
import threading
import time
from collections import deque
from features.audio_classifier.model import AudioClassifierModel


class RealtimeAudioClassifier:
    """实时音频分类器"""
    
    def __init__(self, model_path="best_new_models/final_multilabel_cnn_lstm_model_m8.pth", 
                 label_mapping_path="best_new_models/label_to_idx_m8.json",
                 sample_rate=22050, buffer_duration=2.0):
        """
        初始化实时音频分类器
        
        Args:
            model_path: 模型文件路径
            label_mapping_path: 标签映射文件路径
            sample_rate: 音频采样率
            buffer_duration: 缓冲区时长（秒）
        """
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # 加载模型
        self.model = AudioClassifierModel(model_path, label_mapping_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 音频缓冲区
        self.audio_buffer = np.zeros(self.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # 分类结果缓存
        self.classification_results = deque(maxlen=5)  # 保存最近5次分类结果
        self.last_classification_time = 0
        self.classification_interval = 1.0  # 分类间隔（秒）- 增加到1秒以减少计算量
        
        # 降噪参数
        self.noise_reduction_enabled = False
        self.noise_profile = None
        self.noise_reduction_strength = 0.3
        
    def add_audio_data(self, audio_data):
        """
        添加新的音频数据到缓冲区
        
        Args:
            audio_data: 新的音频数据
        """
        with self.buffer_lock:
            # 将新数据添加到缓冲区
            data_len = min(len(audio_data), self.buffer_size)
            self.audio_buffer = np.roll(self.audio_buffer, -data_len)
            self.audio_buffer[-data_len:] = audio_data[:data_len]
            
    def enable_noise_reduction(self, enabled=True, strength=0.3):
        """
        启用或禁用降噪
        
        Args:
            enabled: 是否启用降噪
            strength: 降噪强度 (0.0-1.0)
        """
        self.noise_reduction_enabled = enabled
        self.noise_reduction_strength = max(0.0, min(1.0, strength))
        
    def capture_noise_profile(self, duration=1.0):
        """
        捕获噪声 profile（用于降噪）
        
        Args:
            duration: 捕获时长（秒）
        """
        with self.buffer_lock:
            if len(self.audio_buffer) > 0:
                # 使用当前缓冲区的前一部分作为噪声 profile
                noise_samples = int(duration * self.sample_rate)
                noise_data = self.audio_buffer[:min(noise_samples, len(self.audio_buffer))]
                self.noise_profile = np.abs(np.fft.rfft(noise_data))
                
    def apply_noise_reduction(self, audio_data):
        """
        对音频数据应用降噪
        
        Args:
            audio_data: 原始音频数据
            
        Returns:
            降噪后的音频数据
        """
        if not self.noise_reduction_enabled or self.noise_profile is None:
            return audio_data
            
        # 简单的谱减法降噪
        fft = np.fft.rfft(audio_data)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # 减少噪声频谱
        reduced_magnitude = magnitude - (self.noise_profile * self.noise_reduction_strength)
        reduced_magnitude = np.maximum(reduced_magnitude, magnitude * 0.1)  # 防止过度降噪
        
        # 重构音频
        reduced_fft = reduced_magnitude * np.exp(1j * phase)
        return np.fft.irfft(reduced_fft).astype(np.float32)
        
    def classify(self):
        """
        对当前缓冲区中的音频进行分类
        
        Returns:
            分类结果字典，包含标签和概率
        """
        current_time = time.time()
        
        # 控制分类频率
        if current_time - self.last_classification_time < self.classification_interval:
            if self.classification_results:
                return self.classification_results[-1]
            else:
                return {"labels": [], "probabilities": []}
                
        self.last_classification_time = current_time
        
        with self.buffer_lock:
            # 获取音频数据
            audio_data = self.audio_buffer.copy()
            
        # 应用降噪
        if self.noise_reduction_enabled:
            audio_data = self.apply_noise_reduction(audio_data)
            
        # 提取梅尔频谱图
        try:
            n_fft = 2048
            hop_length = 512
            n_mels = 128
            
            # 确保音频数据足够长
            if len(audio_data) < n_fft:
                audio_data = np.pad(audio_data, (0, n_fft - len(audio_data)), mode='constant')
                
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_sequence = mel_spectrogram_db.T
            
            # 调整维度
            target_dim = 74
            if mel_sequence.shape[1] >= target_dim:
                mel_sequence = mel_sequence[:, :target_dim]
            else:
                pad_width = target_dim - mel_sequence.shape[1]
                mel_sequence = np.pad(
                    mel_sequence,
                    pad_width=((0, 0), (0, pad_width)),
                    mode="constant",
                    constant_values=0.0,
                )
                
            # 模型预测
            prob_matrix = self.model.predict(mel_sequence)
            
            # 计算平均概率
            avg_probs = np.mean(prob_matrix, axis=0)
            
            # 获取概率大于40%的标签
            threshold = 0.4
            high_prob_indices = np.where(avg_probs > threshold)[0]
            
            labels = []
            probabilities = []
            
            for idx in high_prob_indices:
                if idx < len(self.model.label_names):
                    labels.append(self.model.label_names[idx])
                    probabilities.append(float(avg_probs[idx]))
                    
            # 按概率排序
            if labels:
                sorted_pairs = sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True)
                labels = [pair[0] for pair in sorted_pairs]
                probabilities = [pair[1] for pair in sorted_pairs]
                
            result = {"labels": labels, "probabilities": probabilities}
            
            # 添加到结果缓存
            self.classification_results.append(result)
            
            return result
            
        except Exception as e:
            print(f"分类错误: {e}")
            return {"labels": [], "probabilities": []}