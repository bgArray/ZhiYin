"""
音频处理线程
"""

import numpy as np
import librosa
from PySide6.QtCore import QThread, Signal


class AudioProcessorThread(QThread):
    """后台音频处理线程，避免主线程阻塞"""

    processing_complete = Signal(object, object, object, object, object, float)
    processing_error = Signal(str)
    processing_progress = Signal(int, str)  # 进度百分比, 状态描述

    def __init__(
        self,
        audio_path,
        model,
        max_length=512,
        target_dim=74,
    ):
        super().__init__()
        self.audio_path = audio_path
        self.model = model
        self.max_length = max_length
        self.target_dim = target_dim

    def run(self):
        try:
            # 加载音频
            self.processing_progress.emit(30, "正在加载音频文件...")
            y, sr = librosa.load(self.audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            self.processing_progress.emit(40, "正在提取音频特征...")

            # 提取梅尔频谱图
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
            
            self.processing_progress.emit(60, "正在处理音频数据...")

            # 调整维度
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
            
            self.processing_progress.emit(70, "正在进行模型预测...")

            # 模型预测
            prob_matrix = self.model.predict(mel_sequence)
            
            self.processing_progress.emit(85, "正在计算时间戳...")

            # 计算时间戳
            frame_duration = hop_length / sr
            time_stamps = np.arange(prob_matrix.shape[0]) * frame_duration
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