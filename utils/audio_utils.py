"""
音频处理工具函数
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Tuple, Optional, Union


def load_audio(file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    加载音频文件
    
    Args:
        file_path: 音频文件路径
        sr: 目标采样率，None表示使用原始采样率
        
    Returns:
        tuple: (音频数据, 采样率)
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"加载音频文件失败: {e}")


def save_audio(file_path: str, audio_data: np.ndarray, sr: int) -> None:
    """
    保存音频文件
    
    Args:
        file_path: 输出文件路径
        audio_data: 音频数据
        sr: 采样率
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存音频
        sf.write(file_path, audio_data, sr)
    except Exception as e:
        raise RuntimeError(f"保存音频文件失败: {e}")


def extract_mel_spectrogram(
    audio_data: np.ndarray, 
    sr: int, 
    n_fft: int = 2048, 
    hop_length: int = 512, 
    n_mels: int = 128
) -> np.ndarray:
    """
    提取梅尔频谱图
    
    Args:
        audio_data: 音频数据
        sr: 采样率
        n_fft: FFT窗口大小
        hop_length: 跳跃长度
        n_mels: 梅尔滤波器数量
        
    Returns:
        np.ndarray: 梅尔频谱图 (时间帧, 梅尔频带)
    """
    try:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram_db.T  # 转置为 (时间帧, 梅尔频带)
    except Exception as e:
        raise RuntimeError(f"提取梅尔频谱图失败: {e}")


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    音频归一化
    
    Args:
        audio_data: 音频数据
        
    Returns:
        np.ndarray: 归一化后的音频数据
    """
    if np.max(np.abs(audio_data)) > 0:
        return audio_data / np.max(np.abs(audio_data))
    return audio_data


def apply_fade(
    audio_data: np.ndarray, 
    sr: int, 
    fade_in_duration: float = 0.1, 
    fade_out_duration: float = 0.1
) -> np.ndarray:
    """
    应用淡入淡出效果
    
    Args:
        audio_data: 音频数据
        sr: 采样率
        fade_in_duration: 淡入时长（秒）
        fade_out_duration: 淡出时长（秒）
        
    Returns:
        np.ndarray: 处理后的音频数据
    """
    result = audio_data.copy()
    length = len(audio_data)
    
    # 淡入
    fade_in_samples = int(fade_in_duration * sr)
    if fade_in_samples > 0:
        fade_in = np.linspace(0, 1, fade_in_samples)
        result[:fade_in_samples] *= fade_in
    
    # 淡出
    fade_out_samples = int(fade_out_duration * sr)
    if fade_out_samples > 0:
        fade_out = np.linspace(1, 0, fade_out_samples)
        result[-fade_out_samples:] *= fade_out
    
    return result


def trim_silence(
    audio_data: np.ndarray, 
    sr: int, 
    threshold: float = 0.01, 
    frame_length: int = 2048, 
    hop_length: int = 512
) -> np.ndarray:
    """
    去除音频前后静音
    
    Args:
        audio_data: 音频数据
        sr: 采样率
        threshold: 静音阈值
        frame_length: 帧长度
        hop_length: 跳跃长度
        
    Returns:
        np.ndarray: 去除静音后的音频数据
    """
    try:
        # 使用librosa的trim函数
        trimmed, _ = librosa.effects.trim(
            audio_data, 
            top_db=20 * np.log10(threshold),
            frame_length=frame_length,
            hop_length=hop_length
        )
        return trimmed
    except Exception as e:
        raise RuntimeError(f"去除静音失败: {e}")


def get_audio_duration(file_path: str) -> float:
    """
    获取音频文件时长
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        float: 音频时长（秒）
    """
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        raise RuntimeError(f"获取音频时长失败: {e}")


def mix_audio(
    audio1: np.ndarray, 
    audio2: np.ndarray, 
    weight1: float = 0.5, 
    weight2: float = 0.5
) -> np.ndarray:
    """
    混合两个音频
    
    Args:
        audio1: 第一个音频
        audio2: 第二个音频
        weight1: 第一个音频的权重
        weight2: 第二个音频的权重
        
    Returns:
        np.ndarray: 混合后的音频
    """
    # 确保两个音频长度相同
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # 归一化权重
    total_weight = weight1 + weight2
    if total_weight > 0:
        weight1 /= total_weight
        weight2 /= total_weight
    
    # 混合音频
    mixed = weight1 * audio1 + weight2 * audio2
    
    # 归一化结果
    return normalize_audio(mixed)