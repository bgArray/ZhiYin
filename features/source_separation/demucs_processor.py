"""
Demucs音源分离处理器
"""

import os
import torch
import numpy as np
import librosa
from PySide6.QtCore import QThread, Signal


class DemucsProcessorThread(QThread):
    """后台Demucs处理线程，避免主线程阻塞"""

    processing_complete = Signal(str, dict)  # 输出路径, 分离结果信息
    processing_progress = Signal(int, str)  # 进度百分比, 状态描述
    processing_error = Signal(str)

    def __init__(self, audio_path, output_dir):
        super().__init__()
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        try:
            self.processing_progress.emit(5, "正在加载Demucs模型...")
            
            # 动态导入demucs
            try:
                from demucs import separate
            except ImportError:
                self.processing_error.emit("Demucs库未安装，请使用 pip install demucs 安装")
                return
            
            self.processing_progress.emit(25, "正在进行音源分离...")
            
            # 使用demucs.separate进行分离，与您的原始代码类似
            # 创建临时输出目录
            temp_output = os.path.join(self.output_dir, "temp_demucs_output")
            os.makedirs(temp_output, exist_ok=True)
            
            # 调用demucs分离
            separate.main(["--out", temp_output, self.audio_path])
            
            self.processing_progress.emit(75, "正在整理分离结果...")
            
            # 获取分离后的文件
            base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
            demucs_output_path = os.path.join(temp_output, "htdemucs", base_name)
            
            # 检查输出文件是否存在
            if not os.path.exists(demucs_output_path):
                self.processing_error.emit("分离失败：找不到输出文件")
                return
            
            # 整理输出文件
            source_names = ["drums", "bass", "other", "vocals"]
            output_files = {}
            
            for name in source_names:
                source_file = os.path.join(demucs_output_path, f"{name}.wav")
                if os.path.exists(source_file):
                    # 移动到最终输出目录
                    output_path = os.path.join(self.output_dir, f"{base_name}_{name}.wav")
                    os.rename(source_file, output_path)
                    output_files[name] = output_path
            
            # 清理临时目录
            import shutil
            shutil.rmtree(temp_output)
            
            self.processing_progress.emit(100, "音源分离完成")
            
            # 获取音频信息
            waveform, sample_rate = librosa.load(self.audio_path, sr=None)
            duration = len(waveform) / sample_rate
            
            # 返回结果
            result_info = {
                "sample_rate": sample_rate,
                "duration": duration,
                "sources": source_names,
                "output_files": output_files
            }
            
            self.processing_complete.emit(self.output_dir, result_info)
            
        except Exception as exc:
            self.processing_error.emit(str(exc))