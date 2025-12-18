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

    def __init__(self, audio_path, output_dir, model_name="mdx_extra_q"):
        super().__init__()
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        try:
            self.processing_progress.emit(5, "正在加载Demucs模型...")
            
            # 动态导入demucs
            try:
                import demucs.pretrained
                from demucs.apply import apply_model
                from demucs.audio import save_audio
            except ImportError:
                self.processing_error.emit("Demucs库未安装，请使用 pip install demucs 安装")
                return
            
            # 加载模型
            model = demucs.pretrained.get_model(self.model_name)
            model.to(self.device)
            model.eval()
            
            self.processing_progress.emit(15, "正在加载音频文件...")
            
            # 加载音频文件
            waveform, sample_rate = librosa.load(self.audio_path, sr=None, mono=False)
            
            # 如果是单声道，转换为立体声
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform])
            
            # 转换为torch张量
            waveform = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
            
            self.processing_progress.emit(25, "正在进行音源分离...")
            
            # 应用模型进行分离
            with torch.no_grad():
                sources = apply_model(model, waveform)
            
            self.processing_progress.emit(75, "正在保存分离结果...")
            
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 保存分离的音源
            source_names = ["drums", "bass", "other", "vocals"]
            output_files = {}
            base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
            
            for i, name in enumerate(source_names):
                output_path = os.path.join(self.output_dir, f"{base_name}_{name}.wav")
                
                # 转换为numpy并保存
                source_numpy = sources[0, i].cpu().numpy()
                
                # 确保数据在有效范围内
                if np.max(np.abs(source_numpy)) > 0:
                    source_numpy = source_numpy / np.max(np.abs(source_numpy)) * 0.9
                
                # 使用librosa保存
                librosa.output.write_wav(output_path, source_numpy, sample_rate)
                output_files[name] = output_path
            
            # 保存混合音轨（可选）
            mix_output_path = os.path.join(self.output_dir, f"{base_name}_mix.wav")
            librosa.output.write_wav(mix_output_path, waveform[0].cpu().numpy(), sample_rate)
            output_files["mix"] = mix_output_path
            
            self.processing_progress.emit(100, "音源分离完成")
            
            # 返回结果
            result_info = {
                "sample_rate": sample_rate,
                "duration": waveform.shape[-1] / sample_rate,
                "sources": source_names,
                "output_files": output_files
            }
            
            self.processing_complete.emit(self.output_dir, result_info)
            
        except Exception as exc:
            self.processing_error.emit(str(exc))