"""
歌词识别处理器
使用Whisper进行中文歌词识别和时间戳提取
"""

import whisper
import numpy as np
import librosa
import json
import re
from PySide6.QtCore import QThread, Signal


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

# 尝试导入简繁体转换库
try:
    import opencc
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False
    print("警告: 未安装opencc库，无法进行简繁体转换。可通过pip install opencc-python-reimplemented安装。")


class LyricsRecognizerThread(QThread):
    """后台歌词识别线程，避免主线程阻塞"""

    recognition_complete = Signal(dict)  # 识别结果
    recognition_error = Signal(str)  # 错误信息
    recognition_progress = Signal(int, str)  # 进度百分比, 状态描述

    def __init__(self, audio_path, model_size="base", preserve_pronunciation=False):
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.preserve_pronunciation = preserve_pronunciation  # 是否保留咬字信息
        self.model = None

    def run(self):
        try:
            # 加载Whisper模型
            self.recognition_progress.emit(10, "正在加载Whisper模型...")
            self.model = whisper.load_model(self.model_size)
            
            # 加载音频
            self.recognition_progress.emit(30, "正在加载音频文件...")
            audio = whisper.load_audio(self.audio_path)
            
            # 识别音频并获取时间戳
            self.recognition_progress.emit(50, "正在进行歌词识别...")
            result = self.model.transcribe(
                audio, 
                language="zh",  # 指定为中文
                word_timestamps=True,  # 启用词级时间戳
                task="transcribe"  # 转录任务
            )
            
            # 处理识别结果
            self.recognition_progress.emit(80, "正在处理识别结果...")
            processed_result = self._process_result(result)
            
            # 完成识别
            self.recognition_progress.emit(100, "歌词识别完成")
            self.recognition_complete.emit(processed_result)
            
        except Exception as exc:
            self.recognition_error.emit(f"歌词识别失败: {str(exc)}")

    def _convert_to_simplified(self, text):
        """将繁体中文转换为简体中文"""
        if not HAS_OPENCC or not text:
            return text
        
        try:
            # 创建转换器实例（繁体转简体）
            converter = opencc.OpenCC('t2s')
            return converter.convert(text)
        except Exception as e:
            print(f"简繁体转换失败: {e}")
            return text

    def _process_result(self, result):
        """处理Whisper识别结果，提取汉字时间戳"""
        # 获取原始文本
        original_text = result.get("text", "")
        
        # 根据设置决定是否转换为简体中文
        if self.preserve_pronunciation:
            # 保留原始识别结果（包含咬字信息）
            processed_text = original_text
        else:
            # 转换为简体中文
            processed_text = self._convert_to_simplified(original_text)
        
        processed_data = {
            "text": processed_text,
            "original_text": original_text if self.preserve_pronunciation else None,
            "preserve_pronunciation": self.preserve_pronunciation,
            "segments": [],
            "words": []
        }
        
        # 处理段落级别的时间戳
        for segment in result.get("segments", []):
            # 根据设置决定是否转换段落文本
            if self.preserve_pronunciation:
                segment_text = segment.get("text", "").strip()
            else:
                segment_text = self._convert_to_simplified(segment.get("text", "").strip())
            
            segment_data = {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment_text,
                "original_text": segment.get("text", "").strip() if self.preserve_pronunciation else None
            }
            processed_data["segments"].append(segment_data)
            
            # 处理词级别的时间戳
            for word in segment.get("words", []):
                word_text = word.get("word", "").strip()
                if word_text:  # 确保词不为空
                    if self.preserve_pronunciation:
                        # 保留原始识别结果（包含咬字信息）
                        processed_word = word_text
                    else:
                        # 转换为简体中文
                        processed_word = self._convert_to_simplified(word_text)
                    
                    # 分离中文汉字和非汉字
                    chinese_chars = re.findall(r'[\u4e00-\u9fff]', processed_word)
                    non_chinese = re.sub(r'[\u4e00-\u9fff]', '', processed_word)
                    
                    # 为每个汉字创建单独的时间戳
                    if chinese_chars:
                        # 计算每个汉字的时间分配
                        start_time = word.get("start", 0)
                        end_time = word.get("end", start_time + 0.1)
                        duration = end_time - start_time
                        char_duration = duration / len(chinese_chars)
                        
                        for i, char in enumerate(chinese_chars):
                            char_start = start_time + i * char_duration
                            char_end = char_start + char_duration
                            
                            word_data = {
                                "word": char,
                                "start": char_start,
                                "end": char_end,
                                "confidence": word.get("probability", 0.0),
                                "original_word": word_text if self.preserve_pronunciation else None
                            }
                            processed_data["words"].append(word_data)
                    
                    # 处理非中文部分（如英文、数字等）
                    if non_chinese:
                        word_data = {
                            "word": non_chinese,
                            "start": word.get("start", 0),
                            "end": word.get("end", 0),
                            "confidence": word.get("probability", 0.0),
                            "original_word": word_text if self.preserve_pronunciation else None
                        }
                        processed_data["words"].append(word_data)
        
        # 按时间戳排序
        processed_data["words"].sort(key=lambda x: x["start"])
        
        return processed_data

    def save_result(self, result, output_path):
        """保存识别结果到JSON文件"""
        try:
            # 转换numpy类型为Python原生类型
            serializable_result = convert_numpy_types(result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            return False