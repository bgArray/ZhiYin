"""
歌词搜索与优化模块
通过联网搜索获取准确的歌词，并与音频时间戳对齐
"""

import json
import os
import re
import time
from urllib.parse import quote
from PySide6.QtCore import QThread, Signal

# 尝试导入requests库用于网络请求
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("警告: 未安装requests库，无法进行歌词搜索。可通过pip install requests安装。")


class LyricsOptimizerThread(QThread):
    """歌词优化线程，用于搜索准确歌词并与时间戳对齐"""
    
    optimization_complete = Signal(dict)  # 优化结果
    optimization_error = Signal(str)  # 错误信息
    optimization_progress = Signal(int, str)  # 进度百分比, 状态描述
    
    def __init__(self, audio_path, recognized_lyrics, song_info=None):
        super().__init__()
        self.audio_path = audio_path
        self.recognized_lyrics = recognized_lyrics
        self.song_info = song_info or {}  # 可选的歌曲信息，如歌名、歌手等
        
    def run(self):
        """运行歌词优化流程"""
        try:
            self.optimization_progress.emit(10, "正在分析识别结果...")
            
            # 提取关键词用于搜索
            keywords = self._extract_keywords()
            
            self.optimization_progress.emit(30, "正在搜索歌词...")
            
            # 搜索歌词
            accurate_lyrics = self._search_lyrics(keywords)
            
            if not accurate_lyrics:
                # 如果搜索失败，使用原始识别结果
                self.optimization_progress.emit(90, "使用原始识别结果...")
                optimized_result = self._create_fallback_result()
            else:
                self.optimization_progress.emit(60, "正在对齐时间戳...")
                
                # 对齐准确歌词与原始时间戳
                optimized_result = self._align_lyrics_with_timestamps(accurate_lyrics)
            
            self.optimization_progress.emit(100, "歌词优化完成")
            self.optimization_complete.emit(optimized_result)
            
        except Exception as e:
            self.optimization_error.emit(f"歌词优化失败: {str(e)}")
    
    def _extract_keywords(self):
        """从识别结果中提取关键词用于搜索"""
        # 提取主要歌词文本
        text = self.recognized_lyrics.get("text", "")
        
        # 移除标点符号和特殊字符
        cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 分割成词
        words = cleaned_text.split()
        
        # 过滤掉常见无意义词
        stop_words = {'的', '了', '是', '在', '我', '你', '他', '她', '它', '们', '这', '那', '有', '和', '与', '或'}
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 1]
        
        # 取前几个最有代表性的词作为关键词
        keywords = meaningful_words[:5]
        
        # 如果有歌曲信息，添加到关键词中
        if self.song_info.get("title"):
            keywords.insert(0, self.song_info["title"])
        if self.song_info.get("artist"):
            keywords.insert(1, self.song_info["artist"])
            
        return keywords
    
    def _search_lyrics(self, keywords):
        """使用关键词搜索歌词"""
        if not HAS_REQUESTS or not keywords:
            return None
            
        try:
            # 构建搜索查询
            query = " ".join(keywords)
            encoded_query = quote(query)
            
            # 使用多个歌词源尝试搜索
            lyrics_sources = [
                self._search_from_source1,
                self._search_from_source2,
                self._search_from_source3
            ]
            
            # 尝试使用不同的关键词组合
            keyword_combinations = [
                keywords,  # 原始关键词
                keywords[:3],  # 前3个关键词
                [keywords[0]] if keywords else [],  # 只使用第一个关键词
            ]
            
            # 如果有歌曲信息，添加额外的搜索组合
            if self.song_info.get("song_name"):
                keyword_combinations.insert(0, [self.song_info["song_name"]])
                if self.song_info.get("artist_name"):
                    keyword_combinations.insert(0, [self.song_info["song_name"], self.song_info["artist_name"]])
            
            # 尝试不同的关键词组合和搜索源
            for kw_combination in keyword_combinations:
                if not kw_combination:
                    continue
                    
                self.optimization_progress.emit(35, f"正在搜索关键词: {' '.join(kw_combination)}")
                
                for search_func in lyrics_sources:
                    try:
                        # 为每个搜索源设置超时
                        lyrics = search_func(" ".join(kw_combination))
                        if lyrics and self._validate_lyrics(lyrics):
                            return lyrics
                    except Exception as e:
                        print(f"搜索失败: {e}")
                        continue
                        
            # 所有搜索都失败，尝试使用本地歌词库（如果有）
            self.optimization_progress.emit(50, "正在尝试本地歌词库...")
            local_lyrics = self._search_local_lyrics(keywords)
            if local_lyrics and self._validate_lyrics(local_lyrics):
                return local_lyrics
                    
            return None
            
        except Exception as e:
            print(f"歌词搜索出错: {e}")
            return None
    
    def _search_from_source1(self, query):
        """从歌词源1搜索（示例实现）"""
        # 这里应该实现实际的歌词搜索逻辑
        # 由于版权问题，这里只提供框架
        
        # 示例代码（实际使用时需要替换为真实的API）:
        # url = f"https://api.example.com/search?q={query}"
        # response = requests.get(url, timeout=10)
        # if response.status_code == 200:
        #     data = response.json()
        #     return data.get("lyrics")
        
        return None
    
    def _search_from_source2(self, query):
        """从歌词源2搜索（示例实现）"""
        # 这里应该实现实际的歌词搜索逻辑
        return None
    
    def _search_from_source3(self, query):
        """从歌词源3搜索（示例实现）"""
        # 这里应该实现实际的歌词搜索逻辑
        return None
    
    def _search_local_lyrics(self, keywords):
        """从本地歌词库搜索歌词"""
        try:
            # 这里可以实现本地歌词库的搜索逻辑
            # 例如，检查本地存储的歌词文件或数据库
            
            # 示例：检查是否有本地歌词文件
            # 1. 尝试从音频文件同目录查找.lrc文件
            if self.audio_path:
                base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
                dir_path = os.path.dirname(self.audio_path)
                
                # 常见的歌词文件扩展名
                lyrics_extensions = ['.lrc', '.txt', '.lyrics']
                for ext in lyrics_extensions:
                    lyrics_file = os.path.join(dir_path, base_name + ext)
                    if os.path.exists(lyrics_file):
                        with open(lyrics_file, 'r', encoding='utf-8', errors='ignore') as f:
                            lyrics = f.read()
                            if self._validate_lyrics(lyrics):
                                return lyrics
            
            # 2. 尝试从预定义的本地歌词目录搜索
            local_lyrics_dir = os.path.join(os.path.dirname(__file__), "local_lyrics")
            if os.path.exists(local_lyrics_dir):
                for keyword in keywords:
                    # 尝试匹配文件名
                    for file in os.listdir(local_lyrics_dir):
                        if keyword.lower() in file.lower():
                            file_path = os.path.join(local_lyrics_dir, file)
                            if os.path.isfile(file_path):
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    lyrics = f.read()
                                    if self._validate_lyrics(lyrics):
                                        return lyrics
            
            return None
        except Exception as e:
            print(f"本地歌词搜索失败: {e}")
            return None
    
    def _enhance_lyrics_with_recognition(self, accurate_lyrics):
        """使用识别结果增强准确歌词"""
        try:
            # 如果准确歌词没有时间戳，尝试从识别结果中提取
            if not accurate_lyrics or not self.recognized_lyrics:
                return accurate_lyrics
                
            # 检查准确歌词是否包含时间戳
            has_timestamps = bool(re.search(r'\[\d{2}:\d{2}\.\d{2}\]', accurate_lyrics))
            
            if has_timestamps:
                return accurate_lyrics  # 已经有时间戳，不需要增强
                
            # 获取识别结果中的词级时间戳
            original_words = self.recognized_lyrics.get("words", [])
            if not original_words:
                return accurate_lyrics
                
            # 尝试将识别结果中的时间戳应用到准确歌词
            # 这是一个简化的实现，实际应用中可能需要更复杂的算法
            enhanced_lyrics = self._apply_timestamps_to_lyrics(accurate_lyrics, original_words)
            
            return enhanced_lyrics
        except Exception as e:
            print(f"增强歌词失败: {e}")
            return accurate_lyrics
    
    def _apply_timestamps_to_lyrics(self, lyrics, word_timestamps):
        """将时间戳应用到歌词"""
        try:
            # 将歌词分割成行
            lines = lyrics.split('\n')
            
            # 计算每行的平均时间
            total_words = sum(len(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', line)) for line in lines)
            if total_words == 0:
                return lyrics
                
            # 获取总时长
            total_duration = word_timestamps[-1].get("end", 0) if word_timestamps else 0
            if total_duration == 0:
                return lyrics
                
            # 为每行添加时间戳
            timestamped_lines = []
            current_time = 0
            
            for line in lines:
                # 计算当前行的字数
                word_count = len(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', line))
                
                # 计算当前行的时间长度
                line_duration = (word_count / total_words) * total_duration
                
                # 格式化时间戳为 [mm:ss.xx]
                minutes = int(current_time // 60)
                seconds = current_time % 60
                timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
                
                # 添加时间戳
                timestamped_lines.append(f"{timestamp} {line}")
                
                # 更新当前时间
                current_time += line_duration
                
            return '\n'.join(timestamped_lines)
        except Exception as e:
            print(f"应用时间戳失败: {e}")
            return lyrics
    
    def _validate_lyrics(self, lyrics):
        """验证歌词质量"""
        if not lyrics or len(lyrics) < 10:
            return False
            
        # 检查是否包含足够的中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', lyrics)
        if len(chinese_chars) < 20:
            return False
            
        # 检查是否包含常见的歌词结构元素
        if not any(pattern in lyrics for pattern in ['\n', '，', '。']):
            return False
            
        return True
    
    def _align_lyrics_with_timestamps(self, accurate_lyrics):
        """将准确歌词与原始时间戳对齐"""
        try:
            # 首先尝试使用识别结果增强准确歌词
            enhanced_lyrics = self._enhance_lyrics_with_recognition(accurate_lyrics)
            
            # 获取原始识别结果中的词级时间戳
            original_words = self.recognized_lyrics.get("words", [])
            
            if not original_words:
                # 如果没有词级时间戳，尝试使用段落级时间戳
                segments = self.recognized_lyrics.get("segments", [])
                if segments:
                    # 使用段落级时间戳创建词级时间戳
                    original_words = self._create_word_timestamps_from_segments(enhanced_lyrics, segments)
                else:
                    # 没有任何时间戳信息，返回增强后的歌词
                    return self._create_text_only_result(enhanced_lyrics)
            
            # 将准确歌词分割成行和词
            lyrics_lines = enhanced_lyrics.split('\n')
            lyrics_words = []
            
            for line in lyrics_lines:
                # 分割每行为词
                words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', line)
                lyrics_words.extend(words)
            
            # 使用动态时间规整(DTW)或简单的线性映射对齐歌词
            aligned_words = self._align_word_sequences(original_words, lyrics_words)
            
            # 构建优化后的结果
            optimized_result = {
                "text": enhanced_lyrics,
                "original_text": self.recognized_lyrics.get("text", ""),
                "words": aligned_words,
                "optimization_method": "web_search_and_alignment"
            }
            
            return optimized_result
        except Exception as e:
            print(f"歌词对齐失败: {e}")
            return self._create_fallback_result()
    
    def _create_word_timestamps_from_segments(self, lyrics, segments):
        """从段落级时间戳创建词级时间戳"""
        try:
            # 将歌词分割成行
            lines = lyrics.split('\n')
            
            # 如果没有段落信息或行数不匹配，返回空列表
            if not segments or len(lines) == 0:
                return []
            
            # 计算每行的字数
            line_word_counts = []
            for line in lines:
                words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', line)
                line_word_counts.append(len(words))
            
            total_words = sum(line_word_counts)
            if total_words == 0:
                return []
            
            # 创建词级时间戳
            word_timestamps = []
            current_word_idx = 0
            
            for i, (line, word_count) in enumerate(zip(lines, line_word_counts)):
                if word_count == 0:
                    continue
                    
                # 获取当前段落的时间戳
                start_time = segments[i].get("start", 0) if i < len(segments) else 0
                end_time = segments[i].get("end", 0) if i < len(segments) else start_time + 5
                
                # 计算每个词的平均时间
                duration_per_word = (end_time - start_time) / word_count if word_count > 0 else 0
                
                # 为当前行的每个词创建时间戳
                for j in range(word_count):
                    word_start = start_time + j * duration_per_word
                    word_end = word_start + duration_per_word
                    
                    word_timestamps.append({
                        "word": "",  # 这里不存储实际词，只存储时间戳
                        "start": word_start,
                        "end": word_end
                    })
                    
                    current_word_idx += 1
            
            return word_timestamps
        except Exception as e:
            print(f"从段落创建词级时间戳失败: {e}")
            return []
    
    def _create_text_only_result(self, lyrics):
        """创建只有文本的结果（没有时间戳）"""
        return {
            "text": lyrics,
            "original_text": self.recognized_lyrics.get("text", ""),
            "words": [],  # 没有时间戳信息
            "optimization_method": "web_search_text_only"
        }
    
    def _align_word_sequences(self, original_words, target_words):
        """对齐原始词序列和目标词序列"""
        # 这是一个简化的对齐算法，实际应用中可以使用更复杂的算法
        
        # 如果目标词数量与原始词数量相近，使用简单映射
        if abs(len(target_words) - len(original_words)) <= len(original_words) * 0.2:
            # 按比例映射时间戳
            aligned_words = []
            ratio = len(target_words) / len(original_words) if original_words else 1
            
            for i, word_data in enumerate(original_words):
                # 计算对应的目标词索引
                target_idx = int(i * ratio)
                if target_idx < len(target_words):
                    aligned_word = {
                        "word": target_words[target_idx],
                        "start": word_data.get("start", 0),
                        "end": word_data.get("end", 0)
                    }
                    aligned_words.append(aligned_word)
            
            return aligned_words
        
        # 如果词数量差异较大，使用原始结果
        return original_words
    
    def _create_fallback_result(self):
        """创建备用结果（使用原始识别结果）"""
        return {
            "text": self.recognized_lyrics.get("text", ""),
            "words": self.recognized_lyrics.get("words", []),
            "optimization_method": "original_recognition"
        }