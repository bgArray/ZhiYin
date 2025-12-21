"""
歌词与声乐技术标签对齐器
将歌词时间戳与声乐技术标签结果进行对齐
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any


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


class LyricsTagAligner:
    """歌词与声乐技术标签对齐器"""
    
    def __init__(self, prob_threshold=0.3):
        """
        初始化对齐器
        
        Args:
            prob_threshold: 概率阈值，只保留大于此阈值的标签
        """
        self.prob_threshold = prob_threshold
    
    def align_lyrics_with_tags(
        self, 
        lyrics_data: Dict[str, Any], 
        prob_matrix: np.ndarray, 
        time_stamps: np.ndarray,
        label_names: List[str]
    ) -> Dict[str, Any]:
        """
        将歌词时间戳与声乐技术标签进行对齐
        
        Args:
            lyrics_data: 歌词识别结果
            prob_matrix: 概率矩阵，形状为 (time_steps, num_labels)
            time_stamps: 时间戳数组
            label_names: 标签名称列表
            
        Returns:
            对齐结果，包含每个汉字/词对应的时间戳和声乐技术标签
        """
        aligned_result = {
            "text": lyrics_data.get("text", ""),
            "aligned_words": []
        }
        
        # 确保标签名称和概率矩阵的维度匹配
        if len(label_names) != prob_matrix.shape[1]:
            print(f"警告: 标签数量({len(label_names)})与概率矩阵列数({prob_matrix.shape[1]})不匹配")
            # 如果标签数量不匹配，可能需要调整
            # 这里我们假设概率矩阵的列对应于标签索引0-6（真声到滑音）
            # 而label_names可能包含"说话"标签，其索引为-1
            
            # 创建一个映射，将UI标签索引映射到概率矩阵列索引
            ui_to_prob_idx = {}
            prob_idx = 0
            for i, label_name in enumerate(label_names):
                if label_name == "说话":
                    ui_to_prob_idx[i] = -1  # 说话标签在概率矩阵中没有对应列
                else:
                    ui_to_prob_idx[i] = prob_idx
                    prob_idx += 1
            
            # 如果调整后仍然不匹配，使用截断或填充
            if prob_idx != prob_matrix.shape[1]:
                print(f"调整后仍然不匹配，使用截断或填充")
                min_labels = min(len(label_names), prob_matrix.shape[1])
                label_names = label_names[:min_labels]
        
        # 获取每个时间步的最高概率标签
        max_prob_indices = np.argmax(prob_matrix, axis=1)
        max_prob_values = np.max(prob_matrix, axis=1)
        
        # 为每个歌词字/词找到对应的时间段和标签
        for word_data in lyrics_data.get("words", []):
            word_start = word_data.get("start", 0)
            word_end = word_data.get("end", 0)
            word_text = word_data.get("word", "")
            word_confidence = word_data.get("confidence", 0.0)
            
            # 找到与当前字/词时间重叠的时间步
            overlapping_indices = self._find_overlapping_time_steps(
                word_start, word_end, time_stamps
            )
            
            # 收集这些时间步的标签信息
            tag_info = self._collect_tag_info(
                overlapping_indices, max_prob_indices, max_prob_values, label_names
            )
            
            # 创建对齐结果
            aligned_word = {
                "word": word_text,
                "start": word_start,
                "end": word_end,
                "lyrics_confidence": word_confidence,
                "vocal_tags": tag_info
            }
            
            aligned_result["aligned_words"].append(aligned_word)
        
        return aligned_result
    
    def _find_overlapping_time_steps(
        self, 
        word_start: float, 
        word_end: float, 
        time_stamps: np.ndarray
    ) -> List[int]:
        """
        找到与当前字/词时间重叠的时间步索引
        
        Args:
            word_start: 字/词开始时间
            word_end: 字/词结束时间
            time_stamps: 时间戳数组
            
        Returns:
            重叠的时间步索引列表
        """
        overlapping_indices = []
        
        for i, timestamp in enumerate(time_stamps):
            # 检查时间戳是否在字/词的时间范围内
            # 使用稍微宽松的条件，确保不错过相关时间步
            if timestamp <= word_end and timestamp >= word_start:
                overlapping_indices.append(i)
            # 如果时间戳稍微超出字/词范围，但仍在合理范围内，也包含进来
            elif timestamp < word_start and (i+1 < len(time_stamps) and time_stamps[i+1] > word_start):
                overlapping_indices.append(i)
            elif timestamp > word_end and (i > 0 and time_stamps[i-1] < word_end):
                overlapping_indices.append(i)
        
        # 如果没有找到重叠的时间步，找到最接近的时间步
        if not overlapping_indices and len(time_stamps) > 0:
            closest_idx = np.argmin(np.abs(time_stamps - (word_start + word_end) / 2))
            overlapping_indices.append(closest_idx)
        
        return overlapping_indices
    
    def _collect_tag_info(
        self, 
        indices: List[int], 
        max_prob_indices: np.ndarray, 
        max_prob_values: np.ndarray,
        label_names: List[str]
    ) -> List[Dict[str, Any]]:
        """
        收集指定时间步的标签信息
        
        Args:
            indices: 时间步索引列表
            max_prob_indices: 每个时间步最高概率标签的索引
            max_prob_values: 每个时间步最高概率值
            label_names: 标签名称列表
            
        Returns:
            标签信息列表
        """
        tag_counts = {}
        
        # 创建UI标签索引到概率矩阵列索引的映射（如果需要）
        ui_to_prob_idx = None
        if len(label_names) > 7:  # 如果标签数量大于7，可能包含"说话"标签
            ui_to_prob_idx = {}
            prob_idx = 0
            for i, label_name in enumerate(label_names):
                if label_name == "说话":
                    ui_to_prob_idx[i] = -1  # 说话标签在概率矩阵中没有对应列
                else:
                    ui_to_prob_idx[i] = prob_idx
                    prob_idx += 1
        
        # 统计每个标签出现的次数和概率总和
        for idx in indices:
            if idx < len(max_prob_indices):
                label_idx = max_prob_indices[idx]
                prob_value = max_prob_values[idx]
                
                if prob_value >= self.prob_threshold:  # 只考虑概率大于阈值的标签
                    # 处理标签索引映射
                    if ui_to_prob_idx:
                        # 如果有映射关系，需要找到UI标签索引
                        ui_label_idx = None
                        for ui_idx, prob_idx in ui_to_prob_idx.items():
                            if prob_idx == label_idx:
                                ui_label_idx = ui_idx
                                break
                        
                        if ui_label_idx is not None and ui_label_idx < len(label_names):
                            label_name = label_names[ui_label_idx]
                        else:
                            label_name = f"Label_{label_idx}"
                    else:
                        # 没有映射关系，直接使用标签索引
                        label_name = label_names[label_idx] if label_idx < len(label_names) else f"Label_{label_idx}"
                    
                    if label_name not in tag_counts:
                        tag_counts[label_name] = {
                            "count": 0,
                            "total_prob": 0.0,
                            "max_prob": 0.0
                        }
                    
                    tag_counts[label_name]["count"] += 1
                    tag_counts[label_name]["total_prob"] += prob_value
                    tag_counts[label_name]["max_prob"] = max(tag_counts[label_name]["max_prob"], prob_value)
        
        # 转换为标签信息列表
        tag_info = []
        for label_name, data in tag_counts.items():
            avg_prob = data["total_prob"] / data["count"]
            tag_info.append({
                "tag": label_name,
                "count": data["count"],
                "average_probability": avg_prob,
                "max_probability": data["max_prob"]
            })
        
        # 按平均概率降序排序
        tag_info.sort(key=lambda x: x["average_probability"], reverse=True)
        
        return tag_info
    
    def save_aligned_result(self, aligned_result: Dict[str, Any], output_path: str) -> bool:
        """
        保存对齐结果到JSON文件
        
        Args:
            aligned_result: 对齐结果
            output_path: 输出文件路径
            
        Returns:
            是否成功保存
        """
        try:
            # 转换numpy类型为Python原生类型
            serializable_result = convert_numpy_types(aligned_result)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存对齐结果失败: {str(e)}")
            return False