"""
音频分类模型定义
"""

import torch
import numpy as np
import json
import os
from config.settings import CLASSIFIER_MODEL_PATH, LABEL_MAPPING_PATH


class LightweightMultiLabelCNNLSTM(torch.nn.Module):
    """与M7训练脚本保持一致的模型定义"""

    def __init__(self, input_dim, num_labels, max_length):
        super().__init__()
        self.max_length = max_length
        self.num_labels = num_labels

        self.conv1 = torch.nn.Conv1d(
            in_channels=input_dim, out_channels=32, kernel_size=5, padding=2
        )
        self.bn1 = torch.nn.BatchNorm1d(32)

        self.depthwise_conv = torch.nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
            groups=32,
        )
        self.pointwise_conv = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.lstm = torch.nn.LSTM(
            input_size=64,
            hidden_size=128,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.lstm_dropout = torch.nn.Dropout(0.3)

        self.shared_fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
        )
        self.label_classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(64, 1) for _ in range(num_labels)]
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.pointwise_conv(self.depthwise_conv(x))))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_dropout(lstm_out)
        shared_features = self.shared_fc(lstm_out)

        outputs = []
        for classifier in self.label_classifiers:
            outputs.append(classifier(shared_features))
        multi_label_output = torch.cat(outputs, dim=-1)
        return multi_label_output


class AudioClassifierModel:
    """音频分类模型封装类"""
    
    def __init__(self, model_path=None, label_mapping_path=None, device=None):
        self.model_path = model_path or CLASSIFIER_MODEL_PATH
        self.label_mapping_path = label_mapping_path or LABEL_MAPPING_PATH
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.label_names = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.no_label_idx = None
        self.max_length = 512
        self.target_dim = 74
        
        self._load_model()
        self._load_label_mapping()
    
    def _load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建模型实例
        input_dim = self.target_dim
        num_labels = len(self.label_names) if self.label_names else 8  # 默认8个标签
        
        self.model = LightweightMultiLabelCNNLSTM(
            input_dim=input_dim,
            num_labels=num_labels,
            max_length=self.max_length
        )
        
        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_label_mapping(self):
        """加载标签映射"""
        if not os.path.exists(self.label_mapping_path):
            # 如果标签映射文件不存在，使用默认标签
            self.label_names = [f"标签_{i}" for i in range(8)]
            self.label_to_idx = {name: i for i, name in enumerate(self.label_names)}
            self.idx_to_label = {i: name for i, name in enumerate(self.label_names)}
            self.no_label_idx = 0  # 假设第一个标签是无标签
            return
        
        try:
            with open(self.label_mapping_path, 'r', encoding='utf-8') as f:
                self.label_to_idx = json.load(f)
            
            self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
            self.label_names = list(self.label_to_idx.keys())
            
            # 查找"无标签"或"无"的索引
            for name, idx in self.label_to_idx.items():
                if "无" in name:
                    self.no_label_idx = idx
                    break
            
            # 如果没有找到无标签，则使用第一个标签
            if self.no_label_idx is None:
                self.no_label_idx = 0
                
        except Exception as e:
            print(f"加载标签映射失败: {e}")
            # 使用默认标签
            self.label_names = [f"标签_{i}" for i in range(8)]
            self.label_to_idx = {name: i for i, name in enumerate(self.label_names)}
            self.idx_to_label = {i: name for i, name in enumerate(self.label_names)}
            self.no_label_idx = 0
    
    def predict(self, mel_sequence):
        """对音频序列进行预测"""
        if self.model is None:
            raise ValueError("模型未加载")
        
        total_frames = mel_sequence.shape[0]
        if total_frames == 0:
            raise ValueError("音频帧数为0，无法进行推理")
        
        probabilities = []
        
        with torch.no_grad():
            for start in range(0, total_frames, self.max_length):
                end = min(start + self.max_length, total_frames)
                chunk = mel_sequence[start:end]
                current_len = len(chunk)
                
                if current_len < self.max_length:
                    pad_len = self.max_length - current_len
                    chunk = np.pad(
                        chunk,
                        pad_width=((0, pad_len), (0, 0)),
                        mode="constant",
                        constant_values=0.0,
                    )
                
                chunk_tensor = (
                    torch.from_numpy(chunk.astype(np.float32))
                    .unsqueeze(0)
                    .to(self.device)
                )
                logits = self.model(chunk_tensor).squeeze(0).cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))
                probabilities.append(probs[:current_len])
        
        prob_matrix = np.vstack(probabilities)
        return prob_matrix