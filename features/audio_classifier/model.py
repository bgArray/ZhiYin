"""
音频分类模型定义
"""

import torch
import numpy as np
import json
import os
from config.settings import CLASSIFIER_MODEL_PATH, LABEL_MAPPING_PATH

# 导入Keras相关库用于加载和使用M6LSTM_CNN模型
try:
    import keras
    import sys
    sys.path.append('best_new_models/3t')
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


LABEL_MAPPING = {
  "-1": "说话",
  "0": "真声",
  "1": "混声",
  "2": "假声",
  "3": "气声",
  "4": "咽音",
  "5": "颤音",
  "6": "滑音"
}


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


class EnhancedMultiLabelCNNLSTM(torch.nn.Module):
    """
    增强版多标签CNN-LSTM模型
    设计思路：
    1. 增加卷积层数和通道数，提升特征提取能力
    2. 使用双层LSTM，增加时序建模能力
    3. 增加隐藏单元数，提升模型容量
    4. 扩展特征通道，支持多歌手和多语言
    5. 保留共享特征提取头和多标签分类头结构
    """

    def __init__(self, input_dim, num_labels, max_length):
        super(EnhancedMultiLabelCNNLSTM, self).__init__()
        self.max_length = max_length
        self.num_labels = num_labels

        # 共享特征提取头：增强版CNN
        # 第一层：标准卷积
        self.conv1 = torch.nn.Conv1d(
            in_channels=input_dim, out_channels=64, kernel_size=5, padding=2
        )
        self.bn1 = torch.nn.BatchNorm1d(64)

        # 第二层：深度可分离卷积
        self.depthwise_conv = torch.nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            groups=64,  # 深度可分离卷积
        )
        self.pointwise_conv = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm1d(128)

        # 第三层：标准卷积
        self.conv3 = torch.nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn3 = torch.nn.BatchNorm1d(256)

        # 第四层：标准卷积
        self.conv4 = torch.nn.Conv1d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.bn4 = torch.nn.BatchNorm1d(256)

        # LSTM层：双层LSTM，捕捉更复杂的时序特征
        self.lstm = torch.nn.LSTM(
            input_size=256,
            hidden_size=256,  # 隐藏单元数从128增加到256
            num_layers=2,     # 从1层增加到2层
            batch_first=True,
            dropout=0.3,      # 双层LSTM支持dropout
            bidirectional=False,
        )
        # 在LSTM后添加Dropout层
        self.lstm_dropout = torch.nn.Dropout(0.3)

        # 共享全连接层：增加宽度和深度
        self.shared_fc = torch.nn.Sequential(
            torch.nn.Linear(256, 128), 
            torch.nn.ReLU(), 
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 128),  # 增加一层全连接
            torch.nn.ReLU(), 
            torch.nn.Dropout(0.4)
        )

        # 多标签分类头：每个标签独立分类
        self.label_classifiers = torch.nn.ModuleList(
            [torch.nn.Linear(128, 1) for _ in range(num_labels)]
        )

        # 激活函数
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        前向传播函数

        :param x: 输入特征，形状为(batch_size, seq_len, input_dim)
        :return: 模型输出，形状为(batch_size, seq_len, num_labels)，返回logits值
        """
        # 转换维度：(batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # 第一层CNN
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 第二层：深度可分离卷积
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 第三层CNN
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # 第四层CNN
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # 转换回：(batch_size, seq_len, channels)
        x = x.transpose(1, 2)

        # LSTM层
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        # 应用dropout
        lstm_out = self.lstm_dropout(lstm_out)

        # 共享全连接层
        shared_features = self.shared_fc(lstm_out)  # (batch_size, seq_len, 128)

        # 多标签分类
        outputs = []
        for classifier in self.label_classifiers:
            outputs.append(classifier(shared_features))
        multi_label_output = torch.cat(
            outputs, dim=-1
        )  # (batch_size, seq_len, num_labels)
        return multi_label_output


class AudioClassifierModel:
    """音频分类模型封装类"""
    
    def __init__(self, model_path=None, label_mapping_path=None, device=None):
        self.model_path = model_path or CLASSIFIER_MODEL_PATH
        self.label_mapping_path = label_mapping_path or LABEL_MAPPING_PATH
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.keras_model = None  # Keras模型实例，用于M6LSTM_CNN
        self.label_names = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.no_label_idx = None
        self.max_length = 512
        self.target_dim = 74
        
        # 先加载标签映射，确保有正确的标签信息
        self._load_label_mapping()
        
        # 然后加载PyTorch模型，使用正确的标签数量
        self._load_model()
        
        # 尝试加载Keras模型用于发生位置增强判断
        self._load_keras_model()
        
        # 最后创建config属性以兼容现有代码
        self.config = {
            'label_names': self.label_names,
            'max_length': self.max_length,
            'target_dim': self.target_dim
        }
    
    def _load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 创建模型实例
        num_labels = len(self.label_names) if self.label_names else 8  # 默认8个标签
        
        # 根据模型文件路径选择合适的模型类
        if "enhanced" in self.model_path.lower():
            # 使用增强模型
            # 增强模型需要90维输入（74维梅尔频谱 + 16维额外特征）
            self.model = EnhancedMultiLabelCNNLSTM(
                input_dim=90,  # 增强模型需要90维输入
                num_labels=num_labels,
                max_length=self.max_length
            )
        else:
            # 使用标准模型
            input_dim = self.target_dim
            self.model = LightweightMultiLabelCNNLSTM(
                input_dim=input_dim,
                num_labels=num_labels,
                max_length=self.max_length
            )
        
        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 处理可能的参数不匹配问题
        try:
            # 尝试严格加载权重
            self.model.load_state_dict(checkpoint)
        except RuntimeError as e:
            # 如果出现参数不匹配，尝试非严格加载
            print(f"严格加载模型权重失败: {e}")
            print("尝试非严格加载模型权重...")
            
            # 获取模型当前的状态字典
            model_dict = self.model.state_dict()
            
            # 过滤出匹配的参数
            filtered_checkpoint = {}
            for k, v in checkpoint.items():
                if k in model_dict:
                    # 检查形状是否匹配
                    if v.shape == model_dict[k].shape:
                        filtered_checkpoint[k] = v
                    else:
                        # 如果形状不匹配，创建一个新的张量，使用当前模型的形状，填充0
                        print(f"参数 {k} 形状不匹配: {v.shape} != {model_dict[k].shape}，将使用零填充")
                        new_tensor = torch.zeros_like(model_dict[k])
                        # 尽可能复制可用的数据
                        min_dims = tuple(min(a, b) for a, b in zip(v.shape, model_dict[k].shape))
                        indices = tuple(slice(0, d) for d in min_dims)
                        new_tensor[indices] = v[indices]
                        filtered_checkpoint[k] = new_tensor
                else:
                    print(f"忽略未知参数: {k}")
            
            # 加载过滤后的权重
            model_dict.update(filtered_checkpoint)
            self.model.load_state_dict(model_dict)
        
        self.model.to(self.device)
        self.model.eval()
    
    def _load_label_mapping(self):
        """加载标签映射"""
        global LABEL_MAPPING
        if not os.path.exists(self.label_mapping_path):
            # 如果标签映射文件不存在，使用默认标签
            self.label_names = ["说话", "真声", "混声", "假声", "气声", "咽音", "颤音", "滑音"]
            self.label_to_idx = {name: i for i, name in enumerate(self.label_names)}
            self.idx_to_label = {i: name for i, name in enumerate(self.label_names)}
            self.no_label_idx = None  # 没有无标签
            return
        
        try:
            with open(self.label_mapping_path, 'r', encoding='utf-8') as f:
                self.label_to_idx = json.load(f)
            
            # 过滤掉-1的标签（说话），因为它不是真正的声乐技术标签
            filtered_labels = {k: v for k, v in self.label_to_idx.items() if v >= 0}
            
            # 按索引排序
            sorted_labels = sorted(filtered_labels.items(), key=lambda x: x[1])
            self.label_names = [name for name, idx in sorted_labels]
            self.idx_to_label = {idx: name for name, idx in sorted_labels}
            # print(self.label_names)

            # 标签翻译
            ln = self.label_names
            self.label_names = []
            for i in ln:
                self.label_names.append(LABEL_MAPPING.get(i))
            # print(self.label_names)
            del ln
            
            # 没有无标签
            self.no_label_idx = None
                
        except Exception as e:
            print(f"加载标签映射失败: {e}")
            # 使用默认标签
            self.label_names = ["说话", "真声", "混声", "假声", "气声", "咽音", "颤音", "滑音"]
            self.label_to_idx = {name: i for i, name in enumerate(self.label_names)}
            self.idx_to_label = {i: name for i, name in enumerate(self.label_names)}
            self.no_label_idx = None
    
    def _load_keras_model(self):
        """加载Keras模型用于发生位置增强判断"""
        if not KERAS_AVAILABLE:
            print("Keras不可用，无法加载M6LSTM_CNN模型")
            self.keras_model = None
            return
        
        try:
            # Keras模型路径
            keras_model_path = "best_new_models/3t/best_cnn_lstm_model.keras"
            if not os.path.exists(keras_model_path):
                print(f"Keras模型文件不存在: {keras_model_path}")
                self.keras_model = None
                return
            
            # 加载Keras模型
            print(f"正在加载Keras模型: {keras_model_path}")
            self.keras_model = keras.models.load_model(keras_model_path)
            print("Keras模型加载成功")
        except Exception as e:
            print(f"加载Keras模型失败: {e}")
            self.keras_model = None
    
    def predict(self, mel_sequence, use_pitch_position_judgment=False):
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
                
                # 检查是否需要添加额外通道（针对增强模型）
                # 增强模型需要：74维梅尔频谱 + 14维时域特征 + 1维F0特征 + 1维性别标识 = 90维
                if isinstance(self.model, EnhancedMultiLabelCNNLSTM):
                    # 为增强模型添加额外通道
                    # 原始mel_sequence是74维，需要添加16维零填充
                    seq_len = chunk.shape[0]
                    additional_channels = np.zeros((seq_len, 16), dtype=np.float32)
                    chunk = np.concatenate([chunk, additional_channels], axis=1)
                
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
                # 使用更稳定的方式计算sigmoid，避免溢出
                logits = np.clip(logits, -20, 20)  # 限制logits范围
                probs = 1.0 / (1.0 + np.exp(-logits))
                probabilities.append(probs[:current_len])
        
        prob_matrix = np.vstack(probabilities)
        
        # 如果需要使用发生位置增强判断，且Keras模型可用
        if use_pitch_position_judgment and self.keras_model is not None:
            print("使用发生位置增强判断")
            # 使用Keras模型进行预测
            keras_probs = self._predict_with_keras(mel_sequence)
            
            # 组合结果：对于标签0,1,2（真声、混声、假声），使用0.7权重的Keras模型结果
            # 注意：Keras模型输出的是类别概率，需要调整为与主模型相同的格式
            if keras_probs.shape[0] == prob_matrix.shape[0]:
                # 确保时间步数量匹配
                # Keras模型输出格式：(time_steps, num_classes)，num_classes=4（包括-1）
                # 主模型输出格式：(time_steps, num_labels)，num_labels=8
                
                # 仅对标签0,1,2（对应索引1,2,3，因为主模型的索引0是说话）进行加权组合
                # 主模型标签：0-说话，1-真声，2-混声，3-假声，4-气声，5-咽音，6-颤音，7-滑音
                # Keras模型标签：0-无标签(-1)，1-真声(0)，2-混声(1)，3-假声(2)
                for i in range(prob_matrix.shape[0]):
                    # 真声（主模型索引1，Keras索引1）
                    prob_matrix[i, 1] = prob_matrix[i, 1] * 0.3 + keras_probs[i, 1] * 0.7
                    # 混声（主模型索引2，Keras索引2）
                    prob_matrix[i, 2] = prob_matrix[i, 2] * 0.3 + keras_probs[i, 2] * 0.7
                    # 假声（主模型索引3，Keras索引3）
                    prob_matrix[i, 3] = prob_matrix[i, 3] * 0.3 + keras_probs[i, 3] * 0.7
        
        return prob_matrix
    
    def _predict_with_keras(self, mel_sequence):
        """使用Keras模型进行预测"""
        if self.keras_model is None:
            raise ValueError("Keras模型未加载")
        
        # 准备Keras模型输入
        # Keras模型需要：
        # 1. 频域切片：只使用前74维梅尔频谱
        # 2. 序列填充到固定长度
        
        # 频域切片
        mel_sliced = mel_sequence[:, :74]
        
        # 确保输入形状正确
        max_length = self.max_length
        if mel_sliced.shape[0] < max_length:
            # 填充到max_length
            pad_len = max_length - mel_sliced.shape[0]
            mel_padded = np.pad(
                mel_sliced,
                pad_width=((0, pad_len), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
        else:
            # 截取到max_length
            mel_padded = mel_sliced[:max_length]
        
        # 添加batch维度
        mel_padded = np.expand_dims(mel_padded, axis=0)
        
        # 使用Keras模型预测
        keras_logits = self.keras_model.predict(mel_padded, verbose=0)
        
        # 获取概率分布（softmax输出）
        keras_probs = keras_logits[0]  # 移除batch维度
        
        # 确保输出长度与输入匹配
        if mel_sliced.shape[0] < max_length:
            keras_probs = keras_probs[:mel_sliced.shape[0]]
        
        return keras_probs