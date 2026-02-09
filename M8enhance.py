import os

# import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import copy


# -------------------------- 1. 多标签数据加载函数 --------------------------
def load_multilabel_data(file_path, max_samples=None):
    """
    加载.npy数据，支持多标签格式
    每个时间步可以有多个标签（如同时有真声和颤音）

    参数：
        file_path: 数据文件路径或目录
        max_samples: 最大加载样本数，None表示加载全部
    """
    data = []

    # 如果file_path是文件，直接加载
    if os.path.isfile(file_path) and file_path.endswith(".npy"):
        try:
            loaded_data = np.load(file_path, allow_pickle=True)
            data.extend(loaded_data)
        except Exception as e:
            print(f"警告：加载文件 {file_path} 时出错: {e}")
            return [], []
    else:
        # 如果是目录，加载所有npy文件
        for filename in os.listdir(file_path):
            if filename.endswith(".npy") and filename.startswith("multilabel_"):
                file_full_path = os.path.join(file_path, filename)
                try:
                    loaded_data = np.load(file_full_path, allow_pickle=True)
                    data.extend(loaded_data)
                    print(f"加载文件: {filename}, 样本数: {len(loaded_data)}")
                except Exception as e:
                    print(f"警告：加载文件 {filename} 时出错: {e}")
                    continue

    # 限制样本数量
    if max_samples is not None and len(data) > max_samples:
        print(f"限制样本数从 {len(data)} 到 {max_samples}")
        data = data[:max_samples]

    X = []  # 存储mel特征序列
    y = []  # 存储多标签序列（每个元素是一个标签列表，可能包含多个标签）

    for item in data:
        if len(item) != 2:
            print(f"警告：数据格式不正确，跳过该样本")
            continue

        mel_spectrogram, tags = item
        # 确保mel和标签的时间步数量一致
        if len(mel_spectrogram) != len(tags):
            print(
                f"警告：mel长度({len(mel_spectrogram)})与标签长度({len(tags)})不匹配，已跳过该样本"
            )
            continue

        # 处理每个时间步的标签：保留所有有效标签
        # 定义有效标签集合（数字标签和文本标签）
        valid_label_set = {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "-1",
            "Breathy",
            "Vibrato",
            "Glissando",
            "Pharyngeal",
            "Chest",
            "Mixed",
            "Falsetto",
            "Head",
            "Normal",
        }

        valid_tags = []
        for tag_group in tags:
            if isinstance(tag_group, list):
                # 如果已经是列表，过滤有效标签
                filtered = []
                for t in tag_group:
                    tag_str = str(t).strip()
                    # 检查是否是有效的单个标签
                    if tag_str in valid_label_set:
                        filtered.append(tag_str)
                    elif len(tag_str) > 1 and tag_str not in ["", "None", "nan"]:
                        # 可能是多个标签连在一起（如"04"应该是"0,4"）
                        # 尝试按字符分割：如果标签长度>1且不在有效集合中，尝试按字符分割
                        for char in tag_str:
                            if char in valid_label_set:
                                filtered.append(char)
                if not filtered:
                    filtered = ["-1"]  # 无有效标签
            elif isinstance(tag_group, str):
                # 如果是字符串，先按逗号分割
                tag_str = tag_group.strip()
                if tag_str in ["", "None", "nan"]:
                    filtered = ["-1"]
                else:
                    # 先按逗号分割
                    parts = [t.strip() for t in tag_str.split(",") if t.strip()]
                    filtered = []
                    for part in parts:
                        if part in valid_label_set:
                            filtered.append(part)
                        elif len(part) > 1:
                            # 可能是多个标签连在一起，尝试按字符分割
                            for char in part:
                                if char in valid_label_set:
                                    filtered.append(char)
                    if not filtered:
                        filtered = ["-1"]
            else:
                # 其他情况，转换为字符串
                tag_str = str(tag_group).strip()
                if tag_str in ["", "None", "nan"]:
                    filtered = ["-1"]
                else:
                    # 先按逗号分割
                    parts = [t.strip() for t in tag_str.split(",") if t.strip()]
                    filtered = []
                    for part in parts:
                        if part in valid_label_set:
                            filtered.append(part)
                        elif len(part) > 1:
                            # 可能是多个标签连在一起，尝试按字符分割
                            for char in part:
                                if char in valid_label_set:
                                    filtered.append(char)
                    if not filtered:
                        filtered = ["-1"]

            # 去重并保持顺序
            seen = set()
            unique_filtered = []
            for tag in filtered:
                if tag not in seen:
                    seen.add(tag)
                    unique_filtered.append(tag)
            valid_tags.append(unique_filtered)

        X.append(mel_spectrogram)
        y.append(valid_tags)

    return X, y


# -------------------------- 2. 数据增强函数（与原版相同） --------------------------
def augment_mel_spectrogram(mel_seq):
    """对单条梅尔频谱序列进行轻量级增强：频域偏移+幅度缩放+高斯噪声"""
    aug_seq = mel_seq.copy()
    np.random.seed(42)

    # 微小频域偏移（±2个梅尔系数）
    shift = np.random.randint(-2, 3)
    if shift != 0:
        aug_seq = np.roll(aug_seq, shift, axis=1)
        if shift > 0:
            aug_seq[:, :shift] = 0.0
        else:
            aug_seq[:, shift:] = 0.0

    # 随机幅度缩放（0.8-1.2倍）
    scale = np.random.uniform(0.8, 1.2)
    aug_seq = aug_seq * scale

    # 叠加高斯噪声
    noise = np.random.normal(0, aug_seq.std() * 0.05, aug_seq.shape)
    aug_seq = aug_seq + noise

    return aug_seq


def augment_temporal(mel_seq, tag_seq):
    """时序裁剪增强：随机截取序列80%-90%长度"""
    seq_len = len(mel_seq)
    if seq_len < 10:
        return mel_seq, tag_seq
    crop_len = np.random.randint(int(seq_len * 0.8), int(seq_len * 0.9) + 1)
    start_idx = np.random.randint(0, seq_len - crop_len + 1)
    return (
        mel_seq[start_idx : start_idx + crop_len],
        tag_seq[start_idx : start_idx + crop_len],
    )


# -------------------------- 3. 多标签数据预处理函数 --------------------------
def preprocess_multilabel_data(X, y, label_to_idx=None, max_length=None, is_train=True):
    """
    多标签数据预处理：频域切片+增强+序列填充+多标签编码+特征融合
    :param X: 原始mel特征序列列表
    :param y: 原始多标签序列列表（每个元素是标签列表）
    :param label_to_idx: 标签到索引的映射（训练集为None，测试集传入训练集的映射）
    :param max_length: 序列最大长度
    :param is_train: 是否为训练集
    :return: 预处理后的特征、多标签编码、最大长度、标签映射
    """
    # 3.1 梅尔频谱频域切片：保留200Hz-5kHz关键频带（原128维→74维）
    X_sliced = []
    for mel_seq in X:
        if mel_seq.shape[1] >= 74:
            sliced = mel_seq[:, 0:74]
        else:
            # 如果维度不足，进行填充
            pad_width = 74 - mel_seq.shape[1]
            sliced = np.pad(
                mel_seq, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0
            )
        X_sliced.append(sliced)
    X = X_sliced
    
    # 3.1.1 特征融合：梅尔频谱 + 声学特征 + 性别标识
    X_fused = []
    for i, mel_seq in enumerate(X):
        seq_len = len(mel_seq)
        
        # 模拟生成14维时域特征（过零率、信号行程距离、峰值数等）
        temporal_features = np.random.randn(seq_len, 14)  # 实际应用中应从音频信号提取
        
        # 模拟生成1维F0特征
        f0_features = np.random.randn(seq_len, 1)  # 实际应用中应从音频信号提取
        
        # 生成性别标识特征（0默认女性，1男性）
        gender = np.random.randint(0, 2)  # 实际应用中应从数据中获取
        gender_features = np.full((seq_len, 1), gender, dtype=np.float32)
        
        # 特征融合：梅尔频谱(74) + 时域特征(14) + F0(1) + 性别标识(1) = 90维
        fused_features = np.concatenate([mel_seq, temporal_features, f0_features, gender_features], axis=1)
        X_fused.append(fused_features)
    X = X_fused

    # 3.2 训练集数据增强
    if is_train:
        X_augmented = []
        y_augmented = []
        for mel_seq, tag_seq in zip(X, y):
            # 原始样本
            X_augmented.append(mel_seq)
            y_augmented.append(tag_seq)
            # 频谱增强样本 - 只增强梅尔频谱部分
            aug_mel_spect = augment_mel_spectrogram(mel_seq[:, :74])
            # 重新生成声学特征（简化处理）
            seq_len = len(aug_mel_spect)
            temporal_features = np.random.randn(seq_len, 14)
            f0_features = np.random.randn(seq_len, 1)
            gender = np.random.randint(0, 2)
            gender_features = np.full((seq_len, 1), gender, dtype=np.float32)
            aug_fused = np.concatenate([aug_mel_spect, temporal_features, f0_features, gender_features], axis=1)
            X_augmented.append(aug_fused)
            y_augmented.append([tags.copy() for tags in tag_seq])  # 深拷贝标签
            # 时序增强样本
            aug_mel_temp, aug_tag_temp = augment_temporal(mel_seq, tag_seq)
            X_augmented.append(aug_mel_temp)
            y_augmented.append(aug_tag_temp)
        X, y = X_augmented, y_augmented

    # 3.3 构建标签到索引的映射（训练集构建，测试集复用）
    if is_train:
        # 收集所有出现的标签
        all_labels = set()
        for tag_seq in y:
            for tag_list in tag_seq:
                all_labels.update(tag_list)

        # 排序标签以确保一致性（-1放在最后）
        sorted_labels = sorted([l for l in all_labels if l != "-1"]) + ["-1"]
        label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        num_labels = len(label_to_idx)
    else:
        num_labels = len(label_to_idx)

    # 3.4 序列统一长度（限制最大长度以避免内存溢出）
    if is_train:
        max_length = max(len(seq) for seq in X)
        # 限制最大长度为512，避免内存溢出
        max_length = min(max_length, 512)
        print(f"序列最大长度: {max_length}")

    # 特征序列填充（分批处理以避免内存溢出）
    # 使用更小的批次大小，避免内存溢出
    batch_size = 500  # 每批处理500个样本

    # 预先分配数组（使用更小的数据类型或分批保存）
    X_padded_list = []
    y_multihot_list = []

    for batch_start in range(0, len(X), batch_size):
        batch_end = min(batch_start + batch_size, len(X))
        batch_X = X[batch_start:batch_end]
        batch_y = y[batch_start:batch_end]

        # 处理特征
        batch_padded = []
        for seq in batch_X:
            if len(seq) < max_length:
                pad_width = max_length - len(seq)
                padded = np.pad(
                    seq, ((0, pad_width), (0, 0)), mode="constant", constant_values=0.0
                )
            else:
                padded = seq[:max_length]
            batch_padded.append(padded.astype("float32"))

        # 处理标签
        batch_multihot = np.zeros(
            (len(batch_y), max_length, num_labels), dtype="float32"
        )
        for i, tag_seq in enumerate(batch_y):
            for j, tag_list in enumerate(tag_seq):
                if j >= max_length:
                    break
                # 为每个标签设置1
                for tag in tag_list:
                    if tag in label_to_idx:
                        batch_multihot[i, j, label_to_idx[tag]] = 1.0

        # 保存批次数据
        X_padded_list.append(np.array(batch_padded))
        y_multihot_list.append(batch_multihot)

        if is_train:
            print(f"  处理进度: {batch_end}/{len(X)}")

    # 合并所有批次（如果数据量不大，可以合并；否则可以分批保存）
    if len(X_padded_list) == 1:
        X_padded = X_padded_list[0]
        y_multihot = y_multihot_list[0]
    else:
        # 分批合并，避免一次性创建大数组
        print("  合并批次数据...")
        X_padded = np.concatenate(X_padded_list, axis=0)
        y_multihot = np.concatenate(y_multihot_list, axis=0)

    return X_padded, y_multihot, max_length, label_to_idx


# -------------------------- 4. PyTorch Dataset类 --------------------------
class MultiLabelVocalDataset(Dataset):
    """PyTorch Dataset类，用于加载多标签数据"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------- 5. 多标签损失函数（忽略无标签帧） --------------------------
class MultiLabelIgnoreNoLabelLoss(nn.Module):
    """多标签损失：忽略标签为-1的帧，仅优化有效帧"""

    def __init__(self, no_label_idx):
        super(MultiLabelIgnoreNoLabelLoss, self).__init__()
        self.no_label_idx = no_label_idx
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y_pred, y_true):
        """
        :param y_pred: (batch_size, seq_len, num_labels) 模型预测输出（logits）
        :param y_true: (batch_size, seq_len, num_labels) 多标签编码
        :return: 掩码后的BCE损失
        """
        # 生成掩码：非-1帧为1，-1帧为0
        # 如果-1标签存在，则检查该位置是否为1
        mask = 1 - y_true[:, :, self.no_label_idx]  # (batch_size, seq_len)

        # 计算BCE损失（逐帧逐标签）
        batch_size, seq_len, num_labels = y_pred.shape
        y_pred_flat = y_pred.view(-1, num_labels)
        y_true_flat = y_true.view(-1, num_labels)
        mask_flat = mask.view(-1, 1)  # (batch_size * seq_len, 1)

        # 计算BCE损失
        bce_loss = self.bce_loss(
            y_pred_flat, y_true_flat
        )  # (batch_size * seq_len, num_labels)

        # 应用掩码（只对有效帧计算损失）
        masked_loss = bce_loss * mask_flat

        # 用有效帧数量归一化
        valid_frames = torch.sum(mask_flat)
        if valid_frames > 0:
            return torch.sum(masked_loss) / valid_frames
        else:
            return torch.tensor(0.0, device=y_pred.device)


# -------------------------- 6. 增强版多标签CNN-LSTM模型 --------------------------
class EnhancedMultiLabelCNNLSTM(nn.Module):
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
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=64, kernel_size=5, padding=2
        )
        self.bn1 = nn.BatchNorm1d(64)

        # 第二层：深度可分离卷积
        self.depthwise_conv = nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            groups=64,  # 深度可分离卷积
        )
        self.pointwise_conv = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)

        # 第三层：标准卷积
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm1d(256)

        # 第四层：标准卷积
        self.conv4 = nn.Conv1d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.bn4 = nn.BatchNorm1d(256)

        # LSTM层：双层LSTM，捕捉更复杂的时序特征
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,  # 隐藏单元数从128增加到256
            num_layers=2,     # 从1层增加到2层
            batch_first=True,
            dropout=0.3,      # 双层LSTM支持dropout
            bidirectional=False,
        )
        # 在LSTM后添加Dropout层
        self.lstm_dropout = nn.Dropout(0.3)

        # 共享全连接层：增加宽度和深度
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, 128),  # 增加一层全连接
            nn.ReLU(), 
            nn.Dropout(0.4)
        )

        # 多标签分类头：每个标签独立分类
        self.label_classifiers = nn.ModuleList(
            [nn.Linear(128, 1) for _ in range(num_labels)]
        )

        # 激活函数
        self.relu = nn.ReLU()

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


# -------------------------- 7. 训练和评估函数 --------------------------
def train_epoch(model, dataloader, criterion, optimizer, device, no_label_idx):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_labels = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 前向传播
        optimizer.zero_grad()
        y_pred = model(X_batch)

        # 计算损失
        loss = criterion(y_pred, y_batch)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()

        # 计算准确率（多标签准确率：所有标签都预测正确才算正确）
        with torch.no_grad():
            # 应用sigmoid得到概率
            y_pred_proba = torch.sigmoid(y_pred)
            y_pred_binary = (y_pred_proba > 0.5).float()

            # 生成掩码（排除无标签帧）
            mask = (1 - y_batch[:, :, no_label_idx]).unsqueeze(
                -1
            )  # (batch_size, seq_len, 1)

            # 计算每个标签的准确率
            correct = (y_pred_binary == y_batch).float() * mask
            total_correct += correct.sum().item()
            total_labels += mask.sum().item() * y_batch.shape[-1]

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_labels if total_labels > 0 else 0.0

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, no_label_idx):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_labels = 0

    # 用于计算每个标签的精确率、召回率、F1
    label_stats = {}

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 前向传播
            y_pred = model(X_batch)

            # 计算损失
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

            # 计算准确率
            y_pred_proba = torch.sigmoid(y_pred)
            y_pred_binary = (y_pred_proba > 0.5).float()

            # 生成掩码
            mask = (1 - y_batch[:, :, no_label_idx]).unsqueeze(-1)

            # 计算准确率
            correct = (y_pred_binary == y_batch).float() * mask
            total_correct += correct.sum().item()
            total_labels += mask.sum().item() * y_batch.shape[-1]

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_labels if total_labels > 0 else 0.0

    return avg_loss, accuracy


def balance_labels(X, y, target_labels=None):
    """
    实现标签均衡采样，确保所有目标标签都能被包含在数据中

    :param X: 原始特征数据列表
    :param y: 原始标签数据列表
    :param target_labels: 目标标签集合，确保这些标签都能被包含
    :return: 均衡后的特征数据和标签数据
    """
    if target_labels is None:
        target_labels = {"0", "1", "2", "3", "4", "5", "6"}  # 默认目标标签为数字标签0-6

    # 1. 首先统计每个样本包含的目标标签
    sample_labels = []
    for idx, tag_seq in enumerate(y):
        sample_tag_set = set()
        for tag_list in tag_seq:
            sample_tag_set.update(tag_list)
        # 只保留目标标签
        sample_target_tags = sample_tag_set.intersection(target_labels)
        sample_labels.append(sample_target_tags)

    # 2. 统计每个目标标签出现的样本索引
    label_to_samples = defaultdict(list)
    for idx, tags in enumerate(sample_labels):
        for tag in tags:
            label_to_samples[tag].append(idx)

    # 3. 检查是否所有目标标签都存在
    missing_labels = [label for label in target_labels if not label_to_samples[label]]
    if missing_labels:
        print(f"警告：以下目标标签在数据中不存在: {missing_labels}")

    # 4. 确定每个标签需要的样本数量（取最大样本数）
    if label_to_samples:
        max_samples_per_label = max(
            len(samples) for samples in label_to_samples.values()
        )
    else:
        max_samples_per_label = 0

    # 5. 对每个标签进行过采样或欠采样，确保样本均衡
    selected_indices = set()

    for label in target_labels:
        if label not in label_to_samples or not label_to_samples[label]:
            continue

        samples = label_to_samples[label]
        num_samples = len(samples)

        if num_samples >= max_samples_per_label:
            # 欠采样：随机选择max_samples_per_label个样本
            selected = np.random.choice(samples, max_samples_per_label, replace=False)
        else:
            # 过采样：重复采样直到达到max_samples_per_label个样本
            selected = np.random.choice(samples, max_samples_per_label, replace=True)

        selected_indices.update(selected)

    # 6. 如果没有选择到任何样本（可能所有标签都不存在），则使用原始数据
    if not selected_indices:
        print("警告：没有选择到任何样本，将使用原始数据")
        return X, y

    # 7. 转换为列表并排序
    selected_indices = sorted(list(selected_indices))

    # 8. 返回均衡后的数据
    balanced_X = [X[idx] for idx in selected_indices]
    balanced_y = [y[idx] for idx in selected_indices]

    print(f"标签均衡完成：原始样本数 {len(X)}，均衡后样本数 {len(balanced_X)}")

    # 打印均衡后的标签统计
    all_labels = []
    for tag_seq in balanced_y:
        for tag_list in tag_seq:
            all_labels.extend(tag_list)
    label_counts = Counter(all_labels)
    print(f"均衡后标签统计:")
    for label in sorted(target_labels):
        print(f"  {label}: {label_counts.get(label, 0)} 次")

    return balanced_X, balanced_y


# -------------------------- 8. 主函数 --------------------------
def filter_state_dict(state_dict):
    """
    过滤模型state_dict，移除注意力层的权重

    用途：由于当前版本模型定义中保留了注意力层结构，但forward方法未使用，
    因此在保存模型时需要过滤掉注意力层的权重，以避免模型加载时出现键不匹配错误

    :param state_dict: 原始模型的state_dict，包含所有层的权重
    :return: 过滤后的state_dict，不包含注意力层(attention.)的权重
    """
    filtered_dict = {}
    for key, value in state_dict.items():
        # 移除所有以'attention.'开头的权重键
        if not key.startswith("attention."):
            filtered_dict[key] = value
    return filtered_dict


def main(data_path):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 8.1 加载原始数据
    print("开始加载数据...")
    # 增加样本数量限制，确保能包含更多标签
    MAX_SAMPLES = None  # 加载所有数据，确保能找到所有标签
    X_raw, y_raw = load_multilabel_data(data_path, max_samples=MAX_SAMPLES)
    print(f"数据加载完成，原始样本数: {len(X_raw)}")

    # 打印一些标签统计信息
    all_labels = []
    for tag_seq in y_raw:
        for tag_list in tag_seq:
            all_labels.extend(tag_list)
    label_counts = Counter(all_labels)
    print(f"\n原始数据标签统计:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} 次")

    # 8.2 实现标签均衡，确保所有数字标签(0-6)都能被包含
    print("\n开始实现标签均衡...")
    target_labels = {"0", "1", "2", "3", "4", "5", "6"}  # 确保这些数字标签都能被包含
    X_balanced, y_balanced = balance_labels(X_raw, y_raw, target_labels)

    # 8.3 划分训练集与测试集（8:2）
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"\n训练集样本数: {len(X_train_raw)}, 测试集样本数: {len(X_test_raw)}")

    # 8.4 数据预处理
    print("\n开始预处理训练集...")
    X_train, y_train, max_length, label_to_idx = preprocess_multilabel_data(
        X_train_raw, y_train_raw, is_train=True
    )
    print("开始预处理测试集...")
    X_test, y_test, _, _ = preprocess_multilabel_data(
        X_test_raw,
        y_test_raw,
        label_to_idx=label_to_idx,
        max_length=max_length,
        is_train=False,
    )
    print(
        f"预处理完成：训练集形状{X_train.shape}, 测试集形状{X_test.shape}, 标签数{len(label_to_idx)}"
    )

    # 打印标签映射
    print(f"\n标签映射:")
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    for idx in sorted(idx_to_label.keys()):
        print(f"  {idx}: {idx_to_label[idx]}")

    # 8.5 创建Dataset和DataLoader
    train_dataset = MultiLabelVocalDataset(X_train, y_train)
    test_dataset = MultiLabelVocalDataset(X_test, y_test)

    # 从训练集中划分验证集（10%）
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 8.6 定义模型
    input_dim = X_train.shape[2]  # 74
    num_labels = len(label_to_idx)
    model = EnhancedMultiLabelCNNLSTM(input_dim, num_labels, max_length).to(device)

    # 打印模型结构
    print("\n模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    # 8.7 定义损失函数、优化器和学习率调度器
    no_label_idx = label_to_idx.get("-1", -1)
    if no_label_idx == -1:
        print("警告：未找到-1标签，将使用所有帧计算损失")
        no_label_idx = num_labels  # 设置为不存在的索引，掩码将全为1

    criterion = MultiLabelIgnoreNoLabelLoss(no_label_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 移除verbose参数以兼容旧版本PyTorch
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=True
        )
    except TypeError:
        # 如果版本不支持verbose参数，则不使用
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )

    # 8.8 训练模型
    print("\n开始模型训练...")
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(50):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, no_label_idx
        )

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, no_label_idx)

        # 学习率调度
        scheduler.step(val_loss)

        # 打印进度
        print(
            f"Epoch {epoch+1}/50 - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            patience_counter = 0

            # 保存过滤后的模型状态（去除注意力层权重）
            best_model_state = copy.deepcopy(model.state_dict())
            filtered_best_state = filter_state_dict(best_model_state)
            torch.save(filtered_best_state, "best_multilabel_cnn_lstm_model_m8_enhanced.pth")

            print(
                f"  -> 保存最佳模型 (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})"
            )
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发，在第 {epoch+1} 轮停止训练")
            print(
                f"最佳验证损失: {best_val_loss:.4f}, 最佳验证准确率: {best_val_accuracy:.4f}"
            )
            break

    # 8.9 加载最佳模型并评估测试集
    # 注意：best_model_state包含注意力层权重，但模型定义中也包含注意力层，因此可以直接加载
    # 这些权重在forward方法中未使用，但不会影响评估结果
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n开始模型评估...")
    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device, no_label_idx
    )
    print(f"测试集最终准确率: {test_accuracy:.4f}, 测试集损失: {test_loss:.4f}")

    # 8.10 保存最终模型与标签映射
    # 使用filter_state_dict过滤掉attention层的权重，避免加载错误
    filtered_state_dict = filter_state_dict(model.state_dict())
    torch.save(filtered_state_dict, "final_multilabel_cnn_lstm_model_m8_enhanced.pth")
    # 保存标签映射（使用numpy保存字典）
    np.save("label_to_idx_m8_enhanced.npy", np.array([label_to_idx], dtype=object))
    # 同时保存为JSON格式以便读取
    import json

    with open("label_to_idx_m8_enhanced.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
    print(
        "最终模型（final_multilabel_cnn_lstm_model_m8_enhanced.pth）与标签映射（label_to_idx_m8_enhanced.npy, label_to_idx_m8_enhanced.json）保存完成"
    )


# -------------------------- 9. 程序入口 --------------------------
if __name__ == "__main__":
    # 从Constant.py导入数据集路径
    try:
        from Constant import DATASET_PATH
    except ImportError:
        DATASET_PATH = input("请输入数据集文件夹路径（如./dataset/）：")

    # 检查路径有效性
    if not os.path.exists(DATASET_PATH):
        print(f"错误：数据集路径{DATASET_PATH}不存在，请检查路径是否正确！")
    else:
        main(DATASET_PATH)
