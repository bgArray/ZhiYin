import os
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Conv1D, BatchNormalization  # MaxPooling1D, AveragePooling1D,
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers, losses
# from keras import *
from tensorflow import reduce_sum, maximum  # , expand_dims, squeeze


# -------------------------- 1. 数据加载函数（保留原有逻辑，新增时序增强可选） --------------------------
def load_data(file_path):
    """加载.npy数据，提取0/1/2标签并确保特征与标签长度一致"""
    data = []
    for filename in os.listdir(file_path):
        if filename.endswith('.npy'):
            file_full_path = os.path.join(file_path, filename)
            data.extend(np.load(file_full_path, allow_pickle=True))

    X = []  # 存储mel特征序列（每个元素是一个样本的序列）
    y = []  # 存储标签序列（每个元素是一个样本的标签序列）
    for item in data:
        mel_spectrogram, tags = item
        # 确保mel和标签的时间步数量一致
        if len(mel_spectrogram) != len(tags):
            print(f"警告：mel长度({len(mel_spectrogram)})与标签长度({len(tags)})不匹配，已跳过该样本")
            continue

        # 处理每个时间步的标签：只保留0/1/2，无有效标签标记为-1
        valid_tags = []
        for tag_group in tags:
            filtered = [t for t in tag_group if t in {"0", "1", "2"}]
            valid_tags.append(filtered[0] if filtered else "-1")

        X.append(mel_spectrogram)
        y.append(valid_tags)

    return X, y


# -------------------------- 2. 数据增强函数（整合轻量级频谱增强） --------------------------
def augment_mel_spectrogram(mel_seq):
    """对单条梅尔频谱序列进行轻量级增强：频域偏移+幅度缩放+高斯噪声"""
    aug_seq = mel_seq.copy()
    np.random.seed(42)  # 保证增强可复现

    # 2.1 微小频域偏移（±2个梅尔系数，避免跨关键频带）
    shift = np.random.randint(-2, 3)
    if shift != 0:
        aug_seq = np.roll(aug_seq, shift, axis=1)  # 沿频域轴滚动
        # 偏移后空白处填充0
        if shift > 0:
            aug_seq[:, :shift] = 0.0
        else:
            aug_seq[:, shift:] = 0.0

    # 2.2 随机幅度缩放（0.8-1.2倍，模拟演唱力度变化）
    scale = np.random.uniform(0.8, 1.2)
    aug_seq = aug_seq * scale

    # 2.3 叠加高斯噪声（强度为原序列标准差的5%，避免掩盖特征）
    noise = np.random.normal(0, aug_seq.std() * 0.05, aug_seq.shape)
    aug_seq = aug_seq + noise

    return aug_seq


def augment_temporal(mel_seq, tag_seq):
    """时序裁剪增强（可选）：随机截取序列80%-90%长度，保持帧-标签对应"""
    seq_len = len(mel_seq)
    if seq_len < 10:  # 避免过短序列裁剪后无效
        return mel_seq, tag_seq
    # 截取长度与起始位置
    crop_len = np.random.randint(int(seq_len * 0.8), int(seq_len * 0.9) + 1)
    start_idx = np.random.randint(0, seq_len - crop_len + 1)
    return mel_seq[start_idx:start_idx + crop_len], tag_seq[start_idx:start_idx + crop_len]


# -------------------------- 3. 数据预处理函数（整合增强与标签处理） --------------------------
def preprocess_data(X, y, le=None, max_length=None, is_train=True):
    """
    数据预处理：频域切片+增强（训练集）+序列填充+标签编码
    :param X: 原始mel特征序列列表
    :param y: 原始标签序列列表
    :param le: 标签编码器（训练集为None，测试集传入训练集的le）
    :param max_length: 序列最大长度（训练集为None，测试集传入训练集的max_length）
    :param is_train: 是否为训练集（控制是否启用增强）
    :return: 预处理后的特征、独热编码标签、最大长度、标签编码器
    """
    # 3.1 梅尔频谱频域切片：保留200Hz-5kHz关键频带（原128维→74维）
    X_sliced = []
    for mel_seq in X:
        sliced = mel_seq[:, 0:74]  # 需根据实际梅尔滤波器组参数调整，确保覆盖目标频带
        X_sliced.append(sliced)
    X = X_sliced

    # 3.2 训练集数据增强（频谱增强+可选时序增强）
    if is_train:
        X_augmented = []
        y_augmented = []
        for mel_seq, tag_seq in zip(X, y):
            # 原始样本保留
            X_augmented.append(mel_seq)
            y_augmented.append(tag_seq)
            # 频谱增强样本
            aug_mel_spect = augment_mel_spectrogram(mel_seq)
            X_augmented.append(aug_mel_spect)
            y_augmented.append(tag_seq.copy())  # 标签与原样本一致
            # 可选：时序增强样本（进一步提升鲁棒性）
            aug_mel_temp, aug_tag_temp = augment_temporal(mel_seq, tag_seq)
            X_augmented.append(aug_mel_temp)
            y_augmented.append(aug_tag_temp)
        # 更新为增强后的数据集
        X, y = X_augmented, y_augmented

    # 3.3 标签编码（训练集拟合编码器，测试集复用）
    if is_train:
        # 扁平化所有标签用于拟合编码器
        all_tags = [tag for seq in y for tag in seq]
        le = LabelEncoder()
        le.fit(all_tags)
    # 将标签序列编码为数字
    y_encoded = [le.transform(seq) for seq in y]

    # 3.4 序列统一长度（训练集计算最大长度，测试集复用）
    if is_train:
        max_length = max(len(seq) for seq in X)
    # 特征序列填充（后填充0）
    X_padded = sequence.pad_sequences(
        X, maxlen=max_length, dtype='float32', padding='post', truncating='post', value=0.0
    )
    # 标签序列填充（后填充-1的编码）
    no_label_code = le.transform(["-1"])[0]
    y_padded = sequence.pad_sequences(
        y_encoded, maxlen=max_length, padding='post', truncating='post', value=no_label_code
    )

    # 3.5 标签独热编码（适配多分类）
    num_classes = len(le.classes_)
    y_onehot = to_categorical(y_padded, num_classes=num_classes)

    return X_padded, y_onehot, max_length, le


# # -------------------------- 4. 自定义损失函数（忽略无标签帧） --------------------------
# def ignore_no_label_loss(y_true, y_pred, le):
#     """自定义损失：忽略标签为-1的帧，仅优化有效帧（0/1/2）"""
#     y_true_downsampled = AveragePooling1D(pool_size=2, strides=2)(
#         expand_dims(y_true, axis=-1)  # 增加通道维度
#     )
#     y_true_downsampled = squeeze(y_true_downsampled, axis=-1)  # 移除通道维度
#     # 获取-1对应的独热编码索引
#     no_label_idx = np.where(le.classes_ == "-1")[0][0]
#     # 生成掩码：非-1帧为1，-1帧为0
#     mask = 1 - y_true_downsampled[:, :, no_label_idx]
#     # 计算交叉熵损失并应用掩码
#     ce_loss = losses.categorical_crossentropy(y_true_downsampled, y_pred)
#     masked_loss = ce_loss * mask
#     # 用有效帧数量归一化（避免除以0）
#     return reduce_sum(masked_loss) / maximum(reduce_sum(mask), 1e-6)
# def ignore_no_label_loss(y_true, y_pred, le):
#     """自定义损失：忽略标签为-1的帧，仅优化有效帧（0/1/2）"""
#     # 获取-1对应的独热编码索引
#     no_label_idx = np.where(le.classes_ == "-1")[0][0]
#     # 生成掩码：非-1帧为1，-1帧为0
#     mask = 1 - y_true[:, :, no_label_idx]
#
#     # 解决时间步不匹配问题：对标签进行下采样（与模型输出时间步一致）
#     # 注意：y_true是3维张量(batch_size, timesteps, num_classes)
#     # 直接在时间步维度进行平均池化（无需增加通道维度）
#     y_true_downsampled = AveragePooling1D(pool_size=2, strides=2)(y_true)
#     # 同时对掩码进行下采样（保持与标签形状一致）
#     mask_downsampled = AveragePooling1D(pool_size=2, strides=2)(expand_dims(mask, -1))
#     mask_downsampled = squeeze(mask_downsampled, -1)  # 恢复为2维
#
#     # 计算交叉熵损失并应用掩码
#     ce_loss = losses.categorical_crossentropy(y_true_downsampled, y_pred)
#     masked_loss = ce_loss * mask_downsampled
#     # 用有效帧数量归一化（避免除以0）
#     return reduce_sum(masked_loss) / maximum(reduce_sum(mask_downsampled), 1e-6)
# def ignore_no_label_loss(y_true, y_pred, le):
#     """自定义损失：忽略标签为-1的帧，仅优化有效帧（0/1/2）"""
#     # 获取-1对应的独热编码索引
#     no_label_idx = np.where(le.classes_ == "-1")[0][0]
#     # 生成掩码：非-1帧为1，-1帧为0（保持原始时间步）
#     mask = 1 - y_true[:, :, no_label_idx]
#
#     # 计算交叉熵损失并应用掩码（无需下采样，直接使用原始时间步）
#     ce_loss = losses.categorical_crossentropy(y_true, y_pred)
#     masked_loss = ce_loss * mask
#     # 用有效帧数量归一化（避免除以0）
#     return reduce_sum(masked_loss) / maximum(reduce_sum(mask), 1e-6)
#
#
# # -------------------------- 5. CNN-LSTM串联模型构建（整合单帧与时序特征） --------------------------
# # def build_cnn_lstm_model(input_shape, num_classes, le):
# #     """
# #     构建CNN-LSTM串联模型：1D-CNN提取单帧频域特征 + LSTM捕捉时序关联
# #     :param input_shape: 输入形状 (时间步, 频域维度)
# #     :param num_classes: 分类类别数（4类：-1/0/1/2）
# #     :return: 编译后的模型
# #     """
# #     model = Sequential([
# #         # 5.1 1D-CNN层：提取单帧频域特征（32个卷积核，核大小5）
# #         Conv1D(
# #             filters=32,
# #             kernel_size=5,
# #             activation='relu',
# #             input_shape=input_shape,
# #             padding='same'  # 保持时间步长度不变
# #         ),
# #         # 5.2 1D池化层：压缩频域维度，降低计算量（步长2→维度减半）
# #         MaxPooling1D(pool_size=2, padding='same'),
# #         # 5.3 Masking层：过滤填充的0值，避免干扰训练
# #         Masking(mask_value=0.0),
# #         # 5.4 LSTM层：捕捉帧间时序关联（128单元，逐帧输出）
# #         LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
# #         # 5.5 全连接层：整合特征（轻量化设计，32单元）
# #         Dense(32, activation='relu'),
# #         Dropout(0.3),  # 防止过拟合
# #         # 5.6 输出层：逐帧分类（softmax激活，输出各类别概率）
# #         Dense(num_classes, activation='softmax')
# #     ])
# #
# #     # 编译模型：Adam优化器+自定义损失
# #     model.compile(
# #         optimizer=optimizers.Adam(learning_rate=0.001),
# #         loss=lambda y_true, y_pred: ignore_no_label_loss(y_true, y_pred, le),  # 传入标签编码器
# #         metrics=['accuracy']
# #     )
# #     return model
# def build_cnn_lstm_model(input_shape, num_classes, le):
#     model = Sequential([
#         # 1D-CNN层：保持时间步（padding='same'）
#         Conv1D(
#             filters=32,
#             kernel_size=5,
#             activation='relu',
#             input_shape=input_shape,
#             padding='same'  # 关键：保持时间步不变
#         ),
#         # 移除MaxPooling1D（避免时间步减半），改用BatchNormalization稳定训练
#         # MaxPooling1D(pool_size=2, padding='same'),  <-- 删除这行
#         BatchNormalization(),  # 新增：替代池化的正则化作用
#
#         Masking(mask_value=0.0),
#         LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
#         Dense(32, activation='relu'),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')
#     ])
#
#     model.compile(
#         optimizer=optimizers.Adam(learning_rate=0.001),
#         loss=lambda y_true, y_pred: ignore_no_label_loss(y_true, y_pred, le),
#         metrics=['accuracy']  # 现在时间步一致，准确率计算正常
#     )
#     return model

# # -------------------------- 4. 自定义损失函数（修改后） --------------------------
# def ignore_no_label_loss(no_label_idx):
#     """使用闭包传入no_label_idx，避免直接引用LabelEncoder"""
#
#     def loss(y_true, y_pred):
#         # 生成掩码：非-1帧为1，-1帧为0
#         mask = 1 - y_true[:, :, no_label_idx]
#
#         # 计算交叉熵损失并应用掩码
#         ce_loss = losses.categorical_crossentropy(y_true, y_pred)
#         masked_loss = ce_loss * mask
#         # 用有效帧数量归一化（避免除以0）
#         return reduce_sum(masked_loss) / maximum(reduce_sum(mask), 1e-6)
#
#     return loss
def ignore_no_label_loss(no_label_idx):
    """使用闭包传入无标签索引，避免序列化问题且保持时间步一致"""
    def loss(y_true, y_pred):
        # 生成掩码：非-1帧为1，-1帧为0（与y_true同形状）
        mask = 1 - y_true[:, :, no_label_idx]
        # 直接计算原始时间步的交叉熵（无需下采样）
        ce_loss = losses.categorical_crossentropy(y_true, y_pred)
        masked_loss = ce_loss * mask
        # 用有效帧数量归一化
        return reduce_sum(masked_loss) / maximum(reduce_sum(mask), 1e-6)
    return loss


# # -------------------------- 5. 模型构建（修改损失函数部分） --------------------------
# def build_cnn_lstm_model(input_shape, num_classes, no_label_idx):  # 接收no_label_idx而非le
#     model = Sequential([
#         Conv1D(
#             filters=32,
#             kernel_size=5,
#             activation='relu',
#             input_shape=input_shape,
#             padding='same'
#         ),
#         BatchNormalization(),
#         Masking(mask_value=0.0),
#         LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
#         Dense(32, activation='relu'),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')
#     ])
#
#     model.compile(
#         optimizer=optimizers.Adam(learning_rate=0.001),
#         loss=ignore_no_label_loss(no_label_idx),  # 传入预计算的索引
#         metrics=['accuracy']
#     )
#     return model
def build_cnn_lstm_model(input_shape, num_classes, no_label_idx):
    model = Sequential([
        # 第一层CNN：提取局部频域特征
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        # 新增：深度卷积层（不改变时间步）
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),

        Masking(mask_value=0.0),
        # LSTM层：增加单元数提升时序建模能力
        LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
        # 新增：注意力机制（聚焦关键帧）
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=ignore_no_label_loss(no_label_idx),  # 传入无标签索引
        metrics=['accuracy']
    )
    return model

# -------------------------- 6. 主函数（整合数据流程与模型训练） --------------------------
def main(data_path):
    # 6.1 加载原始数据
    print("开始加载数据...")
    X_raw, y_raw = load_data(data_path)
    print(f"数据加载完成，原始样本数: {len(X_raw)}")

    # 6.2 划分训练集与测试集（8:2）
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"训练集样本数: {len(X_train_raw)}, 测试集样本数: {len(X_test_raw)}")

    # 6.3 数据预处理（训练集单独预处理，测试集复用训练集参数）
    print("开始预处理训练集...")
    X_train, y_train, max_length, le = preprocess_data(
        X_train_raw, y_train_raw, is_train=True
    )
    print("开始预处理测试集...")
    X_test, y_test, _, _ = preprocess_data(
        X_test_raw, y_test_raw, le=le, max_length=max_length, is_train=False
    )
    print(f"预处理完成：训练集形状{X_train.shape}, 测试集形状{X_test.shape}, 类别数{y_train.shape[-1]}")

    # 6.4 定义模型输入形状与构建模型
    input_shape = (max_length, X_train.shape[2])
    num_classes = y_train.shape[-1]
    # 预计算-1对应的索引（代替直接传递le）
    no_label_idx = np.where(le.classes_ == "-1")[0][0]  # 关键修改
    model = build_cnn_lstm_model(input_shape, num_classes, no_label_idx)  # 传入索引
    model.summary()

    # 6.5 定义训练回调函数（早停+模型保存+学习率调度）
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            'best_cnn_lstm_model.keras',  # 改用Keras原生格式，替代 legacy HDF5
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        )
    ]

    # 6.6 模型训练（CPU适配：batch_size=16，避免内存溢出）
    print("开始模型训练...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,  # CPU推荐16，若内存充足可尝试32
        validation_split=0.1,  # 训练集再划分10%为验证集
        shuffle=True,
        callbacks=callbacks,
        verbose=1  # 简化输出，减少CPU IO负担
    )

    # 6.7 模型评估（测试集）
    print("开始模型评估...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"测试集最终准确率: {accuracy:.4f}, 测试集损失: {loss:.4f}")

    # 6.8 保存最终模型与标签编码器
    model.save('final_cnn_lstm_model.keras')
    np.save('label_encoder.npy', le.classes_)
    print("最终模型（final_cnn_lstm_model.keras）与标签编码器（label_encoder.npy）保存完成")


# -------------------------- 7. 程序入口 --------------------------
if __name__ == "__main__":
    # 从Constant.py导入数据集路径（确保该文件存在且路径正确）
    try:
        from Constant import DATASET_PATH
    except ImportError:
        # 若Constant.py不存在，可直接指定路径（示例："D:/vocal_dataset"）
        DATASET_PATH = input("请输入数据集文件夹路径（如D:/vocal_dataset）：")

    # 检查路径有效性
    if not os.path.exists(DATASET_PATH):
        print(f"错误：数据集路径{DATASET_PATH}不存在，请检查路径是否正确！")
    else:
        main(DATASET_PATH)
