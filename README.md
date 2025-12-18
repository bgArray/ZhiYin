# 知音 - AI音频听觉功能集成软件

知音是一款集成多种AI音频处理功能的软件，提供音频分类、音源分离等多种工具。

## 功能特点

- **音频分类**: 使用深度学习模型对音频进行多标签分类
- **音源分离**: 使用Demucs模型进行人声和乐器分离
- **直观界面**: 基于PySide6的现代化深色主题界面
- **模块化设计**: 清晰的代码结构，易于扩展和维护

## 安装说明

### 环境要求

- Python 3.8+
- CUDA支持的GPU（推荐，用于加速模型推理）

### 安装步骤

1. 克隆仓库
```bash
git clone <repository-url>
cd ZhiYin
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明

### 启动应用

```bash
python main.py
```

### 音频分类功能

1. 点击"音频分类"功能卡片
2. 点击"打开音频文件"选择要分析的音频
3. 等待处理完成，查看分类结果
4. 可以调整阈值来过滤低置信度的分类结果
5. 支持音频播放和波形可视化

### 音源分离功能

1. 点击"音源分离"功能卡片
2. 选择要分离的音频文件
3. 选择合适的模型（推荐使用mdx_extra_q）
4. 设置输出目录
5. 点击"开始分离"
6. 等待处理完成，查看分离结果

## 项目结构

```
ZhiYin/
├── main.py                     # 应用程序入口
├── config/
│   ├── __init__.py
│   └── settings.py             # 全局配置
├── core/
│   ├── __init__.py
│   ├── app.py                  # 主应用程序类
│   └── base_window.py          # 基础窗口类
├── ui/
│   ├── __init__.py
│   ├── main_window.py          # 主界面
│   ├── components/
│   │   ├── __init__.py
│   │   └── navigation.py       # 导航组件
│   └── styles/
│       ├── __init__.py
│       └── dark_theme.py       # 深色主题样式
├── features/
│   ├── __init__.py
│   ├── audio_classifier/
│   │   ├── __init__.py
│   │   ├── classifier_window.py
│   │   ├── waveform_widget.py
│   │   ├── model.py
│   │   └── audio_processor.py
│   └── source_separation/
│       ├── __init__.py
│       ├── demucs_window.py
│       └── demucs_processor.py
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py          # 音频处理工具
│   └── model_utils.py          # 模型工具
└── resources/
    ├── icons/                  # 图标资源
    └── models/                 # 预训练模型
```

## 开发说明

### 添加新功能

1. 在`features`目录下创建新的功能模块
2. 实现功能窗口类，继承自`BaseWindow`
3. 在`config/settings.py`中添加功能信息
4. 在`ui/main_window.py`中添加功能窗口的导入和创建逻辑

### 自定义主题

修改`ui/styles/dark_theme.py`文件中的样式定义，或创建新的主题文件。

## 许可证

[许可证信息]

## 贡献

欢迎提交问题和拉取请求。

## 更新日志

### v1.0.0
- 初始版本
- 实现音频分类功能
- 实现音源分离功能
- 实现主界面和导航