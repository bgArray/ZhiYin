"""
知音项目全局配置
"""

# 应用程序信息
APP_NAME = "知音"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI音频听觉功能集成软件"

# 窗口默认设置
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

# 音频处理参数
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_BUFFER_SIZE = 4096

# 模型路径
MODEL_DIR = "resources/models"
CLASSIFIER_MODEL_PATH = "best_new_models/final_multilabel_cnn_lstm_model_m8.pth"
LABEL_MAPPING_PATH = "best_new_models/label_to_idx_m8.json"

# UI主题
THEME = "dark"  # dark/light

# 功能模块
FEATURE_MODULES = {
    "audio_classifier": {
        "name": "音频分类",
        "description": "使用AI模型对音频进行多标签分类",
        "icon": "classifier"
    },
    "source_separation": {
        "name": "音源分离",
        "description": "使用Demucs进行人声和乐器分离",
        "icon": "separator"
    },
    "multi_audio_processor": {
        "name": "多音频处理器",
        "description": "同时处理多个音频文件并进行声乐技术分析",
        "icon": "multi_audio"
    }
}