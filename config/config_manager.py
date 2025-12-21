"""
配置管理模块
用于保存和加载应用程序配置
"""

import os
import json
from typing import Dict, Any


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file="config/user_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return {}
        return {}
    
    def save_config(self) -> bool:
        """保存配置"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存配置文件失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        self.config[key] = value
    
    def get_api_key(self, provider: str = "doubao") -> str:
        """获取API密钥"""
        return self.config.get("api_keys", {}).get(provider, "")
    
    def set_api_key(self, provider: str, api_key: str) -> None:
        """设置API密钥"""
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        self.config["api_keys"][provider] = api_key


# 全局配置管理器实例
config_manager = ConfigManager()