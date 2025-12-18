"""
模型工具函数
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


def load_model_checkpoint(model_path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        model_path: 模型文件路径
        device: 目标设备
        
    Returns:
        dict: 模型检查点
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"加载模型检查点失败: {e}")


def save_model_checkpoint(
    model: torch.nn.Module, 
    optimizer: Optional[torch.optim.Optimizer], 
    epoch: int, 
    loss: float, 
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        save_path: 保存路径
        additional_info: 额外信息
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 创建检查点字典
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        # 保存检查点
        torch.save(checkpoint, save_path)
    except Exception as e:
        raise RuntimeError(f"保存模型检查点失败: {e}")


def load_label_mapping(label_mapping_path: str) -> Tuple[Dict[str, int], List[str], int]:
    """
    加载标签映射
    
    Args:
        label_mapping_path: 标签映射文件路径
        
    Returns:
        tuple: (标签到索引的映射, 标签名称列表, 无标签索引)
    """
    if not os.path.exists(label_mapping_path):
        # 如果标签映射文件不存在，使用默认标签
        label_names = [f"标签_{i}" for i in range(8)]
        label_to_idx = {name: i for i, name in enumerate(label_names)}
        no_label_idx = 0  # 假设第一个标签是无标签
        return label_to_idx, label_names, no_label_idx
    
    try:
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            label_to_idx = json.load(f)
        
        idx_to_label = {v: k for k, v in label_to_idx.items()}
        label_names = list(label_to_idx.keys())
        
        # 查找"无标签"或"无"的索引
        no_label_idx = None
        for name, idx in label_to_idx.items():
            if "无" in name:
                no_label_idx = idx
                break
        
        # 如果没有找到无标签，则使用第一个标签
        if no_label_idx is None:
            no_label_idx = 0
            
        return label_to_idx, label_names, no_label_idx
    except Exception as e:
        raise RuntimeError(f"加载标签映射失败: {e}")


def save_label_mapping(label_to_idx: Dict[str, int], save_path: str) -> None:
    """
    保存标签映射
    
    Args:
        label_to_idx: 标签到索引的映射
        save_path: 保存路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存标签映射
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"保存标签映射失败: {e}")


def get_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]) -> str:
    """
    获取模型摘要信息
    
    Args:
        model: 模型
        input_size: 输入尺寸
        
    Returns:
        str: 模型摘要
    """
    try:
        # 创建随机输入
        dummy_input = torch.randn(1, *input_size)
        
        # 前向传播
        output = model(dummy_input)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 构建摘要
        summary = f"模型摘要:\n"
        summary += f"输入尺寸: {input_size}\n"
        summary += f"输出尺寸: {list(output.shape)}\n"
        summary += f"总参数数量: {total_params:,}\n"
        summary += f"可训练参数数量: {trainable_params:,}\n"
        
        return summary
    except Exception as e:
        raise RuntimeError(f"获取模型摘要失败: {e}")


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    计算模型参数数量
    
    Args:
        model: 模型
        
    Returns:
        tuple: (总参数数量, 可训练参数数量)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def initialize_model_weights(model: torch.nn.Module, init_type: str = "xavier_uniform") -> None:
    """
    初始化模型权重
    
    Args:
        model: 模型
        init_type: 初始化类型
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(param)
            elif init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(param)
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif init_type == "uniform":
                torch.nn.init.uniform_(param, -0.1, 0.1)
            elif init_type == "normal":
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)


def get_device() -> torch.device:
    """
    获取可用设备
    
    Returns:
        torch.device: 可用设备
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用CUDA设备: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    
    return device


def move_model_to_device(model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    将模型移动到指定设备
    
    Args:
        model: 模型
        device: 目标设备
        
    Returns:
        torch.nn.Module: 移动后的模型
    """
    if device is None:
        device = get_device()
    
    return model.to(device)