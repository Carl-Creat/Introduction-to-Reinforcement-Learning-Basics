"""
通用工具函数库

包含强化学习中常用的工具函数:
- 训练曲线可视化
- 模型保存/加载
- 经验回放缓冲区
- 环境包装器
- 日志记录
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from collections import deque
import random


# ==================== 经验回放 ====================

class ReplayBuffer:
    """
    标准经验回放缓冲区
    
    用于 off-policy 算法 (DQN, DDQN, SAC 等)
    存储 (s, a, r, s', done) 元组
    
    Args:
        capacity: 缓冲区最大容量
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size


class PrioritizedReplayBuffer:
    """
    优先经验回放 (Prioritized Experience Replay, PER)
    
    论文: Prioritized Experience Replay (Schaul et al., 2016)
    https://arxiv.org/abs/1511.05952
    
    高 TD 误差的样本被更频繁地采样
    
    Args:
        capacity: 缓冲区容量
        alpha: 优先级指数 (0=均匀采样, 1=完全优先)
        beta: 重要性采样指数
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        self.priorities[indices] = priorities + 1e-6

    def __len__(self):
        return self.size


# ==================== 模型保存/加载 ====================

def save_model(model: nn.Module, path: str, metadata: Optional[Dict] = None):
    """
    保存模型权重和元数据
    
    Args:
        model: PyTorch 模型
        path: 保存路径 (.pth)
        metadata: 额外信息 (episode, reward 等)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
        'timestamp': time.time()
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str) -> Dict:
    """
    加载模型权重
    
    Args:
        model: PyTorch 模型（结构需与保存时一致）
        path: 模型路径
    
    Returns:
        metadata 字典
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return checkpoint.get('metadata', {})


# ==================== 训练日志 ====================

class TrainingLogger:
    """
    训练过程日志记录器
    
    记录每个 episode 的奖励、损失等指标，
    支持保存为 JSON 和生成可视化图表。
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history: Dict[str, List] = {}

    def log(self, **kwargs):
        """记录一步的指标"""
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))

    def get_moving_average(self, key: str, window: int = 100) -> List[float]:
        """计算滑动平均"""
        values = self.history.get(key, [])
        if len(values) < window:
            return values
        return [np.mean(values[max(0, i-window):i+1]) for i in range(len(values))]

    def save(self, filename: str = "training_log.json"):
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Log saved to {path}")

    def plot(self, key: str = "reward", window: int = 100, save_path: Optional[str] = None):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return

        values = self.history.get(key, [])
        ma = self.get_moving_average(key, window)

        plt.figure(figsize=(10, 5))
        plt.plot(values, alpha=0.3, color='blue', label=key)
        plt.plot(ma, color='blue', linewidth=2, label=f'{key} (MA-{window})')
        plt.xlabel('Episode')
        plt.ylabel(key.capitalize())
        plt.title(f'Training Curve - {key}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        plt.close()


# ==================== 环境工具 ====================

def set_seed(seed: int = 42):
    """设置全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """获取可用设备 (GPU/CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


class EpsilonScheduler:
    """
    Epsilon 衰减调度器
    
    支持线性衰减和指数衰减两种模式
    """

    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.01,
        decay_steps: int = 10000,
        mode: str = "exponential"
    ):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.mode = mode
        self.step = 0

    def get_epsilon(self) -> float:
        if self.mode == "linear":
            epsilon = self.start - (self.start - self.end) * min(1.0, self.step / self.decay_steps)
        else:  # exponential
            decay_rate = (self.end / self.start) ** (1.0 / self.decay_steps)
            epsilon = max(self.end, self.start * (decay_rate ** self.step))
        self.step += 1
        return epsilon
