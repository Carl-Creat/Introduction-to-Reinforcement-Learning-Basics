"""
A3C (Asynchronous Advantage Actor-Critic) 算法实现
============================================

A3C 是由 DeepMind 提出的异步 Advantage Actor-Critic 算法。
核心思想：
- 使用多个 worker 异步并行收集经验
- 每个 worker 独立与环境交互并计算梯度
- 主网络定期从 workers 同步参数

特点：
- 异步更新，提高数据效率
- Advantage Actor-Critic 减少方差
- 适合离散和连续动作空间

Author: Carl
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
import os
import time
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 神经网络定义
# ============================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic 网络
    
    包含两个输出头：
    - Policy head (Actor): 输出动作概率分布
    - Value head (Critic): 估计状态值函数
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        初始化 Actor-Critic 网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
        """
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor 策略头 - 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 价值头 - 估计状态值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            action_probs: 动作概率分布
            value: 状态价值估计
        """
        features = self.feature(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        根据当前策略选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率（用于策略梯度）
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, value = self.forward(state)
        
        # 从概率分布中采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()


class SharedAdam(optim.Adam):
    """
    共享 Adam 优化器
    用于 A3C 中多个进程共享同一个优化器
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(params, lr, betas, eps, 
                                         weight_decay, amsgrad)
        # 共享动量状态
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)


# ============================================================
# Worker 进程
# ============================================================

class A3CWorker:
    """
    A3C Worker 类
    
    每个 worker 独立与环境交互，收集经验，计算梯度
    """
    
    def __init__(self, worker_id: int, global_model: ActorCritic,
                 optimizer: SharedAdam, env_name: str = "CartPole-v1",
                 max_episodes: int = 500, gamma: float = 0.99,
                 entropy_coef: float = 0.01, value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5, n_steps: int = 5):
        """
        初始化 Worker
        
        Args:
            worker_id: Worker 编号
            global_model: 全局网络
            optimizer: 全局优化器
            env_name: 环境名称
            max_episodes: 最大训练回合数
            gamma: 折扣因子
            entropy_coef: 熵系数（鼓励探索）
            value_loss_coef: 价值损失系数
            max_grad_norm: 梯度裁剪范数
            n_steps: 每次更新前收集的步数
        """
        self.worker_id = worker_id
        self.global_model = global_model
        self.optimizer = optimizer
        self.env_name = env_name
        self.max_episodes = max_episodes
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        
        # 创建本地网络（全局网络的副本）
        self.local_model = ActorCritic(
            state_dim=4,  # CartPole-v1
            action_dim=2,
            hidden_dim=256
        )
        
        # 创建环境
        self.env = gym.make(env_name)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        
    def sync_global(self):
        """从全局网络同步参数到本地网络"""
        self.local_model.load_state_dict(self.global_model.state_dict())
    
    def compute_loss(self, states, actions, rewards, dones, next_states) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算策略损失和价值损失
        
        Args:
            states: 状态序列
            actions: 动作序列
            rewards: 奖励序列
            dones: 终止标志序列
            next_states: 下一状态序列
            
        Returns:
            policy_loss: 策略梯度损失
            value_loss: 价值损失
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # 获取当前策略的概率和价值
        action_probs, values = self.local_model(states)
        
        # 计算 log 概率
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # 计算熵（鼓励探索）
        entropy = dist.entropy().mean()
        
        # 计算 GAE (Generalized Advantage Estimation)
        # 简化的 Advantage: A = R + gamma * V(s') - V(s)
        with torch.no_grad():
            _, next_values = self.local_model(next_states[-1].unsqueeze(0))
            next_values = next_values.squeeze()
        
        # 计算优势估计
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages)
        
        # 策略损失 (Policy Loss) - 使用策略梯度
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # 价值损失 (Value Loss)
        value_loss = F.mse_loss(values.squeeze(), 
                                 rewards + self.gamma * next_values * (1 - dones[-1]))
        
        # 总损失
        # 策略损失 + 价值损失 * 系数 - 熵 * 系数（最大化熵）
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        return total_loss, policy_loss, value_loss, entropy
    
    def train(self):
        """训练循环"""
        self.sync_global()
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        while len(self.episode_rewards) < self.max_episodes:
            # 存储轨迹
            states, actions, rewards, dones = [], [], [], []
            
            # 收集 n_steps 步数据
            for _ in range(self.n_steps):
                # 选择动作
                action, log_prob, value = self.local_model.get_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 存储经验
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(float(done))
                
                episode_reward += reward
                episode_length += 1
                
                state = next_state
                
                if done:
                    # 记录统计
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # 打印进度
                    if self.worker_id == 0 and len(self.episode_rewards) % 10 == 0:
                        avg_reward = np.mean(self.episode_rewards[-10:])
                        print(f"Worker {self.worker_id} | Episode {len(self.episode_rewards)} | "
                              f"Reward: {episode_reward:.1f} | Avg: {avg_reward:.1f} | "
                              f"Length: {episode_length}")
                    
                    # 重置环境
                    state, _ = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    break
            
            # 计算损失并更新
            if len(states) > 0:
                # 添加最后一个状态用于价值估计
                next_states = states[1:] + [state]
                
                # 计算梯度
                self.optimizer.zero_grad()
                
                # 由于 Python 多进程限制，这里简化为单进程训练
                # 实际 A3C 需要使用 multiprocessing
                loss, policy_loss, value_loss, entropy = self.compute_loss(
                    states, actions, rewards, dones, next_states
                )
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.local_model.parameters(), 
                    self.max_grad_norm
                )
                
                # 更新全局网络
                self.optimizer.step()
                self.sync_global()
        
        self.env.close()
        return self.episode_rewards
    
    def close(self):
        """关闭环境"""
        self.env.close()


# ============================================================
# A3C 主类
# ============================================================

class A3C:
    """
    A3C (Asynchronous Advantage Actor-Critic) 算法
    
    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        num_workers: 并行 worker 数量
        gamma: 折扣因子
        lr: 学习率
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 num_workers: int = 4, gamma: float = 0.99,
                 lr: float = 3e-4, entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5, max_grad_norm: float = 0.5,
                 n_steps: int = 5):
        """
        初始化 A3C
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            num_workers: 并行 worker 数量
            gamma: 折扣因子
            lr: 学习率
            entropy_coef: 熵系数
            value_loss_coef: 价值损失系数
            max_grad_norm: 梯度裁剪范数
            n_steps: 每次更新前收集的步数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_workers = num_workers
        self.gamma = gamma
        
        # 创建全局网络
        self.global_model = ActorCritic(state_dim, action_dim)
        self.global_model.share_memory()  # 共享内存
        
        # 共享优化器
        self.optimizer = SharedAdam(self.global_model.parameters(), lr=lr)
        
        # 创建 workers
        self.workers = []
        
    def train(self, env_name: str = "CartPole-v1", 
              max_episodes: int = 500) -> List[float]:
        """
        训练 A3C
        
        Args:
            env_name: 环境名称
            max_episodes: 最大训练回合数
            
        Returns:
            每个回合的奖励列表
        """
        print(f"开始训练 A3C | Workers: {self.num_workers} | Env: {env_name}")
        
        # 简化版本：使用单进程训练（实际 A3C 使用多进程）
        worker = A3CWorker(
            worker_id=0,
            global_model=self.global_model,
            optimizer=self.optimizer,
            env_name=env_name,
            max_episodes=max_episodes,
            gamma=self.gamma,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            n_steps=n_steps
        )
        
        rewards = worker.train()
        
        return rewards
    
    def evaluate(self, env_name: str = "CartPole-v1",
                 num_episodes: int = 10, render: bool = False) -> Dict:
        """
        评估训练好的策略
        
        Args:
            env_name: 环境名称
            num_episodes: 评估回合数
            render: 是否渲染环境
            
        Returns:
            评估统计字典
        """
        env = gym.make(env_name, render_mode="human" if render else None)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _, _ = self.global_model.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"评估 Episode {episode + 1}: Reward = {episode_reward:.1f}, Length = {episode_length}")
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'rewards': episode_rewards,
            'lengths': episode_lengths
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save(self.global_model.state_dict(), path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        self.global_model.load_state_dict(torch.load(path))
        print(f"模型已从: {path} 加载")


# ============================================================
# 训练脚本
# ============================================================

def train_cartpole():
    """训练 CartPole-v1 环境"""
    print("=" * 60)
    print("A3C 训练 CartPole-v1")
    print("=" * 60)
    
    # 创建 A3C
    a3c = A3C(
        state_dim=4,       # CartPole 状态维度
        action_dim=2,       # CartPole 动作维度
        num_workers=4,
        gamma=0.99,
        lr=3e-4,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        n_steps=5
    )
    
    # 训练
    rewards = a3c.train(env_name="CartPole-v1", max_episodes=500)
    
    # 保存模型
    a3c.save("a3c_cartpole.pth")
    
    # 评估
    print("\n评估训练后的策略...")
    results = a3c.evaluate(env_name="CartPole-v1", num_episodes=5, render=True)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  平均回合长度: {results['mean_length']:.2f}")
    
    return rewards, results


def train_lunarlander():
    """训练 LunarLander-v2 环境"""
    print("=" * 60)
    print("A3C 训练 LunarLander-v2")
    print("=" * 60)
    
    # LunarLander-v2 有 8 维状态
    a3c = A3C(
        state_dim=8,
        action_dim=4,
        num_workers=4,
        gamma=0.99,
        lr=3e-4,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        n_steps=5
    )
    
    # 训练
    rewards = a3c.train(env_name="LunarLander-v2", max_episodes=1000)
    
    # 保存模型
    a3c.save("a3c_lunarlander.pth")
    
    # 评估
    results = a3c.evaluate(env_name="LunarLander-v2", num_episodes=5)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards


if __name__ == "__main__":
    # 训练 CartPole
    train_cartpole()
