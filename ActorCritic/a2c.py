"""
A2C (Advantage Actor-Critic) 算法实现
======================================

A2C 是 A3C 的同步版本，使用同步更新替代异步更新。

核心思想：
- 同步更新所有 worker 的梯度
- 使用 Advantage 估计减少方差
- 比 A3C 更稳定，但训练速度较慢

特点：
- 同步更新，梯度更稳定
- 使用 GAE 进行优势估计
- 适合离散和连续动作空间

Author: Carl
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 神经网络定义
# ============================================================

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络
    
    共享特征提取层，独立的策略头和价值头
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        """
        初始化网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
        """
        super(ActorCriticNetwork, self).__init__()
        
        # 特征提取网络
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.feature = nn.Sequential(*layers)
        
        # Actor (策略网络) - 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (价值网络) - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            action_probs: 动作概率分布
            value: 状态价值
        """
        features = self.feature(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state: np.ndarray, 
                   deterministic: bool = False) -> Tuple[int, float, float]:
        """
        获取动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        
        action_probs, value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs[0, action] + 1e-8)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, states: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态和动作的对数概率和价值
        
        Args:
            states: 状态 batch
            actions: 动作 batch
            
        Returns:
            log_probs: 动作的对数概率
            values: 状态价值
            entropy: 策略熵
        """
        action_probs, values = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


# ============================================================
# GAE (Generalized Advantage Estimation)
# ============================================================

def compute_gae(rewards: List[float], values: List[float],
                dones: List[float], gamma: float = 0.99,
                gae_lambda: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    计算 GAE (广义优势估计)
    
    Args:
        rewards: 奖励列表
        values: 价值列表
        dones: 终止标志列表
        gamma: 折扣因子
        gae_lambda: GAE 参数
        
    Returns:
        advantages: 优势列表
        returns: 回报列表
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    # 回报 = 优势 + 价值
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    return advantages, returns


# ============================================================
# A2C 算法实现
# ============================================================

class A2C:
    """
    A2C (Advantage Actor-Critic) 算法
    
    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        gamma: 折扣因子
        lr: 学习率
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 lr: float = 3e-4,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        """
        初始化 A2C
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度
            gamma: 折扣因子
            gae_lambda: GAE 参数
            lr: 学习率
            value_loss_coef: 价值损失系数
            entropy_coef: 熵系数
            max_grad_norm: 梯度裁剪范数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建网络
        self.policy = ActorCriticNetwork(state_dim, action_dim, hidden_dims)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 训练统计
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        action, log_prob, _ = self.policy.get_action(state)
        return action, log_prob
    
    def collect_rollout(self, env: gym.Env, 
                        max_steps: int = 1000) -> Tuple:
        """
        收集一次 rollout
        
        Args:
            env: 环境
            max_steps: 最大步数
            
        Returns:
            收集的数据
        """
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            # 获取动作
            action, log_prob, value = self.policy.get_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)
            log_probs.append(log_prob)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # 添加最后状态的价值
        with torch.no_grad():
            last_value = 0 if done else self.policy.forward(
                torch.FloatTensor(state).unsqueeze(0)
            )[1].item()
        values.append(last_value)
        
        return (states, actions, rewards, dones, values, log_probs,
                episode_reward, episode_length)
    
    def update(self, states: torch.Tensor, actions: torch.Tensor,
               log_probs: torch.Tensor, returns: torch.Tensor,
               advantages: torch.Tensor) -> Dict:
        """
        更新策略和价值网络
        
        Args:
            states: 状态 batch
            actions: 动作 batch
            log_probs: 旧策略的对数概率
            returns: 回报
            advantages: 优势估计
            
        Returns:
            更新统计信息
        """
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 评估当前策略
        new_log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
        
        # 策略损失 (Policy Loss)
        # 使用策略梯度: -log(π(a|s)) * A(s,a)
        policy_loss = -(new_log_probs * advantages).mean()
        
        # 价值损失 (Value Loss)
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # 熵损失 (鼓励探索)
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = (policy_loss +
                     self.value_loss_coef * value_loss +
                     self.entropy_coef * entropy_loss)
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        
        # 记录统计
        self.train_stats['policy_loss'].append(policy_loss.item())
        self.train_stats['value_loss'].append(value_loss.item())
        self.train_stats['entropy'].append(entropy.mean().item())
        self.train_stats['total_loss'].append(total_loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, env_name: str = "CartPole-v1",
              num_episodes: int = 500,
              max_steps_per_episode: int = 1000,
              update_interval: int = 5) -> Tuple[List[float], List[Dict]]:
        """
        训练 A2C
        
        Args:
            env_name: 环境名称
            num_episodes: 训练回合数
            max_steps_per_episode: 每个回合最大步数
            update_interval: 更新间隔（收集多少步后更新）
            
        Returns:
            奖励列表和更新统计
        """
        env = gym.make(env_name)
        
        # 获取状态和动作维度
        if isinstance(env.observation_space, gym.spaces.Box):
            state_dim = env.observation_space.shape[0]
        else:
            state_dim = env.observation_space.n
        
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
        
        # 重新创建网络（使用正确的维度）
        self.policy = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        episode_rewards = []
        episode_lengths = []
        update_stats = []
        
        state, _ = env.reset()
        
        # 缓冲区
        states_buffer, actions_buffer = [], []
        rewards_buffer, dones_buffer = [], []
        values_buffer, log_probs_buffer = [], []
        
        episode_reward = 0
        episode_length = 0
        
        print(f"开始训练 A2C | Env: {env_name}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            for step in range(max_steps_per_episode):
                # 选择动作
                action, log_prob, value = self.policy.get_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 存储
                states_buffer.append(state)
                actions_buffer.append(action)
                rewards_buffer.append(reward)
                dones_buffer.append(float(done))
                values_buffer.append(value)
                log_probs_buffer.append(log_prob)
                
                episode_reward += reward
                episode_length += 1
                
                state = next_state
                
                # 检查是否需要更新
                if len(states_buffer) >= update_interval or done:
                    # 计算 GAE
                    advantages, returns = compute_gae(
                        rewards_buffer, values_buffer, dones_buffer,
                        self.gamma, self.gae_lambda
                    )
                    
                    # 转换为张量
                    states_tensor = torch.FloatTensor(np.array(states_buffer))
                    actions_tensor = torch.LongTensor(actions_buffer)
                    log_probs_tensor = torch.FloatTensor(log_probs_buffer)
                    returns_tensor = torch.FloatTensor(returns)
                    advantages_tensor = torch.FloatTensor(advantages)
                    
                    # 更新
                    stats = self.update(states_tensor, actions_tensor,
                                       log_probs_tensor, returns_tensor,
                                       advantages_tensor)
                    update_stats.append(stats)
                    
                    # 清空缓冲区
                    states_buffer, actions_buffer = [], []
                    rewards_buffer, dones_buffer = [], []
                    values_buffer, log_probs_buffer = [], []
                
                if done:
                    break
            
            # 记录奖励
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.1f} | "
                      f"Avg: {avg_reward:.1f} | "
                      f"Length: {episode_length} | "
                      f"Loss: {stats['total_loss']:.3f}")
            
            # 重置
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # 提前停止
            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                if env_name == "CartPole-v1" and avg_reward >= 495:
                    print(f"\n🎉 训练完成！平均奖励达到 {avg_reward:.1f}")
                    break
        
        env.close()
        
        return episode_rewards, update_stats
    
    def evaluate(self, env_name: str = "CartPole-v1",
                 num_episodes: int = 10,
                 render: bool = False) -> Dict:
        """
        评估训练好的策略
        
        Args:
            env_name: 环境名称
            num_episodes: 评估回合数
            render: 是否渲染
            
        Returns:
            评估结果
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
                action, _, _ = self.policy.get_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"评估 Episode {episode + 1}: Reward = {episode_reward:.1f}")
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'rewards': episode_rewards
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save(self.policy.state_dict(), path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        self.policy.load_state_dict(torch.load(path))
        print(f"模型已从: {path} 加载")


# ============================================================
# 训练脚本
# ============================================================

def train_cartpole():
    """训练 CartPole-v1"""
    print("=" * 60)
    print("A2C 训练 CartPole-v1")
    print("=" * 60)
    
    a2c = A2C(
        state_dim=4,
        action_dim=2,
        hidden_dims=[256, 256],
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    )
    
    rewards, stats = a2c.train(env_name="CartPole-v1", num_episodes=500)
    
    a2c.save("a2c_cartpole.pth")
    results = a2c.evaluate(env_name="CartPole-v1", num_episodes=5, render=True)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards, stats


def train_lunarlander():
    """训练 LunarLander-v2"""
    print("=" * 60)
    print("A2C 训练 LunarLander-v2")
    print("=" * 60)
    
    a2c = A2C(
        state_dim=8,
        action_dim=4,
        hidden_dims=[256, 256],
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        value_loss_coef=0.5,
        entropy_coef=0.01
    )
    
    rewards, stats = a2c.train(env_name="LunarLander-v2", num_episodes=1000)
    
    a2c.save("a2c_lunarlander.pth")
    results = a2c.evaluate(env_name="LunarLander-v2", num_episodes=5)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards


def train_acrobot():
    """训练 Acrobot-v1"""
    print("=" * 60)
    print("A2C 训练 Acrobot-v1")
    print("=" * 60)
    
    a2c = A2C(
        state_dim=6,
        action_dim=3,
        hidden_dims=[128, 128],
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        entropy_coef=0.01
    )
    
    rewards, stats = a2c.train(env_name="Acrobot-v1", num_episodes=500)
    
    a2c.save("a2c_acrobot.pth")
    results = a2c.evaluate(env_name="Acrobot-v1", num_episodes=5)
    
    return rewards


if __name__ == "__main__":
    train_cartpole()
