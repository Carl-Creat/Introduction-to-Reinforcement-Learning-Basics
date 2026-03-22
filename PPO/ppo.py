"""
PPO (Proximal Policy Optimization) 算法实现
============================================

PPO 是由 OpenAI 提出的基于策略的强化学习算法，是目前最流行的 RL 算法之一。

核心思想：
- 使用信任域方法优化策略
- 通过裁剪机制限制策略更新的幅度
- 能够在复杂环境中稳定训练

特点：
- 裁剪的目标函数防止策略大幅变化
- 简单高效，易于实现
- 适用于离散和连续动作空间
- 超参数友好

Paper: "Proximal Policy Optimization Algorithms" - Schulman et al., 2017

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
    Actor-Critic 神经网络
    
    共享特征提取层，独立的策略头和价值头
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256]):
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


class RolloutBuffer:
    """
    经验回放缓冲区
    
    存储轨迹数据，用于 PPO 更新
    """
    
    def __init__(self, buffer_size: int, state_dim: int, gamma: float = 0.99, 
                 gae_lambda: float = 0.95):
        """
        初始化缓冲区
        
        Args:
            buffer_size: 缓冲区大小
            state_dim: 状态维度
            gamma: 折扣因子
            gae_lambda: GAE 参数
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 初始化存储数组
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.trajectory_start = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            done: bool, log_prob: float, value: float):
        """添加一条经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr += 1
    
    def compute_returns_and_advantages(self, last_value: float = 0):
        """
        计算回报和优势估计
        
        使用 GAE (Generalized Advantage Estimation)
        
        Args:
            last_value: 最后一个状态的估计价值
        """
        advantages = np.zeros(self.ptr, dtype=np.float32)
        returns = np.zeros(self.ptr, dtype=np.float32)
        
        # 反向计算 GAE
        gae = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
        
        return returns, advantages
    
    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取所有数据"""
        return (self.states[:self.ptr], self.actions[:self.ptr], 
                self.log_probs[:self.ptr], self.returns[:self.ptr] 
                if hasattr(self, 'returns') else None,
                self.advantages[:self.ptr] if hasattr(self, 'advantages') else None)
    
    def reset(self):
        """重置缓冲区"""
        self.ptr = 0
        self.trajectory_start = 0


# ============================================================
# PPO 算法实现
# ============================================================

class PPO:
    """
    PPO (Proximal Policy Optimization) 算法
    
    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        gamma: 折扣因子
        lr: 学习率
        clip_epsilon: 裁剪参数
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 gamma: float = 0.99, lr: float = 3e-4,
                 clip_epsilon: float = 0.2, 
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 gae_lambda: float = 0.95,
                 update_epochs: int = 10,
                 batch_size: int = 64):
        """
        初始化 PPO
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度
            gamma: 折扣因子
            lr: 学习率
            clip_epsilon: PPO 裁剪参数
            value_loss_coef: 价值损失系数
            entropy_coef: 熵系数
            max_grad_norm: 梯度裁剪范数
            gae_lambda: GAE 参数
            update_epochs: 每次更新的 epoch 数
            batch_size: mini-batch 大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # 创建策略网络
        self.policy = ActorCriticNetwork(state_dim, action_dim, hidden_dims)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 训练统计
        self.train_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': []
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
    
    def update(self, states: torch.Tensor, actions: torch.Tensor,
               old_log_probs: torch.Tensor, returns: torch.Tensor,
               advantages: torch.Tensor) -> Dict:
        """
        更新策略网络
        
        Args:
            states: 状态 batch
            actions: 动作 batch
            old_log_probs: 旧策略的对数概率
            returns: 回报
            advantages: 优势估计
            
        Returns:
            更新统计信息
        """
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多次更新
        for _ in range(self.update_epochs):
            # 打乱数据
            indices = torch.randperm(states.size(0))
            
            for start in range(0, states.size(0), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 评估当前策略
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # 计算比率 r(θ) = π(a|s) / π_old(a|s)
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # 裁剪目标函数
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 
                                   1 + self.clip_epsilon) * batch_advantages
                
                # 策略损失（最小化）
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # 熵损失（鼓励探索）
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
                
                # 计算 KL 散度（用于监控）
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean()
                
                # 记录统计
                self.train_stats['policy_loss'].append(policy_loss.item())
                self.train_stats['value_loss'].append(value_loss.item())
                self.train_stats['entropy'].append(entropy.mean().item())
                self.train_stats['kl_divergence'].append(kl.item())
        
        return {
            'policy_loss': np.mean(self.train_stats['policy_loss'][-10:]),
            'value_loss': np.mean(self.train_stats['value_loss'][-10:]),
            'entropy': np.mean(self.train_stats['entropy'][-10:])
        }
    
    def train(self, env_name: str = "CartPole-v1",
              num_episodes: int = 500,
              buffer_size: int = 2048,
              max_steps: int = 500) -> Tuple[List[float], List[Dict]]:
        """
        训练 PPO
        
        Args:
            env_name: 环境名称
            num_episodes: 训练回合数
            buffer_size: 回放缓冲区大小
            max_steps: 每个回合最大步数
            
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
        
        # 创建缓冲区
        buffer = RolloutBuffer(buffer_size, state_dim, self.gamma)
        
        episode_rewards = []
        episode_lengths = []
        update_stats = []
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"开始训练 PPO | Env: {env_name}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            for step in range(buffer_size):
                # 选择动作
                action, log_prob = self.select_action(state)
                
                # 获取价值
                with torch.no_grad():
                    _, value = self.policy.forward(
                        torch.FloatTensor(state).unsqueeze(0)
                    )
                    value = value.item()
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 存储经验
                buffer.add(state, action, reward, done, log_prob, value)
                
                episode_reward += reward
                episode_length += 1
                
                state = next_state
                
                # 检查是否结束
                if done or episode_length >= max_steps:
                    # 计算最后的价值（用于 GAE）
                    if done:
                        last_value = 0
                    else:
                        with torch.no_grad():
                            _, last_value = self.policy.forward(
                                torch.FloatTensor(state).unsqueeze(0)
                            )
                            last_value = last_value.item()
                    
                    # 计算回报和优势
                    returns, advantages = buffer.compute_returns_and_advantages(last_value)
                    buffer.returns = returns
                    buffer.advantages = advantages
                    
                    # 更新策略
                    states = torch.FloatTensor(buffer.states[:buffer.ptr])
                    actions = torch.LongTensor(buffer.actions[:buffer.ptr])
                    old_log_probs = torch.FloatTensor(buffer.log_probs[:buffer.ptr])
                    returns_tensor = torch.FloatTensor(returns)
                    advantages_tensor = torch.FloatTensor(advantages)
                    
                    stats = self.update(states, actions, old_log_probs, 
                                       returns_tensor, advantages_tensor)
                    update_stats.append(stats)
                    
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
                              f"Policy Loss: {stats['policy_loss']:.3f}")
                    
                    # 重置
                    state, _ = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    buffer.reset()
            
            # 达到目标时提前停止
            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                if env_name == "CartPole-v1" and avg_reward >= 495:
                    print(f"\n🎉 提前停止！平均奖励达到 {avg_reward:.1f}")
                    break
                elif env_name in ["LunarLander-v2", "Pendulum-v1"] and avg_reward >= 200:
                    print(f"\n🎉 提前停止！平均奖励达到 {avg_reward:.1f}")
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
                action, _ = self.select_action(state)
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
    print("PPO 训练 CartPole-v1")
    print("=" * 60)
    
    ppo = PPO(
        state_dim=4,
        action_dim=2,
        hidden_dims=[256, 256],
        gamma=0.99,
        lr=3e-4,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        update_epochs=10,
        batch_size=64
    )
    
    rewards, stats = ppo.train(env_name="CartPole-v1", num_episodes=500)
    
    ppo.save("ppo_cartpole.pth")
    results = ppo.evaluate(env_name="CartPole-v1", num_episodes=5, render=True)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards, stats


def train_lunarlander():
    """训练 LunarLander-v2"""
    print("=" * 60)
    print("PPO 训练 LunarLander-v2")
    print("=" * 60)
    
    ppo = PPO(
        state_dim=8,
        action_dim=4,
        hidden_dims=[256, 256],
        gamma=0.99,
        lr=3e-4,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        update_epochs=10
    )
    
    rewards, stats = ppo.train(env_name="LunarLander-v2", num_episodes=1000)
    
    ppo.save("ppo_lunarlander.pth")
    results = ppo.evaluate(env_name="LunarLander-v2", num_episodes=5)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards


def train_pendulum():
    """训练 Pendulum-v1（连续动作空间）"""
    print("=" * 60)
    print("PPO 训练 Pendulum-v1")
    print("=" * 60)
    
    # Pendulum-v1 有连续动作空间
    ppo = PPO(
        state_dim=3,
        action_dim=11,  # 离散化
        hidden_dims=[256, 256],
        gamma=0.99,
        lr=3e-4,
        clip_epsilon=0.2,
        entropy_coef=0.0
    )
    
    rewards, stats = ppo.train(env_name="Pendulum-v1", num_episodes=500)
    
    ppo.save("ppo_pendulum.pth")
    results = ppo.evaluate(env_name="Pendulum-v1", num_episodes=5)
    
    return rewards


if __name__ == "__main__":
    train_cartpole()
