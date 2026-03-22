"""
Rainbow DQN 算法实现
====================

Rainbow 是 DeepMind 提出的结合多种改进技术的 DQN 算法，整合了：
1. Double DQN - 减少 Q 值过估计
2. Dueling DQN - 分离状态值和动作优势
3. Prioritized Experience Replay - 优先回放重要经验
4. Dueling DQN - (已包含)
5. Distributional RL (C51) - 估计价值分布
6. Noisy Nets - 替代 epsilon-greedy 探索
7. Multi-step Learning - 多步回报

Paper: "Rainbow: Combining Improvements in Deep Reinforcement Learning" - Hessel et al., 2017

Author: Carl
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict, Optional
from collections import deque, namedtuple
import random
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 经验回放缓冲区
# ============================================================

class PrioritizedReplayBuffer:
    """
    优先经验回放 (Prioritized Experience Replay)
    
    核心思想：优先回放 TD 误差大的经验，这些经验包含更多信息
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, 
                 beta: float = 0.4, beta_increment: float = 1e-3):
        """
        初始化优先回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级 exponent
            beta: IS 权重 exponent
            beta_increment: beta 增长量
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # 经验元组
        self.Experience = namedtuple('Experience', 
                                      ['state', 'action', 'reward', 
                                       'next_state', 'done'])
    
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, td_error: float = None):
        """
        添加经验
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            td_error: TD 误差（用于优先级）
        """
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        experience = self.Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # 新经验赋予最大优先级
        priority = max_priority if td_error is None else abs(td_error) + 1e-5
        self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        采样 batch
        
        Args:
            batch_size: batch 大小
            
        Returns:
            batch: 经验 batch
            weights: IS 权重
            indices: 采样索引
        """
        # 更新 beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # 计算 IS 权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        更新优先级
        
        Args:
            indices: 经验索引
            td_errors: TD 误差
        """
        for idx, error in zip(indices, td_errors):
            priority = abs(error) + 1e-5
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size


# ============================================================
# Noisy Linear 层
# ============================================================

class NoisyLinear(nn.Module):
    """
    Noisy Linear 层
    
    使用参数化的噪声替代 epsilon-greedy 探索
    """
    
    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.5):
        """
        初始化 Noisy Linear 层
        
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            sigma_init: 噪声初始化标准差
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # 可学习参数
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / np.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成缩放噪声"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


# ============================================================
# 分布价值网络 (C51)
# ============================================================

class CategoricalDQN(nn.Module):
    """
    Categorical DQN (C51)
    
    估计价值分布而非单点估计
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 512,
                 num_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0):
        """
        初始化 Categorical DQN
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            num_atoms: 分布原子数 (C51)
            v_min: 价值最小值
            v_max: 价值最大值
        """
        super(CategoricalDQN, self).__init__()
        
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # 支持向量
        self.register_buffer('support', 
                            torch.linspace(v_min, v_max, num_atoms))
        
        # 网络
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值头和优势头
        self.value_head = NoisyLinear(hidden_dim, num_atoms)
        self.advantage_head = NoisyLinear(hidden_dim, action_dim * num_atoms)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            动作-价值分布 (batch, action_dim, num_atoms)
        """
        features = self.network(state)
        
        value_dist = self.value_head(features)  # (batch, num_atoms)
        advantage_dist = self.advantage_head(features)  # (batch, action_dim * num_atoms)
        
        # 重新整形
        value_dist = value_dist.view(-1, 1, self.num_atoms)
        advantage_dist = advantage_dist.view(-1, self.action_dim, self.num_atoms)
        
        # 结合价值和优势
        q_dist = value_dist + advantage_dist - advantage_dist.mean(dim=1, keepdim=True)
        
        # 归一化为概率分布
        q_dist = F.softmax(q_dist, dim=-1)
        
        return q_dist
    
    @property
    def action_dim(self):
        return self.advantage_head.out_features // self.num_atoms


class DuelingNetwork(nn.Module):
    """
    Dueling 网络
    
    分离状态价值和动作优势
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 512):
        """
        初始化 Dueling 网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(DuelingNetwork, self).__init__()
        
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, 1)
        )
        
        # 优势头
        self.advantage_head = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            Q 值
        """
        features = self.feature(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


# ============================================================
# Rainbow DQN
# ============================================================

class RainbowDQN:
    """
    Rainbow DQN 算法
    
    结合了 7 种 DQN 改进技术：
    1. Double DQN - 减少 Q 值过估计
    2. Dueling DQN - 分离价值和优势
    3. Prioritized Experience Replay - 优先回放
    4. Distributional RL (C51) - 价值分布
    5. Noisy Nets - 参数化噪声探索
    6. Multi-step Learning - 多步回报
    
    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        gamma: 折扣因子
        lr: 学习率
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 512,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 tau: float = 1e-3,
                 alpha: float = 0.5,  # PER alpha
                 beta: float = 0.4,   # PER beta
                 beta_increment: float = 1e-3,
                 num_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 n_step: int = 3,
                 target_update_freq: int = 500):
        """
        初始化 Rainbow DQN
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            gamma: 折扣因子
            lr: 学习率
            tau: 软更新系数
            alpha: PER 优先级参数
            beta: PER IS 权重参数
            beta_increment: PER beta 增量
            num_atoms: C51 原子数
            v_min: 价值分布最小值
            v_max: 价值分布最大值
            n_step: 多步学习步数
            target_update_freq: 目标网络更新频率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.target_update_freq = target_update_freq
        
        # 使用 Categorical DQN
        self.policy_net = CategoricalDQN(state_dim, action_dim, hidden_dim,
                                         num_atoms, v_min, v_max)
        self.target_net = CategoricalDQN(state_dim, action_dim, hidden_dim,
                                        num_atoms, v_min, v_max)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=100000,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment
        )
        
        # N-step 缓冲区
        self.n_step_buffer = deque(maxlen=n_step)
        
        # 训练统计
        self.train_step = 0
        self.train_stats = {
            'loss': [],
            'q_value': [],
            'priority': []
        }
    
    def select_action(self, state: np.ndarray, 
                     epsilon: float = 0.0) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            epsilon: 探索率
            
        Returns:
            选择的动作
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_dist = self.policy_net(state)  # (1, action_dim, num_atoms)
            q_values = (q_dist * self.policy_net.support).sum(dim=-1)  # (1, action_dim)
            action = q_values.argmax(dim=-1).item()
        
        return action
    
    def compute_n_step_return(self) -> Optional[Tuple]:
        """计算 N-step 回报"""
        if len(self.n_step_buffer) < self.n_step:
            return None
        
        # 获取 N-step 经验
        states, actions, rewards, next_states, dones = zip(*self.n_step_buffer)
        
        # 计算折扣回报
        gamma_n = self.gamma ** self.n_step
        n_step_reward = sum(r * (self.gamma ** i) for i, r in enumerate(rewards))
        
        return states[-1], actions[-1], n_step_reward, next_states[-1], dones[-1]
    
    def add_experience(self, state: np.ndarray, action: int, 
                       reward: float, next_state: np.ndarray, done: bool):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        # 添加到 N-step 缓冲区
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # 计算 N-step 回报
        n_step_data = self.compute_n_step_return()
        
        if n_step_data is not None:
            n_state, n_action, n_reward, n_next_state, n_done = n_step_data
            
            # 估计 TD 误差用于优先级
            td_error = self._compute_td_error(
                n_state, n_action, n_reward, n_next_state, n_done
            )
            
            self.replay_buffer.add(n_state, n_action, n_reward, 
                                  n_next_state, n_done, td_error)
    
    def _compute_td_error(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray,
                         done: bool) -> float:
        """计算 TD 误差"""
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        
        with torch.no_grad():
            # Double DQN
            next_action = self.policy_net(next_state).mean(dim=-1).argmax(dim=-1)
            
            # 目标分布
            next_dist = self.target_net(next_state)
            next_prob = next_dist[0, next_action].numpy()
            
            # 计算目标分布
            v_max = self.policy_net.v_max
            v_min = self.policy_net.v_min
            support = self.policy_net.support.numpy()
            
            delta_z = (v_max - v_min) / (self.policy_net.num_atoms - 1)
            
           tz_reward = np.clip(reward + self.gamma ** self.n_step * (1 - done) * support,
                              v_min, v_max)
            
            b = (tz_reward - v_min) / delta_z
            l = np.floor(b).astype(int)
            u = np.ceil(b).astype(int)
            
            target_dist = np.zeros_like(next_prob)
            for i in range(len(next_prob)):
                target_dist[l[i]] += next_prob[i] * (u[i] - b[i])
                target_dist[u[i]] += next_prob[i] * (b[i] - l[i])
            
            # 当前分布
            current_dist = self.policy_net(state)[0, action].detach().numpy()
            
            td_error = np.abs(target_dist - current_dist).sum()
        
        return td_error
    
    def update(self, batch_size: int) -> Dict:
        """
        更新网络
        
        Args:
            batch_size: batch 大小
            
        Returns:
            更新统计
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # 采样
        batch, weights, indices = self.replay_buffer.sample(batch_size)
        weights = torch.FloatTensor(weights)
        
        # 解包 batch
        states = torch.FloatTensor(np.array([e.state for e in batch]))
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch]))
        dones = torch.FloatTensor([e.done for e in batch])
        
        # 计算当前分布
        dist = self.policy_net(states)  # (batch, action_dim, num_atoms)
        action_dist = dist[np.arange(batch_size), actions]  # (batch, num_atoms)
        
        # 计算目标分布
        with torch.no_grad():
            # Double DQN: 使用 policy_net 选择动作
            next_dist = self.target_net(next_states)
            next_action = next_dist.mean(dim=-1).argmax(dim=-1)
            next_prob = next_dist[np.arange(batch_size), next_action]
            
            # 投影到支持向量
            v_max = self.policy_net.v_max
            v_min = self.policy_net.v_min
            support = self.policy_net.support
            delta_z = (v_max - v_min) / (self.policy_net.num_atoms - 1)
            
            tz = rewards.unsqueeze(1) + self.gamma ** self.n_step * support.unsqueeze(0) * (1 - dones.unsqueeze(1))
            tz = torch.clamp(tz, v_min, v_max)
            
            b = (tz - v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            target_dist = torch.zeros_like(next_prob)
            for i in range(batch_size):
                for j in range(self.policy_net.num_atoms):
                    target_dist[i, l[i, j]] += next_prob[i, j] * (u[i, j] - b[i, j])
                    target_dist[i, u[i, j]] += next_prob[i, j] * (b[i, j] - l[i, j])
        
        # 计算交叉熵损失
        log_p = torch.log(action_dist + 1e-6)
        loss = -(target_dist * log_p).sum(dim=-1)
        loss = (weights * loss).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # 更新优先级
        with torch.no_grad():
            td_errors = loss.detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # 软更新目标网络
        self.soft_update()
        
        # 记录统计
        self.train_stats['loss'].append(loss.item())
        self.train_step += 1
        
        return {'loss': loss.item()}
    
    def soft_update(self):
        """软更新目标网络"""
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + 
                (1 - self.tau) * target_param.data
            )
    
    def hard_update(self):
        """硬更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, env_name: str = "CartPole-v1",
              num_episodes: int = 500,
              batch_size: int = 64,
              buffer_start_size: int = 1000,
              epsilon_start: float = 1.0,
              epsilon_end: float = 0.01,
              epsilon_decay: int = 10000,
              max_steps: int = 500) -> List[float]:
        """
        训练 Rainbow DQN
        
        Args:
            env_name: 环境名称
            num_episodes: 训练回合数
            batch_size: batch 大小
            buffer_start_size: 开始训练的缓冲区大小
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            max_steps: 每个回合最大步数
            
        Returns:
            奖励列表
        """
        env = gym.make(env_name)
        
        episode_rewards = []
        best_reward = 0
        
        state, _ = env.reset()
        
        print(f"开始训练 Rainbow DQN | Env: {env_name}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                # epsilon 衰减
                epsilon = max(
                    epsilon_end,
                    epsilon_start - self.train_step / epsilon_decay
                )
                
                # 选择动作
                action = self.select_action(state, epsilon)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 添加经验
                self.add_experience(state, action, reward, next_state, done)
                
                # 更新网络
                if self.train_step >= buffer_start_size:
                    stats = self.update(batch_size)
                    
                    if self.train_step % self.target_update_freq == 0:
                        self.hard_update()
                
                episode_reward += reward
                state = next_state
                steps += 1
            
            episode_rewards.append(episode_reward)
            
            # 重置环境
            state, _ = env.reset()
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.1f} | "
                      f"Avg: {avg_reward:.1f} | "
                      f"Epsilon: {epsilon:.3f} | "
                      f"Loss: {self.train_stats['loss'][-1] if self.train_stats['loss'] else 0:.4f}")
                
                # 保存最佳模型
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save("rainbow_dqn_best.pth")
            
            # 提前停止
            if len(episode_rewards) >= 10:
                avg_reward = np.mean(episode_rewards[-10:])
                if env_name == "CartPole-v1" and avg_reward >= 495:
                    print(f"\n🎉 训练完成！平均奖励达到 {avg_reward:.1f}")
                    break
        
        env.close()
        
        return episode_rewards
    
    def evaluate(self, env_name: str = "CartPole-v1",
                 num_episodes: int = 10,
                 render: bool = False) -> Dict:
        """
        评估策略
        
        Args:
            env_name: 环境名称
            num_episodes: 评估回合数
            render: 是否渲染
            
        Returns:
            评估结果
        """
        env = gym.make(env_name, render_mode="human" if render else None)
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, epsilon=0)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
            print(f"评估 Episode {episode + 1}: Reward = {episode_reward:.1f}")
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'rewards': episode_rewards
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step
        }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']
        print(f"模型已从: {path} 加载")


# ============================================================
# 训练脚本
# ============================================================

def train_cartpole():
    """训练 CartPole-v1"""
    print("=" * 60)
    print("Rainbow DQN 训练 CartPole-v1")
    print("=" * 60)
    
    rainbow = RainbowDQN(
        state_dim=4,
        action_dim=2,
        hidden_dim=128,
        gamma=0.99,
        lr=1e-4,
        tau=5e-3,
        alpha=0.5,
        beta=0.4,
        num_atoms=51,
        v_min=0,
        v_max=200,
        n_step=3
    )
    
    rewards = rainbow.train(env_name="CartPole-v1", num_episodes=500)
    
    rainbow.save("rainbow_dqn_cartpole.pth")
    results = rainbow.evaluate(env_name="CartPole-v1", num_episodes=5, render=True)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards


def train_lunarlander():
    """训练 LunarLander-v2"""
    print("=" * 60)
    print("Rainbow DQN 训练 LunarLander-v2")
    print("=" * 60)
    
    rainbow = RainbowDQN(
        state_dim=8,
        action_dim=4,
        hidden_dim=256,
        gamma=0.99,
        lr=5e-4,
        num_atoms=51,
        v_min=-200,
        v_max=200,
        n_step=3
    )
    
    rewards = rainbow.train(env_name="LunarLander-v2", num_episodes=1000)
    
    rainbow.save("rainbow_dqn_lunarlander.pth")
    results = rainbow.evaluate(env_name="LunarLander-v2", num_episodes=5)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards


if __name__ == "__main__":
    train_cartpole()
