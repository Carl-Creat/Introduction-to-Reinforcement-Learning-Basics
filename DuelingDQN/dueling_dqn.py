"""
Dueling DQN (Dueling Deep Q-Network)

论文: Dueling Network Architectures for Deep Reinforcement Learning (Wang et al., 2016)
https://arxiv.org/abs/1511.06581

核心思想:
将 Q 网络分为两个分支:
  - Value Stream V(s): 状态价值，与动作无关
  - Advantage Stream A(s,a): 每个动作相对于平均水平的优势

Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))

优势:
- 更高效地学习状态价值，尤其在动作对结果影响不大时
- 比标准 DQN 收敛更快、更稳定

时间复杂度: O(n) per step
空间复杂度: O(|S| * |A|)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Tuple


class DuelingNetwork(nn.Module):
    """
    Dueling DQN 网络结构
    
    将网络分为共享特征层 + Value 分支 + Advantage 分支
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(DuelingNetwork, self).__init__()
        self.action_dim = action_dim

        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value 分支: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage 分支: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        减去均值是为了保证可识别性 (identifiability)
        """
        features = self.feature(x)
        value = self.value_stream(features)          # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
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


class DuelingDQNAgent:
    """
    Dueling DQN Agent
    
    Args:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        lr: 学习率
        gamma: 折扣因子
        epsilon: 探索率
        epsilon_decay: 探索率衰减
        epsilon_min: 最小探索率
        batch_size: 批次大小
        target_update: 目标网络更新频率
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        target_update: int = 10
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 在线网络 & 目标网络
        self.online_net = DuelingNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def select_action(self, state: np.ndarray) -> int:
        """epsilon-greedy 动作选择"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> float:
        """从经验回放中采样并更新网络"""
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前 Q 值
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 目标 Q 值 (Double DQN 思想: 用在线网络选动作，目标网络评估)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 定期同步目标网络
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()


def train(env_name: str = "CartPole-v1", episodes: int = 500):
    """训练 Dueling DQN"""
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DuelingDQNAgent(state_dim, action_dim)
    rewards_history = []

    for episode in range(episodes):
        state, _ = env.reset() if hasattr(env.reset(), '__iter__') else (env.reset(), {})
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            agent.store(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        if (episode + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Episode {episode+1}/{episodes} | Avg Reward: {avg:.1f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards_history


if __name__ == "__main__":
    train()
