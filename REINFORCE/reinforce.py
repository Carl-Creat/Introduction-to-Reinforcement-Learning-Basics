"""
REINFORCE (Monte Carlo Policy Gradient)

论文: Simple Statistical Gradient-Following Algorithms for Connectionist
      Reinforcement Learning (Williams, 1992)

核心思想:
直接对策略参数化，使用蒙特卡洛方法估计梯度。
对于每个 episode，收集完整轨迹后计算回报 G_t，
然后沿着增大高回报动作概率的方向更新策略。

策略梯度定理:
  ∇J(θ) = E[∇log π(a|s;θ) * G_t]

优点:
- 直接优化策略，适合连续动作空间
- 无需价值函数近似

缺点:
- 高方差（蒙特卡洛估计）
- 需要完整 episode 才能更新
- 样本效率低

时间复杂度: O(T) per episode，T 为 episode 长度
空间复杂度: O(|θ|)，θ 为策略参数
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


class PolicyNetwork(nn.Module):
    """
    策略网络 π(a|s;θ)
    输出每个动作的概率分布
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class REINFORCEAgent:
    """
    REINFORCE Agent (带 baseline 的版本)
    
    使用 baseline（平均回报）减少方差:
    ∇J(θ) = E[∇log π(a|s;θ) * (G_t - b)]
    
    Args:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        lr: 学习率
        gamma: 折扣因子
        use_baseline: 是否使用 baseline 减少方差
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True
    ):
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 存储一个 episode 的轨迹
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        """根据策略网络采样动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float):
        self.rewards.append(reward)

    def compute_returns(self) -> torch.Tensor:
        """计算折扣回报 G_t"""
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        # 标准化回报，减少方差
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self) -> float:
        """在 episode 结束后更新策略"""
        returns = self.compute_returns()

        # baseline: 使用回报均值
        baseline = returns.mean() if self.use_baseline else 0.0

        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # 负号：因为我们要最大化期望回报，但优化器做梯度下降
            policy_loss.append(-log_prob * (G - baseline))

        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # 清空轨迹
        self.log_probs = []
        self.rewards = []

        return loss.item()


def train(env_name: str = "CartPole-v1", episodes: int = 1000):
    """训练 REINFORCE Agent"""
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, action_dim)
    rewards_history = []

    for episode in range(episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            next_state, reward, done = result[0], result[1], result[2]
            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        agent.update()
        rewards_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}/{episodes} | Avg Reward: {avg:.1f}")

    env.close()
    return rewards_history


if __name__ == "__main__":
    train()
