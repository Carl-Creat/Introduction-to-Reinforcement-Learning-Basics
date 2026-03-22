"""
TRPO (Trust Region Policy Optimization) 算法实现
================================================

TRPO 是一种基于信任域策略优化算法，由 Schulman 等人提出。
与 PPO 类似，都是为了解决策略梯度算法中策略更新过大的问题。

核心思想：
- 使用 KL 散度约束限制策略更新
- 通过共轭梯度法近似求解约束优化问题
- 理论上保证单调策略提升

特点：
- 理论上更严格的一致性保证
- 使用线搜索确保约束满足
- 计算开销较大

Paper: "Trust Region Policy Optimization" - Schulman et al., 2015

Author: Carl
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 策略网络和价值网络
# ============================================================

class PolicyNetwork(nn.Module):
    """
    策略网络（Actor）
    
    输出动作概率分布
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dims: List[int] = [64, 64]):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度
        """
        super(PolicyNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回动作 logits"""
        features = self.hidden(state)
        logits = self.output(features)
        return logits
    
    def get_action_probs(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """获取动作概率分布"""
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """计算给定状态-动作对的对数概率"""
        dist = self.get_action_probs(state)
        return dist.log_prob(action)
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """计算策略熵"""
        dist = self.get_action_probs(state)
        return dist.entropy()


class ValueNetwork(nn.Module):
    """
    价值网络（Critic）
    
    估计状态价值函数
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [64, 64]):
        """
        初始化价值网络
        
        Args:
            state_dim: 状态空间维度
            hidden_dims: 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回状态价值"""
        return self.network(state)


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
# 共轭梯度法
# ============================================================

def conjugate_gradient(fvp: callable, b: torch.Tensor, 
                       max_iter: int = 10, tolerance: float = 1e-10) -> torch.Tensor:
    """
    共轭梯度法求解 Fx = b
    
    其中 F 是 Fisher 信息矩阵（用于自然梯度）
    
    Args:
        fvp: Fisher-vector product 函数
        b: 右侧向量
        max_iter: 最大迭代次数
        tolerance: 收敛容差
        
    Returns:
        x: 近似解
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rsold = torch.dot(r, r)
    
    for _ in range(max_iter):
        Fp = fvp(p)
        alpha = rsold / (torch.dot(p, Fp) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Fp
        rsnew = torch.dot(r, r)
        
        if torch.sqrt(rsnew) < tolerance:
            break
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x


# ============================================================
# TRPO 算法实现
# ============================================================

class TRPO:
    """
    TRPO (Trust Region Policy Optimization) 算法
    
    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        gamma: 折扣因子
        kl_target: 目标 KL 散度
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 kl_target: float = 0.01,
                 max_kl: float = 0.01,
                 cg_iterations: int = 10,
                 max_backtrack: int = 10,
                 damping: float = 1e-3,
                 value_lr: float = 1e-3,
                 value_epochs: int = 5):
        """
        初始化 TRPO
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度
            gamma: 折扣因子
            gae_lambda: GAE 参数
            kl_target: 目标 KL 散度
            max_kl: 最大 KL 散度
            cg_iterations: 共轭梯度迭代次数
            max_backtrack: 最大回溯次数
            damping: 阻尼系数
            value_lr: 价值网络学习率
            value_epochs: 价值网络训练轮数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.kl_target = kl_target
        self.max_kl = max_kl
        self.cg_iterations = cg_iterations
        self.max_backtrack = max_backtrack
        self.damping = damping
        
        # 创建网络
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims)
        self.value = ValueNetwork(state_dim, hidden_dims)
        
        # 优化器
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)
        
        # 训练统计
        self.train_stats = {
            'kl_divergence': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': []
        }
    
    def flat_gradients(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """将梯度列表展平为向量"""
        return torch.cat([grad.view(-1) for grad in gradients])
    
    def flat_params(self, model: nn.Module) -> torch.Tensor:
        """将模型参数展平为向量"""
        return torch.cat([param.view(-1) for param in model.parameters()])
    
    def set_params(self, model: nn.Module, flat_params: torch.Tensor):
        """从展平的向量设置模型参数"""
        offset = 0
        for param in model.parameters():
            num_params = param.numel()
            param.data.copy_(flat_params[offset:offset + num_params].view(param.shape))
            offset += num_params
    
    def fisher_vector_product(self, states: torch.Tensor, 
                              damping: float = None) -> callable:
        """
        计算 Fisher-vector product
        
        F * x = (d(KL)/d(θ_old)) @ (d²(KL)/d(θ)² @ x)
        
        Args:
            states: 状态 batch
            damping: 阻尼系数
            
        Returns:
            Fisher-vector product 函数
        """
        if damping is None:
            damping = self.damping
        
        def fvp(v: torch.Tensor) -> torch.Tensor:
            # 计算当前策略的对数概率
            logits = self.policy.forward(states)
            dist = torch.distributions.Categorical(logits=logits)
            
            # 计算策略分布的梯度
            grads = torch.autograd.grad(
                dist.entropy().mean(),
                self.policy.parameters(),
                create_graph=True
            )
            
            # Fisher 信息矩阵与向量的乘积
            flat_grads = self.flat_gradients(grads)
            flat_grads_dot_v = torch.dot(flat_grads, v)
            
            second_grads = torch.autograd.grad(
                flat_grads_dot_v,
                self.policy.parameters()
            )
            
            flat_second_grads = self.flat_gradients(second_grads)
            
            # 添加阻尼
            return flat_second_grads + damping * v
        
        return fvp
    
    def compute_surrogate_loss(self, states: torch.Tensor, 
                               actions: torch.Tensor,
                               old_log_probs: torch.Tensor,
                               advantages: torch.Tensor) -> torch.Tensor:
        """
        计算替代损失函数
        
        L(θ) = E[π_θ(a|s) / π_θ_old(a|s) * A]
        
        Args:
            states: 状态 batch
            actions: 动作 batch
            old_log_probs: 旧策略的对数概率
            advantages: 优势估计
            
        Returns:
            替代损失
        """
        log_probs = self.policy.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return -(ratio * advantages).mean()
    
    def update_policy(self, states: torch.Tensor, actions: torch.Tensor,
                      old_log_probs: torch.Tensor, 
                      advantages: torch.Tensor) -> Dict:
        """
        更新策略网络
        
        Args:
            states: 状态 batch
            actions: 动作 batch
            old_log_probs: 旧策略的对数概率
            advantages: 优势估计
            
        Returns:
            更新统计信息
        """
        # 计算损失梯度
        loss = self.compute_surrogate_loss(states, actions, old_log_probs, advantages)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        policy_gradient = self.flat_gradients(grads)
        
        # 计算自然梯度方向
        fvp = self.fisher_vector_product(states)
        natural_gradient = conjugate_gradient(fvp, policy_gradient)
        
        # 计算步长
        # H^-1 * g @ g = g^T * H^-1 * g
        step_size = torch.sqrt(
            2 * self.kl_target / (torch.dot(natural_gradient, fvp(natural_gradient)) + 1e-8)
        ).item()
        
        # 归一化自然梯度
        step = natural_gradient * step_size
        
        # 保留旧参数用于回溯
        old_params = self.flat_params(self.policy)
        
        # 线搜索
        for backtrack in range(self.max_backtrack):
            # 尝试新的参数
            new_params = old_params - step * (0.5 ** backtrack)
            self.set_params(self.policy, new_params)
            
            # 计算新的 KL 散度
            with torch.no_grad():
                new_logits = self.policy.forward(states)
                new_dist = torch.distributions.Categorical(logits=new_logits)
                old_dist = torch.distributions.Categorical(logits=old_logits 
                                                           if 'old_logits' in dir() 
                                                           else self.policy.forward(states))
                kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            
            # 检查 KL 约束
            if kl <= self.kl_target:
                break
        
        # 恢复参数如果 KL 约束不满足
        if kl > self.kl_target:
            self.set_params(self.policy, old_params)
        
        # 计算并记录统计
        with torch.no_grad():
            entropy = self.policy.get_entropy(states).mean().item()
        
        self.train_stats['kl_divergence'].append(kl.item())
        self.train_stats['policy_loss'].append(loss.item())
        self.train_stats['entropy'].append(entropy)
        
        return {
            'kl': kl.item(),
            'policy_loss': loss.item(),
            'entropy': entropy
        }
    
    def update_value(self, states: torch.Tensor, 
                     returns: torch.Tensor) -> float:
        """
        更新价值网络
        
        Args:
            states: 状态 batch
            returns: 回报
            
        Returns:
            价值损失
        """
        for _ in range(self.value_epochs):
            values = self.value(states).squeeze()
            loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()
        
        self.train_stats['value_loss'].append(loss.item())
        return loss.item()
    
    def collect_rollout(self, env: gym.Env, max_steps: int = 1000) -> Tuple:
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
            # 获取动作概率和价值
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                dist = self.policy.get_action_probs(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.value(state_tensor).item()
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # 存储
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)
            log_probs.append(log_prob.item())
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # 添加最后状态的价值
        with torch.no_grad():
            last_value = 0 if done else self.value(
                torch.FloatTensor(state).unsqueeze(0)
            ).item()
        values.append(last_value)
        
        return (states, actions, rewards, dones, values, log_probs, 
                episode_reward, episode_length)
    
    def train(self, env_name: str = "CartPole-v1",
              num_episodes: int = 500,
              max_steps_per_episode: int = 1000) -> Tuple[List[float], List[Dict]]:
        """
        训练 TRPO
        
        Args:
            env_name: 环境名称
            num_episodes: 训练回合数
            max_steps_per_episode: 每个回合最大步数
            
        Returns:
            奖励列表和更新统计
        """
        env = gym.make(env_name)
        
        episode_rewards = []
        episode_lengths = []
        update_stats = []
        
        print(f"开始训练 TRPO | Env: {env_name}")
        print("=" * 60)
        
        for episode in range(num_episodes):
            # 收集数据
            (states, actions, rewards, dones, values, 
             log_probs, episode_reward, episode_length) = self.collect_rollout(
                env, max_steps_per_episode
            )
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # 转换为张量
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            old_log_probs = torch.FloatTensor(log_probs)
            
            # 计算 GAE
            advantages, returns = compute_gae(
                rewards, values[:-1], dones, 
                self.gamma, self.gae_lambda
            )
            advantages = torch.FloatTensor(advantages)
            returns = torch.FloatTensor(returns)
            
            # 归一化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 更新策略
            policy_stats = self.update_policy(
                states, actions, old_log_probs, advantages
            )
            
            # 更新价值网络
            for _ in range(5):
                value_loss = self.update_value(states, returns)
            
            update_stats.append({
                'policy': policy_stats,
                'value_loss': value_loss
            })
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.1f} | "
                      f"Avg: {avg_reward:.1f} | "
                      f"KL: {policy_stats['kl']:.4f} | "
                      f"Entropy: {policy_stats['entropy']:.2f}")
            
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
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    dist = self.policy.get_action_probs(state_tensor)
                    action = dist.sample()
                
                next_state, reward, terminated, truncated, _ = env.step(action.item())
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
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        print(f"模型已从: {path} 加载")


# ============================================================
# 训练脚本
# ============================================================

def train_cartpole():
    """训练 CartPole-v1"""
    print("=" * 60)
    print("TRPO 训练 CartPole-v1")
    print("=" * 60)
    
    trpo = TRPO(
        state_dim=4,
        action_dim=2,
        hidden_dims=[64, 64],
        gamma=0.99,
        gae_lambda=0.95,
        kl_target=0.01,
        max_kl=0.01,
        damping=1e-3
    )
    
    rewards, stats = trpo.train(env_name="CartPole-v1", num_episodes=500)
    
    trpo.save("trpo_cartpole.pth")
    results = trpo.evaluate(env_name="CartPole-v1", num_episodes=5, render=True)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards, stats


def train_lunarlander():
    """训练 LunarLander-v2"""
    print("=" * 60)
    print("TRPO 训练 LunarLander-v2")
    print("=" * 60)
    
    trpo = TRPO(
        state_dim=8,
        action_dim=4,
        hidden_dims=[128, 128],
        gamma=0.99,
        gae_lambda=0.95,
        kl_target=0.005,
        damping=1e-3
    )
    
    rewards, stats = trpo.train(env_name="LunarLander-v2", num_episodes=1000)
    
    trpo.save("trpo_lunarlander.pth")
    results = trpo.evaluate(env_name="LunarLander-v2", num_episodes=5)
    
    print(f"\n最终结果:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    return rewards


if __name__ == "__main__":
    train_cartpole()
