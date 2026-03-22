"""
多智能体强化学习系统

支持协作型、竞争型和混合型多智能体场景。

核心概念：
- MADDPG (Multi-Agent DDPG)
- QMIX (Value Decomposition)
- 通信与协作机制
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """智能体类型"""
    COOPERATIVE = "cooperative"  # 协作型
    COMPETITIVE = "competitive"  # 竞争型
    MIXED = "mixed"             # 混合型


@dataclass
class Agent:
    """智能体基类"""
    id: int
    name: str
    obs_dim: int
    action_dim: int
    agent_type: AgentType = AgentType.COOPERATIVE
    
    def policy(self, obs: np.ndarray) -> np.ndarray:
        """策略函数"""
        raise NotImplementedError
    
    def update(self, *args, **kwargs):
        """更新函数"""
        raise NotImplementedError


class CentralizedTrainer:
    """
    集中式训练器
    
    训练时使用全局信息，
    执行时只使用局部观测
    """
    
    def __init__(self, agents: List[Agent], env_fn=None):
        self.agents = agents
        self.env_fn = env_fn
        self.total_steps = 0
        self.episode_rewards: List[float] = []
    
    def collect_experience(self, env, n_steps: int = 100) -> List[Dict]:
        """
        收集经验
        
        Args:
            env: 环境
            n_steps: 收集步数
        
        Returns:
            经验列表
        """
        experiences = []
        
        for _ in range(n_steps):
            # 收集所有智能体的观测
            observations = []
            for agent in self.agents:
                # 模拟观测（实际从环境获取）
                obs = np.random.randn(agent.obs_dim)
                observations.append(obs)
            
            # 所有智能体选择动作
            actions = []
            for agent, obs in zip(self.agents, observations):
                action = agent.policy(obs)
                actions.append(action)
            
            # 环境执行（模拟）
            rewards = [np.random.randn() for _ in self.agents]
            dones = [False] * len(self.agents)
            
            experiences.append({
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": dones
            })
            
            self.total_steps += 1
        
        return experiences
    
    def update(self, experiences: List[Dict]):
        """更新所有智能体"""
        for agent in self.agents:
            # 模拟更新
            agent.update(experiences)
    
    def train(self, n_episodes: int = 1000):
        """
        训练循环
        
        Args:
            n_episodes: 训练轮数
        """
        print(f"开始训练 {n_episodes} 轮...")
        
        for episode in range(n_episodes):
            episode_reward = 0
            
            # 收集经验
            if self.env_fn:
                env = self.env_fn()
            else:
                # 模拟环境
                experiences = self.collect_experience(None, n_steps=100)
            
            # 更新
            self.update(experiences)
            
            # 记录
            if episode % 100 == 0:
                print(f"Episode {episode}: 平均奖励 = {episode_reward:.2f}")
        
        print("训练完成!")


class QMIXAgent:
    """
    QMIX 算法实现
    
    QMIX 使用混合网络（Mixing Network）
    将个体 Q 值合并为全局 Q 值，
    实现中心化训练+去中心化执行
    """
    
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 每个智能体的网络
        self.agent_networks = [
            self._build_agent_network() 
            for _ in range(n_agents)
        ]
        
        # 混合网络
        self.mixer_network = self._build_mixer_network()
        
        # 超参数
        self.gamma = 0.99      # 折扣因子
        self.lr = 0.0005       # 学习率
        self.target_update_freq = 200
    
    def _build_agent_network(self) -> dict:
        """构建单个智能体的网络"""
        return {
            "obs_layers": [128, 128],
            "action_dim": self.action_dim
        }
    
    def _build_mixer_network(self) -> dict:
        """构建混合网络"""
        return {
            "hyper_w1": [64, self.n_agents * 32],
            "hyper_b1": [32],
            "hyper_w2": [32, 32],
            "hyper_b2": [1]
        }
    
    def get_q_values(self, observations: List[np.ndarray]) -> List[np.ndarray]:
        """
        计算所有智能体的 Q 值
        
        Args:
            observations: 各智能体的观测列表
        
        Returns:
            Q 值列表
        """
        q_values = []
        for i, obs in enumerate(observations):
            # 模拟 Q 值计算
            q = np.random.randn(self.action_dim)
            q_values.append(q)
        return q_values
    
    def mix_q_values(self, q_values: List[np.ndarray], 
                     states: np.ndarray) -> np.ndarray:
        """
        混合 Q 值
        
        将个体 Q 值通过混合网络合并为全局 Q 值
        
        Args:
            q_values: 个体 Q 值列表
            states: 全局状态
        
        Returns:
            全局 Q 值
        """
        # 拼接个体 Q 值
        q_concat = np.concatenate(q_values)
        
        # 模拟混合网络计算
        global_q = np.sum(q_concat) / len(q_values)
        
        return np.array([global_q] * len(q_values[0]))
    
    def update(self, batch: Dict):
        """
        从经验回放更新
        
        Args:
            batch: 经验批次
        """
        # 提取数据
        observations = batch["observations"]  # [batch, n_agents, obs_dim]
        actions = batch["actions"]            # [batch, n_agents]
        rewards = batch["rewards"]             # [batch]
        next_states = batch["next_states"]    # [batch, state_dim]
        dones = batch["dones"]                # [batch]
        
        # 计算当前 Q 值
        current_q_list = self.get_q_values([observations[0]])
        
        # 混合
        current_q_mixed = self.mix_q_values(
            [q[actions[0][i]] for i, q in enumerate(current_q_list)],
            states=next_states[0:1]
        )
        
        # 计算目标 Q 值（简化版）
        with np.no_grad():
            next_q_values = self.get_q_values([next_states[0]])
            next_q_mixed = self.mix_q_values(
                [np.max(q) for q in next_q_values],
                states=next_states[0:1]
            )
            target_q = rewards[0] + self.gamma * (1 - dones[0]) * next_q_mixed[0]
        
        # 打印更新信息
        print(f"Q值更新: {current_q_mixed[0]:.3f} -> {target_q:.3f}")


class CooperativeAgent(Agent):
    """协作型智能体"""
    
    def __init__(self, agent_id: int, name: str, obs_dim: int, action_dim: int):
        super().__init__(agent_id, name, obs_dim, action_dim, AgentType.COOPERATIVE)
        self.shared_rewards: List[float] = []
    
    def policy(self, obs: np.ndarray) -> np.ndarray:
        """
        协作策略：考虑其他智能体
        
        协作智能体的策略不仅考虑自身观测，
        还会考虑团队整体利益
        """
        # 添加一点探索
        action = np.random.randn(self.action_dim)
        # 倾向于选择有利于团队的动作
        action = np.tanh(action) * 0.5
        return action
    
    def update(self, experiences: List[Dict]):
        """协作更新：使用团队奖励"""
        team_reward = sum(exp["rewards"][self.id] for exp in experiences)
        self.shared_rewards.append(team_reward)
        
        # 协作更新逻辑
        print(f"Agent {self.name}: 团队奖励 = {team_reward:.3f}")


class CompetitiveAgent(Agent):
    """竞争型智能体"""
    
    def __init__(self, agent_id: int, name: str, obs_dim: int, action_dim: int):
        super().__init__(agent_id, name, obs_dim, action_dim, AgentType.COMPETITIVE)
        self.wins = 0
        self.losses = 0
    
    def policy(self, obs: np.ndarray) -> np.ndarray:
        """
        竞争策略：最大化自身利益
        
        竞争智能体的目标是最大化自身奖励，
        即使这意味着降低对手的奖励
        """
        # 贪婪策略
        action = np.random.randn(self.action_dim) * 0.3
        return action
    
    def update(self, experiences: List[Dict]):
        """竞争更新：基于胜负"""
        total_reward = sum(exp["rewards"][self.id] for exp in experiences)
        
        if total_reward > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        print(f"Agent {self.name}: 奖励 = {total_reward:.3f}, "
              f"战绩 = {self.wins}W/{self.losses}L")


class CommunicationProtocol:
    """
    智能体通信协议
    
    智能体之间可以通信共享信息
    """
    
    def __init__(self, message_dim: int = 32):
        self.message_dim = message_dim
        self.message_history: List[Dict[int, np.ndarray]] = []
    
    def send_message(self, sender_id: int, content: np.ndarray):
        """
        发送消息
        
        Args:
            sender_id: 发送者 ID
            content: 消息内容
        """
        self.message_history.append({sender_id: content})
    
    def receive_messages(self, receiver_id: int, 
                        all_agents: List[Agent]) -> List[np.ndarray]:
        """
        接收消息
        
        Args:
            receiver_id: 接收者 ID
            all_agents: 所有智能体
        
        Returns:
            接收到的消息列表
        """
        messages = []
        for msg_dict in self.message_history[-5:]:  # 最近5条消息
            for agent_id, content in msg_dict.items():
                if agent_id != receiver_id:
                    messages.append(content)
        return messages
    
    def clear_history(self):
        """清空历史消息"""
        self.message_history = []


# ============== 演示 ==============

def demo_multi_agent():
    """多智能体演示"""
    print("=" * 50)
    print("多智能体强化学习演示")
    print("=" * 50)
    
    # 创建智能体
    agents = [
        CooperativeAgent(0, "Ally1", obs_dim=10, action_dim=5),
        CooperativeAgent(1, "Ally2", obs_dim=10, action_dim=5),
        CompetitiveAgent(2, "Opponent", obs_dim=10, action_dim=5),
    ]
    
    print(f"\n创建了 {len(agents)} 个智能体:")
    for agent in agents:
        print(f"  - {agent.name}: {agent.agent_type.value}")
    
    # 模拟训练
    print("\n--- 模拟训练过程 ---")
    trainer = CentralizedTrainer(agents)
    
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        
        # 收集经验
        experiences = trainer.collect_experience(None, n_steps=5)
        
        # 更新
        for agent in agents:
            agent.update(experiences)
    
    # 通信演示
    print("\n--- 通信协议演示 ---")
    comm = CommunicationProtocol(message_dim=8)
    
    # Agent 0 发送消息
    msg = np.random.randn(8)
    comm.send_message(0, msg)
    print(f"Agent 0 发送消息: {msg[:3]}...")
    
    # Agent 1 接收消息
    received = comm.receive_messages(1, agents)
    print(f"Agent 1 收到 {len(received)} 条消息")


def demo_qmix():
    """QMIX 算法演示"""
    print("\n" + "=" * 50)
    print("QMIX 算法演示")
    print("=" * 50)
    
    # 创建 QMIX 智能体
    qmix = QMIXAgent(n_agents=3, obs_dim=10, action_dim=5)
    
    print(f"智能体数量: {qmix.n_agents}")
    print(f"观测维度: {qmix.obs_dim}")
    print(f"动作维度: {qmix.action_dim}")
    
    # 模拟更新
    print("\n--- 模拟更新 ---")
    batch = {
        "observations": [np.random.randn(3, 10) for _ in range(4)],
        "actions": np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 1, 2]]),
        "rewards": np.random.randn(4),
        "next_states": np.random.randn(4, 32),
        "dones": np.array([False, False, False, True])
    }
    
    qmix.update(batch)


if __name__ == "__main__":
    demo_multi_agent()
    demo_qmix()
