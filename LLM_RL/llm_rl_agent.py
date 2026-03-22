"""
LLM 驱动的强化学习智能体

将大语言模型作为 RL Agent 的决策大脑，
结合环境反馈进行推理和决策。

核心思路：
- 用 LLM 生成候选动作
- 环境反馈 + 奖励信号
- LLM 反思和自我改进
"""
import random
from typing import List, Dict, Optional, Callable, Any


class LLMRLAgent:
    """
    LLM 驱动的 RL Agent
    
    用 LLM 替代传统策略网络，实现：
    - 零样本泛化
    - 链式推理
    - 自我反思和调整
    
    Attributes:
        name: Agent 名称
        model: LLM 模型名称
        temperature: 生成温度
        memory: 记忆列表
    """
    
    def __init__(
        self,
        name: str = "LLM-Agent",
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_memory: int = 100
    ):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_memory = max_memory
        self.memory: List[Dict[str, str]] = []
        self.total_reward = 0
        self.episode_count = 0
        
    def think(self, state: Any, available_actions: List[str], 
              context: str = "") -> str:
        """
        LLM 推理选择动作
        
        Args:
            state: 当前环境状态
            available_actions: 可选动作列表
            context: 额外上下文信息
        
        Returns:
            选择的动作
        """
        # 构建推理 prompt
        prompt = self._build_prompt(state, available_actions, context)
        
        # 模拟 LLM 推理（在实际使用时调用真实 LLM API）
        reasoning = self._simulate_llm_reasoning(state, available_actions)
        
        # 记录到记忆
        self.memory.append({
            "state": str(state)[:100],
            "actions": available_actions,
            "reasoning": reasoning,
            "context": context
        })
        
        # 保持记忆长度
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        
        return reasoning
    
    def _build_prompt(self, state: Any, actions: List[str], 
                      context: str) -> str:
        """构建推理 prompt"""
        prompt = f"""你是{self.name}，一个强化学习智能体。
当前状态: {state}
可用动作: {', '.join(actions)}
{context}

请选择最佳动作并解释你的推理过程。"""
        return prompt
    
    def _simulate_llm_reasoning(self, state: Any, 
                                 actions: List[str]) -> str:
        """
        模拟 LLM 推理
        
        在实际使用中，这里应该调用真实的 LLM API
        """
        # 简单的启发式选择（模拟 LLM 决策）
        if isinstance(state, (int, float)):
            # 数值型状态：根据状态值选择
            if state > 0:
                return actions[0] if len(actions) > 0 else "no-op"
            else:
                return actions[-1] if len(actions) > 0 else "no-op"
        else:
            # 随机选择
            return random.choice(actions) if actions else "no-op"
    
    def reflect(self, state: Any, action: str, reward: float, 
                next_state: Any, done: bool) -> str:
        """
        LLM 反思：根据奖励进行自我调整
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        
        Returns:
            反思结果
        """
        self.total_reward += reward
        
        reflection = f"动作 {action} 获得奖励 {reward:.2f}"
        
        if done:
            self.episode_count += 1
            avg_reward = self.total_reward / max(1, self.episode_count)
            reflection += f"，本轮平均奖励: {avg_reward:.2f}"
            self.total_reward = 0
        
        return reflection
    
    def get_memory_context(self, recent_k: int = 5) -> str:
        """
        获取最近的记忆作为上下文
        
        Args:
            recent_k: 获取最近 k 条记忆
        
        Returns:
            记忆上下文字符串
        """
        recent = self.memory[-recent_k:] if self.memory else []
        if not recent:
            return "暂无历史记忆"
        
        context = "最近的经验：\n"
        for i, m in enumerate(recent, 1):
            context += f"{i}. 状态: {m['state']}, 推理: {m['reasoning']}\n"
        return context


# ============== RLHF: 奖励塑形 ==============

class RewardShaper:
    """
    奖励塑形器
    
    将稀疏奖励转换为密集奖励，引导 Agent 学习
    
    奖励类型：
    - 稀疏奖励：只在任务完成时给奖励
    - 密集奖励：通过奖励塑形提供即时反馈
    """
    
    def __init__(self, base_reward_fn: Callable, shaping_fn: Optional[Callable] = None):
        """
        Args:
            base_reward_fn: 基础奖励函数
            shaping_fn: 塑形奖励函数（可选）
        """
        self.base_reward_fn = base_reward_fn
        self.shaping_fn = shaping_fn or self._default_shaping
    
    def get_reward(self, state: Any, action: str, 
                   next_state: Any, done: bool, **kwargs) -> float:
        """
        计算综合奖励
        
        Returns:
            total_reward = base_reward + shaping_reward
        """
        base = self.base_reward_fn(state, action, next_state, done, **kwargs)
        shaped = self.shaping_fn(state, action, next_state, done, **kwargs)
        return base + shaped
    
    def _default_shaping(self, state: Any, action: str,
                         next_state: Any, done: bool, **kwargs) -> float:
        """默认塑形：给予接近目标的奖励"""
        shaping_reward = 0.0
        
        if isinstance(state, (int, float)) and isinstance(next_state, (int, float)):
            # 鼓励数值增加
            delta = next_state - state
            shaping_reward = 0.01 * delta
        
        return shaping_reward


class RLHFRewardModel:
    """
    RLHF 奖励模型
    
    用人类偏好数据训练奖励模型，
    然后用该模型生成奖励信号
    
    流程：
    1. 收集人类偏好数据
    2. 训练奖励模型
    3. 用奖励模型生成奖励
    4. 用 PPO 等算法优化策略
    """
    
    def __init__(self, model: str = "reward-model"):
        self.model_name = model
        self.preferences: List[Dict[str, Any]] = []
        self.is_trained = False
    
    def add_preference(self, trajectory_a: List, trajectory_b: List, 
                       preference: str):
        """
        添加人类偏好数据
        
        Args:
            trajectory_a: 轨迹 A
            trajectory_b: 轨迹 B
            preference: 偏好："A" 或 "B"
        """
        self.preferences.append({
            "trajectory_a": trajectory_a,
            "trajectory_b": trajectory_b,
            "preference": preference
        })
    
    def train(self):
        """训练奖励模型"""
        if len(self.preferences) < 10:
            print("Warning: 偏好数据不足，建议收集更多数据")
        
        # 模拟训练过程
        self.is_trained = True
        print(f"奖励模型训练完成，使用了 {len(self.preferences)} 条偏好数据")
    
    def predict_reward(self, trajectory: List) -> float:
        """
        预测轨迹的奖励
        
        Args:
            trajectory: 轨迹列表
        
        Returns:
            预测的奖励分数
        """
        if not self.is_trained:
            return 0.0
        
        # 简单的奖励预测（实际应用中用神经网络）
        return sum(step.get("reward", 0) for step in trajectory)
    
    def get_comparison_reward(self, trajectory_a: List, 
                               trajectory_b: List) -> tuple:
        """
        比较两条轨迹的奖励差异
        
        Returns:
            (reward_a, reward_b, preference)
        """
        r_a = self.predict_reward(trajectory_a)
        r_b = self.predict_reward(trajectory_b)
        pref = "A" if r_a > r_b else "B" if r_b > r_a else "equal"
        return r_a, r_b, pref


# ============== 演示示例 ==============

def demo_llm_rl_agent():
    """演示 LLM 驱动的 RL Agent"""
    print("=" * 50)
    print("LLM 驱动的 RL Agent 演示")
    print("=" * 50)
    
    # 创建 Agent
    agent = LLMRLAgent(name="ShoppingBot", model="gpt-4")
    
    # 模拟对话场景
    actions = ["推荐", "搜索", "比较", "下单", "取消"]
    
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        
        # 模拟状态
        state = random.randint(0, 100)
        
        # LLM 推理选择动作
        context = agent.get_memory_context(recent_k=2)
        action = agent.think(state, actions, context=context)
        print(f"状态: {state} -> 选择动作: {action}")
        
        # 模拟奖励
        reward = random.uniform(-1, 1)
        next_state = state + random.randint(-5, 5)
        done = episode == 2
        
        # LLM 反思
        reflection = agent.reflect(state, action, reward, next_state, done)
        print(f"奖励: {reward:.2f} -> {reflection}")
    
    print("\n--- Agent 记忆统计 ---")
    print(f"总记忆条数: {len(agent.memory)}")
    print(f"完成轮数: {agent.episode_count}")


def demo_reward_shaping():
    """演示奖励塑形"""
    print("\n" + "=" * 50)
    print("奖励塑形演示")
    print("=" * 50)
    
    # 定义基础奖励函数
    def base_reward(state, action, next_state, done, **kwargs):
        if done:
            return 10.0  # 完成任务的奖励
        return 0.0
    
    shaper = RewardShaper(base_reward)
    
    # 模拟奖励计算
    states = [0, 5, 10, 15, 20]
    for i in range(len(states) - 1):
        state = states[i]
        next_state = states[i + 1]
        total = shaper.get_reward(state, "move", next_state, False)
        base = shaper.base_reward_fn(state, "move", next_state, False)
        shaped = shaper.shaping_fn(state, "move", next_state, False)
        print(f"状态 {state} -> {next_state}: "
              f"总奖励={total:.3f} (基础={base:.1f}, 塑形={shaped:.3f})")


if __name__ == "__main__":
    demo_llm_rl_agent()
    demo_reward_shaping()
