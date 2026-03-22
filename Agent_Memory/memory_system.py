"""
Agent 记忆系统

实现三种记忆类型：
- 情景记忆（Episodic Memory）：存储具体经验
- 语义记忆（Semantic Memory）：存储抽象知识
- 工作记忆（Working Memory）：当前任务的临时存储

核心思想：
记忆 -> 检索 -> 推理 -> 行动
"""
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class Experience:
    """经验单元（用于情景记忆）"""
    state: Any
    action: Any
    reward: float
    next_state: Any
    observation: str = ""
    reflection: str = ""
    timestamp: float = field(default_factory=time.time)


class EpisodicMemory:
    """情景记忆：存储 Agent 的具体经历经验"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.experiences: List[Experience] = []

    def store(self, experience: Experience):
        self.experiences.append(experience)
        if len(self.experiences) > self.max_size:
            self.experiences.pop(0)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[Experience, float]]:
        """基于关键词检索相关经验"""
        results = []
        for exp in reversed(self.experiences[-100:]):
            score = 0.0
            if query.lower() in str(exp.action).lower():
                score += 0.5
            if query.lower() in str(exp.observation).lower():
                score += 0.3
            if abs(exp.reward) > 0.5:
                score += 0.2
            if score > 0:
                results.append((exp, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_recent(self, n: int = 5) -> List[Experience]:
        return self.experiences[-n:]


class SemanticMemory:
    """语义记忆：存储抽象知识和概念"""

    def __init__(self):
        self.knowledge: Dict[str, Dict] = {}

    def store_fact(self, subject: str, predicate: str, obj: Any):
        key = f"{subject}::{predicate}"
        self.knowledge[key] = {"subject": subject, "predicate": predicate, "object": obj}

    def retrieve(self, subject: str = None) -> List[Dict]:
        if subject:
            return [v for k, v in self.knowledge.items() if subject.lower() in k.lower()]
        return list(self.knowledge.values())

    def learn_rule(self, condition: str, conclusion: str, confidence: float = 1.0):
        self.store_fact("Rule", condition, {"conclusion": conclusion, "confidence": confidence})

    def query(self, query: str) -> Optional[Any]:
        for k, v in self.knowledge.items():
            if query.lower() in k.lower():
                return v["object"]
        return None


class WorkingMemory:
    """工作记忆：当前任务的临时存储"""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[Any] = []

    def store(self, item: Any):
        if item in self.items:
            self.items.remove(item)
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(item)

    def retrieve(self, query: str) -> Optional[Any]:
        for item in reversed(self.items):
            if query.lower() in str(item).lower():
                return item
        return None

    def clear(self):
        self.items = []

    def get_all(self) -> List[Any]:
        return list(self.items)


class AgentMemorySystem:
    """Agent 综合记忆系统"""

    def __init__(self):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.working = WorkingMemory()
        self._init_knowledge()

    def _init_knowledge(self):
        """初始化基础知识"""
        self.semantic.store_fact("RL", "核心概念", "状态、动作、奖励")
        self.semantic.store_fact("MDP", "组成", "状态空间、动作空间、转移概率、奖励函数")
        self.semantic.store_fact("Q-Learning", "类型", "值迭代算法")
        self.semantic.store_fact("Policy Gradient", "类型", "策略迭代算法")
        self.semantic.learn_rule("探索充分", "收敛概率高", 0.9)
        self.semantic.learn_rule("奖励稀疏", "学习困难", 0.8)

    def remember(self, state: Any, action: Any, reward: float, next_state: Any):
        exp = Experience(state=state, action=action, reward=reward, next_state=next_state)
        self.episodic.store(exp)
        self.working.store(f"{action} -> {reward:.2f}")

    def recall(self, query: str, k: int = 3) -> List[Any]:
        working = self.working.retrieve(query)
        episodic = self.episodic.retrieve(query, k)
        semantic = self.semantic.query(query)
        results = []
        if working:
            results.append(f"工作记忆: {working}")
        for exp, score in episodic:
            results.append(f"经验: {exp.action} (奖励={exp.reward:.2f}, 相似度={score:.2f})")
        if semantic:
            results.append(f"知识: {semantic}")
        return results

    def get_context(self) -> str:
        """获取当前上下文"""
        recent = self.episodic.get_recent(3)
        working_items = self.working.get_all()
        context = "当前上下文:\n"
        if working_items:
            context += f"  工作记忆: {', '.join(str(i) for i in working_items)}\n"
        if recent:
            context += "  最近经验:\n"
            for e in recent:
                context += f"    - {e.action}: 奖励={e.reward:.2f}\n"
        return context


def demo():
    print("=" * 50)
    print("Agent 记忆系统演示")
    print("=" * 50)

    memory = AgentMemorySystem()

    # 存储经验
    print("\n--- 存储经验 ---")
    for i in range(5):
        state = f"state_{i}"
        action = random.choice(["探索", "利用", "学习", "决策"])
        reward = random.uniform(-1, 1)
        memory.remember(state, action, reward, f"next_{state}")
        print(f"记住: {action} -> 奖励={reward:.2f}")

    # 检索记忆
    print("\n--- 检索 '奖励' 相关 ---")
    results = memory.recall("奖励")
    for r in results:
        print(f"  {r}")

    # 查询知识
    print("\n--- 查询知识库 'RL' ---")
    knowledge = memory.semantic.retrieve("RL")
    for k in knowledge:
        print(f"  {k['subject']}: {k['object']}")

    # 获取上下文
    print("\n--- 当前上下文 ---")
    print(memory.get_context())


if __name__ == "__main__":
    import random
    demo()
