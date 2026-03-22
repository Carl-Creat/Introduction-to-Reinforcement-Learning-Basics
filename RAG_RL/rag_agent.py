"""
RAG + RL: 检索增强的强化学习决策系统

将 RAG (Retrieval-Augmented Generation) 与 RL 结合：
- 从知识库检索相关信息
- 结合检索结果进行 RL 决策
- 实现更智能的 Agent
"""
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class Document:
    """文档单元"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorStore:
    """向量知识库"""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []

    def add(self, doc: Document):
        """添加文档"""
        self.documents.append(doc)
        if doc.embedding is not None:
            self.embeddings.append(doc.embedding)
        else:
            # 简单文本嵌入（模拟）
            emb = self._simple_embed(doc.content)
            self.embeddings.append(emb)

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        相似性检索

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            (文档, 相似度分数) 列表
        """
        query_emb = self._simple_embed(query)

        # 计算余弦相似度
        similarities = []
        for emb in self.embeddings:
            sim = self._cosine_sim(query_emb, emb)
            similarities.append(sim)

        # 排序
        indexed = list(enumerate(similarities))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed[:k]:
            results.append((self.documents[idx], score))

        return results

    def _simple_embed(self, text: str) -> np.ndarray:
        """简单文本嵌入（实际应用中用 BERT 等模型）"""
        # 基于词频的简单嵌入
        words = text.lower().split()
        emb = np.random.randn(self.embedding_dim)
        for i, word in enumerate(words[:10]):  # 取前10个词
            emb[i % self.embedding_dim] += hash(word) % 100 / 100.0
        return emb / (np.linalg.norm(emb) + 1e-8)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度"""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / (norm + 1e-8))


class KnowledgeBase:
    """领域知识库"""

    def __init__(self):
        self.vector_store = VectorStore()

    def build_from_texts(self, texts: List[str]):
        """从文本构建知识库"""
        for i, text in enumerate(texts):
            doc = Document(
                id=f"doc_{i}",
                content=text,
                metadata={"source": "manual"}
            )
            self.vector_store.add(doc)

    def query(self, question: str, k: int = 3) -> List[str]:
        """
        查询知识库

        Args:
            question: 问题
            k: 返回结果数

        Returns:
            相关文档内容列表
        """
        results = self.vector_store.search(question, k)
        return [doc.content for doc, score in results if score > 0.1]


class RLDecisionMaker:
    """
    RL 决策器

    基于状态和检索结果进行 RL 决策
    """

    def __init__(self, n_actions: int = 5):
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)
        self.learning_rate = 0.1
        self.gamma = 0.95

    def decide(self, state: Any, retrieved_knowledge: List[str]) -> int:
        """
        做出决策

        结合状态和检索到的知识选择动作

        Args:
            state: 当前状态
            retrieved_knowledge: 检索到的知识

        Returns:
            选择的动作索引
        """
        # 根据检索到的知识调整 Q 值
        knowledge_bonus = np.zeros(self.n_actions)

        for knowledge in retrieved_knowledge:
            # 简单的知识增强（实际应用中更复杂）
            if "高奖励" in knowledge or "好" in knowledge:
                knowledge_bonus += 0.2
            elif "危险" in knowledge or "避免" in knowledge:
                knowledge_bonus -= 0.3

        # 综合 Q 值和知识奖励
        final_scores = self.q_values + knowledge_bonus

        # epsilon-greedy
        if np.random.random() < 0.1:
            return np.random.randint(self.n_actions)

        return int(np.argmax(final_scores))

    def update(self, state: Any, action: int, reward: float,
               next_state: Any, done: bool):
        """更新 Q 值"""
        # 简单更新
        best_next_q = np.max(self.q_values) if not done else 0
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_values[action]
        self.q_values[action] += self.learning_rate * td_error


class RAGRLAgent:
    """
    RAG + RL 智能体

    结合知识检索和强化学习决策
    """

    def __init__(self, name: str = "RAG-RL-Agent"):
        self.name = name
        self.knowledge_base = KnowledgeBase()
        self.decision_maker = RLDecisionMaker(n_actions=5)
        self.action_names = ["探索", "利用", "等待", "学习", "决策"]
        self.episode_count = 0
        self.total_reward = 0

    def build_knowledge_base(self):
        """构建知识库"""
        # 强化学习知识
        rl_knowledge = [
            "当状态价值高时，选择利用策略可以获得更高奖励",
            "当环境不确定时，探索策略有助于发现新知识",
            "高奖励状态通常意味着更接近目标",
            "避免进入危险状态可以减少负奖励",
            "延迟满足通常能获得更大奖励",
            "在复杂环境中，多步规划比即时决策更好",
            "记忆重要经验可以帮助未来做出更好决策",
            "平衡探索与利用是 RL 的核心挑战"
        ]
        self.knowledge_base.build_from_texts(rl_knowledge)

    def perceive(self, state: Any, question: str = "") -> Dict:
        """
        感知：结合状态和知识

        Args:
            state: 当前状态
            question: 可选的查询

        Returns:
            综合感知结果
        """
        # 检索相关知识
        if question:
            retrieved = self.knowledge_base.query(question)
        else:
            retrieved = self.knowledge_base.query(str(state))

        # 决策
        action_idx = self.decision_maker.decide(state, retrieved)
        action_name = self.action_names[action_idx]

        return {
            "state": state,
            "retrieved_knowledge": retrieved,
            "action": action_name,
            "action_idx": action_idx
        }

    def act(self, perception: Dict, reward: float, next_state: Any,
            done: bool) -> None:
        """
        行动并学习

        Args:
            perception: 感知结果
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        # 更新决策器
        self.decision_maker.update(
            perception["state"],
            perception["action_idx"],
            reward,
            next_state,
            done
        )

        # 记录
        self.total_reward += reward
        if done:
            self.episode_count += 1
            avg_reward = self.total_reward / max(1, self.episode_count)
            print(f"Episode {self.episode_count} 完成: "
                  f"本轮奖励={reward:.2f}, 平均奖励={avg_reward:.2f}")
            self.total_reward = 0


class RLHFRAGAgent:
    """
    RLHF + RAG 智能体

    用人类反馈微调 RAG + RL 系统
    """

    def __init__(self):
        self.rag_rl_agent = RAGRLAgent()
        self.feedback_history: List[Dict] = []

    def add_feedback(self, action: str, quality: str, comment: str = ""):
        """
        添加人类反馈

        Args:
            action: 采取的行动
            quality: 质量评级 "good" / "bad" / "neutral"
            comment: 额外评论
        """
        reward = {"good": 1.0, "neutral": 0.0, "bad": -1.0}.get(quality, 0.0)
        self.feedback_history.append({
            "action": action,
            "quality": quality,
            "reward": reward,
            "comment": comment
        })

    def adjust_knowledge_base(self):
        """
        根据反馈调整知识库

        从人类反馈中学习，更新检索结果的权重
        """
        if not self.feedback_history:
            return

        # 统计反馈
        action_quality = {}
        for fb in self.feedback_history[-20:]:  # 最近20条
            action = fb["action"]
            if action not in action_quality:
                action_quality[action] = []
            action_quality[action].append(fb["reward"])

        # 更新决策器（模拟）
        print("\n[知识调整]")
        for action, rewards in action_quality.items():
            avg = np.mean(rewards)
            print(f"  {action}: 平均反馈={avg:.2f}, 次数={len(rewards)}")


def demo_rag_rl():
    """RAG + RL 演示"""
    print("=" * 60)
    print("RAG + RL: 检索增强的强化学习决策系统")
    print("=" * 60)

    agent = RAGRLAgent(name="KnowledgeableAgent")
    agent.build_knowledge_base()

    print("\n--- 模拟决策过程 ---")
    states = ["状态A(高价值)", "状态B(不确定)", "状态C(接近目标)"]

    for i, state in enumerate(states):
        print(f"\n[Step {i+1}]")
        print(f"当前状态: {state}")

        # 感知
        perception = agent.perceive(state, question="如何获得高奖励")
        print(f"检索到知识: {perception['retrieved_knowledge'][:1]}")
        print(f"决定行动: {perception['action']}")

        # 模拟奖励
        reward = np.random.randn()
        done = (i == len(states) - 1)

        # 行动学习
        agent.act(perception, reward, states[(i+1) % len(states)], done)

    print("\n--- Q 值学习结果 ---")
    for i, name in enumerate(agent.action_names):
        print(f"  {name}: {agent.decision_maker.q_values[i]:.3f}")


def demo_rlhf_rag():
    """RLHF + RAG 演示"""
    print("\n" + "=" * 60)
    print("RLHF + RAG: 人类反馈增强的系统")
    print("=" * 60)

    agent = RLHFRAGAgent()
    agent.rag_rl_agent.build_knowledge_base()

    # 模拟人类反馈
    print("\n--- 模拟人类反馈 ---")
    feedbacks = [
        ("探索", "good", "好的探索策略"),
        ("利用", "bad", "太早了，应该探索"),
        ("学习", "good", "继续学习"),
    ]

    for action, quality, comment in feedbacks:
        agent.add_feedback(action, quality, comment)
        print(f"反馈: {action} -> {quality}")

    # 调整知识库
    agent.adjust_knowledge_base()


if __name__ == "__main__":
    demo_rag_rl()
    demo_rlhf_rag()
