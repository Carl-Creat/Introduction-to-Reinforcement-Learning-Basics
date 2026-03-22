"""
Agent 规划系统

实现多种推理规划范式：
- Chain of Thought (CoT): 链式推理
- Tree of Thought (ToT): 树状推理
- Graph of Thought (GoT): 图状推理
- Agentic RL: 让 Agent 学会规划
"""
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ThoughtType(Enum):
    """思考类型"""
    REASONING = "reasoning"
    ACTION = "action"
    REFLECTION = "reflection"
    PLANNING = "planning"
    EVALUATION = "evaluation"


@dataclass
class Thought:
    """思考节点"""
    content: str
    type: ThoughtType
    value: float = 0.0
    parent: Optional['Thought'] = None
    children: List['Thought'] = None
    depth: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []


class ChainOfThought:
    """
    Chain of Thought (CoT) 推理

    简单的线性推理链：
    问题 -> 思考1 -> 思考2 -> ... -> 答案
    """

    def __init__(self, name: str = "CoT-Agent"):
        self.name = name
        self.thought_chain: List[Thought] = []

    def think(self, problem: str, n_steps: int = 5) -> str:
        """
        CoT 推理

        Args:
            problem: 问题描述
            n_steps: 推理步数

        Returns:
            最终答案
        """
        print(f"\n[CoT] 问题: {problem}")
        self.thought_chain = []

        # 初始思考
        current = Thought(
            content=f"分析问题: {problem}",
            type=ThoughtType.REASONING,
            depth=0
        )
        self.thought_chain.append(current)

        # 逐步推理
        for step in range(n_steps):
            prev = self.thought_chain[-1]

            # 生成下一步思考
            next_content = self._generate_next_thought(prev, step)

            next_thought = Thought(
                content=next_content,
                type=self._classify_thought(next_content),
                depth=step + 1,
                parent=prev
            )
            prev.children.append(next_thought)
            self.thought_chain.append(next_thought)

            print(f"  Step {step+1}: {next_content}")

            # 检查是否得出答案
            if self._is_answer(next_content):
                break

        # 返回最终答案
        final = self.thought_chain[-1].content
        return self._extract_answer(final)

    def _generate_next_thought(self, prev: Thought, step: int) -> str:
        """生成下一步思考"""
        templates = [
            f"基于上一步 '{prev.content[:30]}...'，我现在理解到...",
            f"继续分析，当前推理是: {prev.content[:40]}...",
            f"反思: {prev.content[:30]}... 是否正确？",
            f"综合前面的推理: {prev.content[:30]}...",
            f"得出结论: {prev.content[:30]}...",
        ]
        return templates[min(step, len(templates)-1)]

    def _classify_thought(self, content: str) -> ThoughtType:
        """分类思考类型"""
        if "结论" in content or "答案" in content:
            return ThoughtType.EVALUATION
        elif "反思" in content:
            return ThoughtType.REFLECTION
        elif "行动" in content or "执行" in content:
            return ThoughtType.ACTION
        elif "计划" in content:
            return ThoughtType.PLANNING
        return ThoughtType.REASONING

    def _is_answer(self, content: str) -> bool:
        """检查是否是答案"""
        answer_keywords = ["结论", "答案是", "因此", "最终", "所以"]
        return any(kw in content for kw in answer_keywords)

    def _extract_answer(self, content: str) -> str:
        """提取答案"""
        # 简单提取
        if "结论:" in content:
            return content.split("结论:")[-1].strip()
        return content


class TreeOfThought:
    """
    Tree of Thought (ToT) 推理

    树状搜索推理：
            问题
           /  |  \\
        分支1 分支2 分支3
         |     |     |
       子节点...    ...
         \\     |     /
           评估选择
    """

    def __init__(self, name: str = "ToT-Agent", beam_width: int = 3):
        self.name = name
        self.beam_width = beam_width
        self.root: Optional[Thought] = None

    def think(self, problem: str, depth: int = 3, branching: int = 3) -> str:
        """
        ToT 推理

        Args:
            problem: 问题
            depth: 推理深度
            branching: 每层分支数

        Returns:
            最佳答案
        """
        print(f"\n[ToT] 问题: {problem}")
        print(f"深度={depth}, 分支={branching}")

        # 创建根节点
        self.root = Thought(
            content=problem,
            type=ThoughtType.REASONING,
            depth=0
        )

        # 递归生成树
        candidates = self._expand(self.root, depth, branching)

        # 评估选择最佳路径
        best = self._select_best(candidates)

        # 展示结果
        self._print_path(best)

        return best.content

    def _expand(self, node: Thought, remaining_depth: int,
                branching: int) -> List[Thought]:
        """扩展节点"""
        if remaining_depth == 0:
            return [node]

        candidates = []
        for b in range(branching):
            child_content = f"{node.content[:20]}... -> 思考路径{node.depth+1}-{b+1}"

            child = Thought(
                content=child_content,
                type=ThoughtType.REASONING,
                depth=node.depth + 1,
                parent=node,
                value=np.random.uniform(0.5, 1.0)  # 模拟评估
            )
            node.children.append(child)

            # 递归扩展
            sub_candidates = self._expand(child, remaining_depth - 1, branching)
            candidates.extend(sub_candidates)

        return candidates

    def _select_best(self, candidates: List[Thought]) -> Thought:
        """选择最佳节点"""
        if not candidates:
            return self.root
        # 选择评分最高的
        return max(candidates, key=lambda x: x.value)

    def _print_path(self, node: Thought):
        """打印路径"""
        print("\n[最佳推理路径]")
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent

        path.reverse()
        for node in path:
            print(f"  {'  '*node.depth}[{node.depth}] {node.content[:50]}")


class GraphOfThought:
    """
    Graph of Thought (GoT) 推理

    图状推理，节点间可以有复杂连接
    支持聚合、转换等操作
    """

    def __init__(self, name: str = "GoT-Agent"):
        self.name = name
        self.nodes: Dict[str, Thought] = {}
        self.edges: List[Tuple[str, str]] = []

    def add_node(self, node_id: str, content: str,
                 node_type: ThoughtType = ThoughtType.REASONING):
        """添加节点"""
        self.nodes[node_id] = Thought(
            content=content,
            type=node_type,
            depth=0
        )

    def add_edge(self, from_id: str, to_id: str):
        """添加边"""
        if from_id in self.nodes and to_id in self.nodes:
            self.edges.append((from_id, to_id))
            self.nodes[from_id].children.append(self.nodes[to_id])
            self.nodes[to_id].parent = self.nodes[from_id]

    def aggregate(self, node_ids: List[str], method: str = "merge") -> str:
        """
        聚合多个节点

        Args:
            node_ids: 要聚合的节点ID列表
            method: 聚合方法 ("merge", "vote", "average")

        Returns:
            聚合结果
        """
        contents = [self.nodes[nid].content for nid in node_ids if nid in self.nodes]

        if method == "merge":
            return " + ".join(contents)
        elif method == "vote":
            # 简单投票
            return max(set(contents), key=contents.count)
        return contents[0] if contents else ""

    def transform(self, node_id: str, transform_fn: Callable) -> str:
        """转换节点内容"""
        if node_id in self.nodes:
            original = self.nodes[node_id].content
            transformed = transform_fn(original)
            return transformed
        return ""


class AgenticRL:
    """
    Agentic RL: 会规划的强化学习智能体

    核心思想：
    - 学会将大目标分解为小目标
    - 规划行动序列
    - 从规划失败中学习
    """

    def __init__(self, name: str = "AgenticRL"):
        self.name = name
        self.planning_agent = ChainOfThought(name + "-Planner")
        self.high_level_policy = RLHighLevelPolicy()
        self.low_level_policy = RLLowLevelPolicy()

    def set_goal(self, goal: str):
        """设置目标"""
        print(f"\n[Agentic RL] 设定目标: {goal}")

    def plan(self, goal: str, context: str = "") -> List[str]:
        """
        规划行动序列

        Args:
            goal: 目标
            context: 上下文

        Returns:
            行动计划列表
        """
        print(f"\n[规划阶段] 目标: {goal}")

        # 使用 CoT 规划
        plan_text = self.planning_agent.think(
            f"如何实现目标: {goal}？制定具体行动计划。",
            n_steps=4
        )

        # 分解为具体行动
        actions = self._decompose_to_actions(plan_text)

        print(f"[计划] 分解为 {len(actions)} 个行动:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")

        return actions

    def _decompose_to_actions(self, plan_text: str) -> List[str]:
        """分解为具体行动"""
        # 简单的行动分解（实际用 LLM）
        actions = [
            "分解目标为子目标",
            "评估当前状态与目标差距",
            "选择下一步行动",
            "执行行动并评估结果",
            "根据反馈调整计划"
        ]
        # 根据计划文本调整
        if "探索" in plan_text:
            actions.insert(0, "探索环境")
        if "利用" in plan_text:
            actions.append("执行最优策略")

        return actions[:5]

    def execute_plan(self, plan: List[str]) -> Dict[str, Any]:
        """
        执行计划

        Returns:
            执行结果
        """
        print(f"\n[执行阶段]")

        results = []
        for i, action in enumerate(plan):
            print(f"  执行 {i+1}/{len(plan)}: {action}")

            # 模拟执行
            success = np.random.random() > 0.2  # 80% 成功率
            reward = 1.0 if success else -0.5

            results.append({
                "action": action,
                "success": success,
                "reward": reward
            })

        # 评估结果
        total_reward = sum(r["reward"] for r in results)
        success_rate = sum(1 for r in results if r["success"]) / len(results)

        print(f"\n[结果] 成功率: {success_rate:.1%}, 总奖励: {total_reward:.2f}")

        return {
            "results": results,
            "total_reward": total_reward,
            "success_rate": success_rate
        }

    def learn_from_planning(self, plan: List[str], results: Dict):
        """从规划结果学习"""
        print("\n[学习] 分析规划效果...")

        if results["success_rate"] < 0.5:
            print("  规划效果不佳，尝试改进...")
            # 简单学习逻辑
            for i, result in enumerate(results["results"]):
                if not result["success"]:
                    print(f"  行动 {i+1} 失败，重新规划...")


class RLHighLevelPolicy:
    """RL 高层策略：目标分解"""

    def __init__(self):
        self.value_estimates = {}

    def estimate(self, goal: str) -> float:
        """估计目标价值"""
        return np.random.uniform(0.5, 1.0)


class RLLowLevelPolicy:
    """RL 低层策略：动作执行"""

    def __init__(self):
        self.q_table = {}

    def get_action(self, state: str) -> str:
        """获取动作"""
        actions = ["探索", "利用", "等待", "学习"]
        return np.random.choice(actions)


def demo_cot():
    """CoT 演示"""
    print("=" * 60)
    print("Chain of Thought (CoT) 推理")
    print("=" * 60)

    agent = ChainOfThought("MathAgent")
    result = agent.think("如何用强化学习解决机器人路径规划问题？", n_steps=5)
    print(f"\n最终答案: {result}")


def demo_tot():
    """ToT 演示"""
    print("\n" + "=" * 60)
    print("Tree of Thought (ToT) 推理")
    print("=" * 60)

    agent = TreeOfThought("ProblemSolver")
    result = agent.think("强化学习中如何平衡探索与利用？", depth=3, branching=2)
    print(f"\n最佳答案: {result}")


def demo_agentic_rl():
    """Agentic RL 演示"""
    print("\n" + "=" * 60)
    print("Agentic RL: 会规划的智能体")
    print("=" * 60)

    agent = AgenticRL()

    # 设置目标
    agent.set_goal("在 CartPole 环境中获得 500 分")

    # 规划
    plan = agent.plan("在 CartPole 环境中获得 500 分")

    # 执行
    results = agent.execute_plan(plan)

    # 学习
    agent.learn_from_planning(plan, results)


if __name__ == "__main__":
    demo_cot()
    demo_tot()
    demo_agentic_rl()
