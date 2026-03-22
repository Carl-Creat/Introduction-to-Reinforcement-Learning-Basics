"""
ReAct Agent: 推理 + 行动

实现 ReAct (Reasoning + Acting) 范式：
Thought -> Action -> Observation -> ...

核心思想：
让 Agent 在采取行动前先进行推理，
同时能够使用外部工具
"""
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re


class ActionType(Enum):
    """行动类型"""
    SEARCH = "search"
    CALCULATE = "calculate"
    RETRIEVE = "retrieve"
    MOVE = "move"
    COMMUNICATE = "communicate"
    CUSTOM = "custom"


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    func: Callable
    input_schema: Dict = None
    output_schema: Dict = None

    def __post_init__(self):
        if self.input_schema is None:
            self.input_schema = {}
        if self.output_schema is None:
            self.output_schema = {}


@dataclass
class Step:
    """推理步骤"""
    thought: str
    action: str
    action_input: Any
    observation: Any
    step_num: int


class ToolRegistry:
    """工具注册器"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())


class ReActAgent:
    """
    ReAct Agent 实现

    核心循环：
    1. Thought: 分析当前状态，决定下一步行动
    2. Action: 执行行动
    3. Observation: 观察结果
    4. 重复直到完成
    """

    def __init__(self, name: str = "ReAct-Agent", max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps
        self.tools = ToolRegistry()
        self.history: List[Step] = []
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""

        def search(query: str) -> str:
            """搜索工具"""
            return f"搜索结果: 关于 '{query}' 的信息... (模拟)"

        def calculate(expr: str) -> float:
            """计算工具"""
            try:
                # 安全计算（仅支持基本运算）
                allowed = set("0123456789+-*/.() ")
                if all(c in allowed for c in expr):
                    result = eval(expr)
                    return float(result)
                return 0.0
            except:
                return 0.0

        def retrieve(keyword: str) -> str:
            """知识检索工具"""
            knowledge = {
                "强化学习": "强化学习是机器学习的一个分支，研究智能体如何通过与环境交互来学习最优策略。",
                "Q-Learning": "Q-Learning 是一种无模型的强化学习算法，用于学习在给定状态下采取特定动作的价值。",
                "DQN": "DQN 是深度 Q 网络，使用深度神经网络来近似 Q 函数，解决高维状态空间问题。",
                "PPO": "PPO 是近端策略优化算法，是一种策略梯度方法，以稳定性和效率著称。",
                "Agent": "Agent 是智能体，是能够感知环境并采取行动的实体。"
            }
            for key, value in knowledge.items():
                if key in keyword:
                    return value
            return "未找到相关信息"

        def move(direction: str, steps: int = 1) -> str:
            """移动工具"""
            return f"向 {direction} 移动了 {steps} 步 (模拟)"

        # 注册工具
        self.tools.register(Tool(
            name="search",
            description="搜索互联网获取信息",
            func=search
        ))
        self.tools.register(Tool(
            name="calculate",
            description="执行数学计算",
            func=calculate
        ))
        self.tools.register(Tool(
            name="retrieve",
            description="从知识库检索相关信息",
            func=retrieve
        ))
        self.tools.register(Tool(
            name="move",
            description="执行移动动作",
            func=move
        ))

    def think(self, state: Any, goal: str, context: str = "") -> str:
        """
        主推理循环

        Args:
            state: 当前环境状态
            goal: 目标描述
            context: 额外上下文

        Returns:
            最终行动和推理链
        """
        self.history = []
        current_state = state
        step_num = 0

        print(f"\n{'='*50}")
        print(f"{self.name} 开始推理")
        print(f"目标: {goal}")
        print(f"{'='*50}\n")

        while step_num < self.max_steps:
            step_num += 1

            # 1. Thought: 分析当前状态
            thought = self._reason(current_state, goal, context)
            print(f"[Step {step_num}]")
            print(f"  Thought: {thought}")

            # 2. 决定行动
            action, action_input = self._decide_action(thought, current_state, goal)
            print(f"  Action: {action}({action_input})")

            # 3. 执行行动
            observation = self._execute_action(action, action_input)
            print(f"  Observation: {observation}")

            # 4. 记录历史
            self.history.append(Step(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                step_num=step_num
            ))

            # 5. 检查是否完成
            if self._is_complete(goal, observation):
                print(f"\n[完成] 目标达成!")
                break

            # 6. 更新状态
            current_state = observation

        return self._format_result()

    def _reason(self, state: Any, goal: str, context: str) -> str:
        """
        推理过程

        分析当前状态，决定如何推进目标
        """
        history_summary = ""
        if self.history:
            recent = self.history[-3:]
            history_summary = "最近行动: " + ", ".join(
                f"{s.action}({s.observation})" for s in recent
            )

        # 基于规则的简单推理
        if "搜索" in goal or "查找" in goal:
            return f"需要搜索相关信息来完成目标: {goal}"
        elif "计算" in goal or any(c in goal for c in "0123456789+-"):
            return f"需要进行数学计算来分析: {goal}"
        elif "移动" in goal or "到达" in goal:
            return f"需要规划移动路径来: {goal}"
        elif history_summary:
            return f"基于历史经验继续: {history_summary}"
        else:
            return f"分析当前状态 '{state}'，制定下一步计划"

    def _decide_action(self, thought: str, state: Any, goal: str) -> tuple:
        """决定具体行动"""
        thought_lower = thought.lower()

        if "搜索" in thought or "查找" in thought:
            return "search", goal
        elif "计算" in thought or any(c in thought for c in "+-*/"):
            # 提取计算表达式
            nums = re.findall(r'[\d.]+', thought)
            if nums:
                expr = "+".join(nums[:4])
                return "calculate", expr
            return "calculate", "1+2+3"
        elif "移动" in thought or "到达" in thought:
            return "move", "前进"
        elif "知识" in thought or "检索" in thought:
            return "retrieve", goal
        else:
            # 默认知识检索
            return "retrieve", goal

    def _execute_action(self, action: str, action_input: Any) -> Any:
        """执行行动"""
        tool = self.tools.get_tool(action)
        if tool:
            try:
                result = tool.func(action_input)
                return result
            except Exception as e:
                return f"执行失败: {str(e)}"
        return f"未知行动: {action}"

    def _is_complete(self, goal: str, observation: Any) -> bool:
        """检查目标是否完成"""
        obs_str = str(observation).lower()
        goal_lower = goal.lower()

        # 检查关键词
        success_keywords = ["完成", "成功", "找到", "结果"]
        for kw in success_keywords:
            if kw in obs_str:
                return True

        return False

    def _format_result(self) -> str:
        """格式化结果"""
        result = f"\n{'='*50}\n"
        result += f"{self.name} 推理结果\n"
        result += f"{'='*50}\n"
        result += f"总步数: {len(self.history)}\n\n"

        for step in self.history:
            result += f"Step {step.step_num}:\n"
            result += f"  Thought: {step.thought}\n"
            result += f"  Action: {step.action}({step.action_input})\n"
            result += f"  Observation: {step.observation}\n\n"

        return result


class ToolformerAgent(ReActAgent):
    """
    Toolformer 风格 Agent

    能够学习和使用新工具的 Agent
    """

    def __init__(self, name: str = "Toolformer-Agent"):
        super().__init__(name)
        self.discovered_tools: List[Tool] = []

    def discover_tool(self, tool_def: Dict):
        """
        发现并注册新工具

        Args:
            tool_def: 工具定义字典
        """
        tool = Tool(
            name=tool_def["name"],
            description=tool_def.get("description", ""),
            func=tool_def.get("func", lambda x: x),
            input_schema=tool_def.get("input_schema", {}),
            output_schema=tool_def.get("output_schema", {})
        )
        self.tools.register(tool)
        self.discovered_tools.append(tool)
        print(f"[工具发现] 新工具: {tool.name} - {tool.description}")


def demo_react():
    """ReAct Agent 演示"""
    print("=" * 60)
    print("ReAct Agent (推理 + 行动) 演示")
    print("=" * 60)

    agent = ReActAgent(name="Researcher", max_steps=5)

    # 演示不同任务
    tasks = [
        ("搜索强化学习相关信息", "了解 RL"),
        ("计算 2+3*4-5", "数值分析"),
        ("查找 Q-Learning 相关信息", "知识检索"),
    ]

    for goal, context in tasks:
        print(f"\n任务: {goal}")
        result = agent.think(state="idle", goal=goal, context=context)
        print(result)


def demo_toolformer():
    """Toolformer Agent 演示"""
    print("\n" + "=" * 60)
    print("Toolformer Agent 演示")
    print("=" * 60)

    agent = ToolformerAgent()

    # 发现新工具
    agent.discover_tool({
        "name": "python_executor",
        "description": "执行 Python 代码",
        "func": lambda code: f"执行结果: {code[:50]}... (模拟)"
    })

    # 使用新工具
    print("\n使用发现的工具:")
    result = agent.think(state="idle", goal="执行代码 print('Hello')", context="")
    print(result)


if __name__ == "__main__":
    demo_react()
    demo_toolformer()
