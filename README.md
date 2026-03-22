# Introduction-to-Reinforcement-Learning-Basics

> 从 Q-Learning 到 AI Agent，构建智能决策系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目覆盖从经典强化学习算法到前沿 AI Agent 智能体系统的完整学习体系，包括：
- 经典 RL 算法（Q-Learning → SAC）
- 大语言模型 + 强化学习（LLM + RL）
- 多智能体系统（Multi-Agent RL）
- Agent 记忆与规划系统
- RAG + RL 检索增强决策

---

## 算法学习路线图

### 第一阶段：经典强化学习

| 算法 | 文件夹 | 难度 | 核心特点 |
|------|--------|------|----------|
| **Q-Learning** | `Q-Learning/` | ⭐ | 表格型 Q 学习，入门首选 |
| **DQN** | `DQN/` | ⭐⭐ | 深度 Q 网络，处理高维状态 |
| **Double DQN** | `DDQN/` | ⭐⭐ | 解决 Q 值过估计问题 |
| **Dueling DQN** | `DuelingDQN/` | ⭐⭐ | 分离状态价值和优势函数 |
| **Rainbow DQN** | `Rainbow/` | ⭐⭐⭐ | 集成 6 种改进的 SOTA 算法 |

### 第二阶段：策略梯度与 Actor-Critic

| 算法 | 文件夹 | 难度 | 核心特点 |
|------|--------|------|----------|
| **REINFORCE** | `REINFORCE/` | ⭐⭐ | 蒙特卡洛策略梯度 |
| **A2C** | `ActorCritic/` | ⭐⭐ | 优势 Actor-Critic |
| **A3C** | `A3C/` | ⭐⭐⭐ | 异步优势 Actor-Critic |
| **PPO** | `PPO/` | ⭐⭐⭐ | 近端策略优化，稳定高效 |
| **TRPO** | `TRPO/` | ⭐⭐⭐⭐ | 信任区域策略优化 |
| **SAC** | `SAC/` | ⭐⭐⭐⭐ | 软演员-评论家，最大熵 |

### 第三阶段：AI Agent 与 LLM + RL（前沿）

| 方向 | 文件夹 | 难度 | 核心特点 |
|------|--------|------|----------|
| **LLM 驱动的 RL Agent** | `LLM_RL/` | ⭐⭐⭐ | 用大模型做决策推理 |
| **多智能体协作** | `MultiAgent/` | ⭐⭐⭐⭐ | Agent 间协作与博弈 |
| **Agent 记忆系统** | `Agent_Memory/` | ⭐⭐⭐ | 让 Agent 记住经验 |
| **RAG + RL** | `RAG_RL/` | ⭐⭐⭐⭐ | 检索增强的决策系统 |
| **工具调用 Agent** | `ToolAgent/` | ⭐⭐⭐ | Toolformer / ReAct 范式 |
| **Agent 规划系统** | `PlanningAgent/` | ⭐⭐⭐⭐⭐ | CoT / ToT / GoT 推理 |

### 深度学习基础

| 项目 | 文件夹 | 描述 |
|------|--------|------|
| **MNIST 手写识别** | `DL/` | CNN 实现 |
| **数据预处理** | `DL/` | 数据标准化与增强 |

---

## 项目结构

```
Introduction-to-Reinforcement-Learning-Basics/
├── README.md
├── requirements.txt
├── setup.py
│
├── Q-Learning/              # Q-Learning 算法
├── DQN/                    # 深度 Q 网络
├── DDQN/                   # Double DQN
├── DuelingDQN/            # Dueling DQN
├── Rainbow/               # Rainbow DQN
│
├── REINFORCE/             # 策略梯度
├── ActorCritic/           # Actor-Critic
├── A3C/                   # 异步 A3C
├── PPO/                   # 近端策略优化
├── TRPO/                  # 信任区域策略优化
├── SAC/                   # 软演员-评论家
│
├── LLM_RL/                # LLM + 强化学习
│   ├── llm_rl_agent.py    # LLM 作为决策大脑
│   ├── reward_shaping.py # RLHF / 奖励塑形
│   └── train_cartpole.py # 训练示例
│
├── MultiAgent/            # 多智能体系统
│   ├── cooperative.py    # 协作型多 Agent
│   ├── competitive.py    # 竞争型多 Agent
│   └── emergent.py        # 涌现行为
│
├── Agent_Memory/          # Agent 记忆系统
│   ├── episodic_memory.py # 情景记忆
│   ├── semantic_memory.py # 语义记忆
│   └── working_memory.py  # 工作记忆
│
├── RAG_RL/                # RAG + 强化学习
│   ├── rag_agent.py      # RAG 检索增强 Agent
│   └── knowledge_rl.py    # 知识库 + RL 决策
│
├── ToolAgent/             # 工具调用 Agent
│   ├── react.py          # ReAct: 推理 + 行动
│   └── toolformer.py     # Toolformer 范式
│
├── PlanningAgent/         # 规划型 Agent
│   ├── cot.py            # Chain of Thought
│   ├── tot.py            # Tree of Thought
│   └── agentic_rl.py     # Agentic RL
│
├── DL/                    # 深度学习基础
├── utils/                 # 工具函数
└── tutorials/             # 教程文档
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/Carl-Creat/Introduction-to-Reinforcement-Learning-Basics.git
cd Introduction-to-Reinforcement-Learning-Basics
pip install -r requirements.txt
```

### 运行示例

**Q-Learning 入门**
```bash
python Q-Learning/q_learning.py
```

**PPO 训练**
```bash
python PPO/ppo.py --env CartPole-v1 --epochs 500
```

**SAC 连续控制**
```bash
python SAC/sac.py --env Pendulum-v1
```

**LLM + RL Agent**
```bash
python LLM_RL/llm_rl_agent.py --model gpt-4
```

---

## 核心概念

### 1. 强化学习基础

**MDP 四元组**：状态 (S)、动作 (A)、奖励 (R)、转移概率 (P)

**价值函数**：
- V(s)：状态价值
- Q(s,a)：动作价值
- V(s) = E[R_t | s_t = s]
- Q(s,a) = E[R_t + γV(s') | s,a]

### 2. LLM + RL

用大语言模型作为 Agent 的"大脑"：

```
用户指令 → LLM推理 → 行动 → 环境反馈 → 奖励评估 → LLM反思
```

**RLHF**：用 RL 调优 LLM，使其符合人类偏好

**Reward Shaping**：设计奖励函数引导 Agent 行为

### 3. Multi-Agent RL

- **协作**：多个 Agent 共同完成复杂任务
- **竞争**：零和博弈，Agent 间的对抗
- **涌现**：简单规则产生复杂群体行为

### 4. Agent 记忆系统

```
短期记忆（Working Memory）
    ↓ 抽象化
长期记忆（Episodic + Semantic Memory）
    ↓ 检索
决策上下文（Context）
```

---

## AI Agent 学习路径

### Step 1: 经典 RL 基础
掌握 Q-Learning → DQN → PPO → SAC

### Step 2: LLM 决策入门
学习 LLM_RL 模块，理解如何用 RL 调优或引导 LLM

### Step 3: Agent 记忆系统
理解 Agent 如何"记住"和"回忆"经验

### Step 4: 工具调用与规划
掌握 ReAct、CoT、ToT 等推理范式

### Step 5: 多智能体协作
学习多个 Agent 如何协作与竞争

---

## 推荐资源

### 书籍
- 《Reinforcement Learning: An Introduction》- Sutton & Barto
- 《动手学强化学习》- 张伟楠
- 《Designing Language Agents》- LLM + RL 实践指南

### 论文
- RLHF: [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- SAC: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- Toolformer: [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- ReAct: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

### 课程
- [David Silver RL Course](https://www.davidsilver.uk/teaching/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [HuggingFace RL Course](https://huggingface.co/learn/rl-course/)

---

## 更新日志

### 2026-03-22
- 新增 LLM_RL 模块：LLM + 强化学习
- 新增 MultiAgent 模块：多智能体系统
- 新增 Agent_Memory 模块：Agent 记忆系统
- 新增 RAG_RL 模块：检索增强决策
- 新增 ToolAgent 模块：工具调用 Agent
- 新增 PlanningAgent 模块：规划推理系统
- 重写 README，完整 AI Agent 学习路线图

### 2026-03-21
- 新增 A3C、PPO、TRPO、Rainbow DQN 算法
- 新增 ActorCritic 模块

### 2026-02-13
- 初始化仓库，Q-Learning、DQN、DDQN、SAC 基础实现

---

## License

MIT License

---

*Keep learning, keep building!* 🚀
