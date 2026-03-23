# 强化学习基础入门

> 本教程面向零基础读者，从直觉出发理解强化学习的核心概念。

---

## 1. 什么是强化学习？

强化学习（Reinforcement Learning, RL）是机器学习的一个分支，研究**智能体（Agent）如何在环境中通过试错来学习最优行为**。

类比：训练一只狗
- 狗做对了 → 给零食（正奖励）
- 狗做错了 → 不给零食（负奖励）
- 狗通过反复尝试，学会了什么行为能得到零食

在 RL 中：
- **狗** = Agent（智能体）
- **主人的指令** = 环境（Environment）
- **零食** = 奖励（Reward）
- **狗的行为** = 动作（Action）

---

## 2. 核心概念

### MDP（马尔可夫决策过程）

RL 问题通常用 MDP 建模，由四元组 **(S, A, R, P)** 定义：

| 符号 | 名称 | 含义 |
|------|------|------|
| S | 状态空间 | 环境所有可能的状态 |
| A | 动作空间 | Agent 可以执行的所有动作 |
| R | 奖励函数 | R(s, a) → 执行动作 a 后获得的即时奖励 |
| P | 转移概率 | P(s'|s, a) → 在状态 s 执行 a 后转移到 s' 的概率 |

**马尔可夫性质**：下一个状态只依赖当前状态，与历史无关。

### 策略（Policy）π

策略是 Agent 的"行为准则"，定义在每个状态下选择哪个动作：

- **确定性策略**：π(s) = a（给定状态，输出确定动作）
- **随机性策略**：π(a|s) = P(A=a|S=s)（给定状态，输出动作概率分布）

### 价值函数（Value Function）

价值函数衡量"从某个状态出发，未来能获得多少累积奖励"：

**状态价值函数 V(s)**：
```
V^π(s) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s, π]
```

**动作价值函数 Q(s, a)**：
```
Q^π(s, a) = E[R_t + γR_{t+1} + ... | s_t = s, a_t = a, π]
```

其中 **γ（gamma）** 是折扣因子（0 < γ ≤ 1），控制对未来奖励的重视程度。

### Bellman 方程

Bellman 方程是 RL 的核心递推关系：

```
Q(s, a) = R(s, a) + γ * max_{a'} Q(s', a')
```

直觉：当前状态的价值 = 即时奖励 + 折扣后的未来最优价值

---

## 3. RL 算法分类

```
强化学习算法
├── 基于价值 (Value-Based)
│   ├── Q-Learning          ← 表格型，入门首选
│   ├── DQN                 ← 深度神经网络 + Q-Learning
│   ├── Double DQN          ← 解决过估计问题
│   ├── Dueling DQN         ← 分离 V(s) 和 A(s,a)
│   └── Rainbow DQN         ← 集成多种改进
│
├── 基于策略 (Policy-Based)
│   ├── REINFORCE           ← 蒙特卡洛策略梯度
│   └── PPO / TRPO          ← 近端/信任区域策略优化
│
└── Actor-Critic (混合)
    ├── A2C / A3C           ← 优势 Actor-Critic
    └── SAC                 ← 最大熵 Actor-Critic
```

---

## 4. 学习路线建议

### 第一阶段（1-2周）：Q-Learning
```bash
python Q-Learning/q_learning.py
```
- 理解 Q 表的更新规则
- 在 FrozenLake 或 Taxi 环境上实验

### 第二阶段（2-3周）：DQN
```bash
python DQN/dqn.py
```
- 理解经验回放（Experience Replay）
- 理解目标网络（Target Network）
- 在 CartPole 上训练

### 第三阶段（3-4周）：PPO
```bash
python PPO/ppo.py --env CartPole-v1
```
- 理解策略梯度定理
- 理解 Clip 目标函数
- 在 LunarLander 上训练

### 第四阶段（4周+）：SAC + Agent
```bash
python SAC/sac.py --env Pendulum-v1
python LLM_RL/llm_rl_agent.py
```

---

## 5. 推荐资源

### 入门书籍
- 《Reinforcement Learning: An Introduction》Sutton & Barto（免费在线版）
- 《动手学强化学习》张伟楠（中文，代码友好）

### 视频课程
- [David Silver RL Course](https://www.davidsilver.uk/teaching/) - DeepMind 经典课程
- [OpenAI Spinning Up](https://spinningup.openai.com/) - 实践导向

### 练习环境
- [Gymnasium](https://gymnasium.farama.org/) - OpenAI Gym 继任者
- CartPole-v1：平衡杆（入门）
- LunarLander-v2：月球着陆（进阶）
- Pendulum-v1：连续控制（进阶）

---

*下一篇：[02_dqn_deep_dive.md](02_dqn_deep_dive.md) - DQN 深度解析*
