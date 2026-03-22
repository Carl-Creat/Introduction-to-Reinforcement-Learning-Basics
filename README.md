# Introduction-to-Reinforcement-Learning-Basics

> 强化学习入门到精通，从 Q-Learning 到 SAC 的完整实现

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目包含强化学习（RL）从基础到高级的完整算法实现，涵盖值函数方法、策略梯度方法、Actor-Critic 方法等。所有算法均使用 PyTorch 实现，基于 Gymnasium 环境进行训练和测试。

---

## 算法学习路线图

### 基础算法 (Value-Based)

| 算法 | 文件夹 | 难度 | 核心特点 |
|------|--------|------|----------|
| **Q-Learning** | `Q-Learning/` | ⭐ | 表格型 Q 学习，入门首选 |
| **DQN** | `DQN/` | ⭐⭐ | 深度 Q 网络，处理高维状态 |
| **Double DQN** | `DDQN/` | ⭐⭐ | 解决 Q 值过估计问题 |
| **Dueling DQN** | `DuelingDQN/` | ⭐⭐ | 分离状态价值和优势函数 |
| **Rainbow DQN** | `Rainbow/` | ⭐⭐⭐ | 集成 6 种改进的 SOTA 算法 |

### 策略梯度方法 (Policy Gradient)

| 算法 | 文件夹 | 难度 | 核心特点 |
|------|--------|------|----------|
| **REINFORCE** | `REINFORCE/` | ⭐⭐ | 蒙特卡洛策略梯度 |
| **A2C** | `ActorCritic/` | ⭐⭐ | 优势 Actor-Critic |
| **A3C** | `A3C/` | ⭐⭐⭐ | 异步优势 Actor-Critic |
| **PPO** | `PPO/` | ⭐⭐⭐ | 近端策略优化，稳定高效 |
| **TRPO** | `TRPO/` | ⭐⭐⭐⭐ | 信任区域策略优化 |
| **SAC** | `SAC/` | ⭐⭐⭐⭐ | 软演员-评论家，最大熵 |

### 深度学习基础

| 项目 | 文件夹 | 描述 |
|------|--------|------|
| **MNIST 手写识别** | `DL/` | CNN 实现 |
| **数据预处理** | `DL/` | 数据标准化与增强 |

---

## 项目结构

```
Introduction-to-Reinforcement-Learning-Basics/
├── README.md                      # 本文件
├── requirements.txt               # 依赖包
├── setup.py                       # 安装脚本
├── .gitignore                     # Git 忽略文件
├── LICENSE                        # MIT 许可证
│
├── Q-Learning/                    # Q-Learning 算法
│   ├── q_learning.py             # 基础实现
│   └── cartpole_qlearning.py     # CartPole 训练
│
├── DQN/                          # 深度 Q 网络
│   ├── dqn.py                    # DQN 实现
│   └── train_cartpole.py         # 训练脚本
│
├── DDQN/                         # Double DQN
│   ├── ddqn.py                   # DDQN 实现
│   └── train.py                  # 训练脚本
│
├── DuelingDQN/                   # Dueling DQN
│   ├── dueling_dqn.py            # Dueling 结构实现
│   └── train.py                  # 训练脚本
│
├── Rainbow/                      # Rainbow DQN
│   ├── rainbow_dqn.py            # 完整实现
│   └── train.py                  # 训练脚本
│
├── REINFORCE/                    # 策略梯度基础
│   ├── reinforce.py              # REINFORCE 实现
│   └── train.py                  # 训练脚本
│
├── ActorCritic/                  # Actor-Critic
│   ├── a2c.py                    # A2C 实现
│   └── train.py                  # 训练脚本
│
├── A3C/                          # 异步 A3C
│   ├── a3c.py                    # A3C 实现
│   └── train.py                  # 训练脚本
│
├── PPO/                          # 近端策略优化
│   ├── ppo.py                    # PPO 实现
│   └── train.py                  # 训练脚本
│
├── TRPO/                         # 信任区域策略优化
│   ├── trpo.py                   # TRPO 实现
│   └── train.py                  # 训练脚本
│
├── SAC/                          # 软演员-评论家
│   ├── sac.py                    # SAC 实现
│   └── train.py                  # 训练脚本
│
├── DL/                           # 深度学习基础
│   ├── mnist_cnn.py              # MNIST 分类
│   └── data_preprocessing.py     # 数据预处理
│
├── utils/                        # 工具函数
│   ├── networks.py               # 神经网络结构
│   ├── replay_buffer.py          # 经验回放缓冲区
│   ├── visualization.py          # 可视化工具
│   ├── logger.py                 # 日志记录
│   └── plotter.py                # 绘图工具
│
├── environments/                 # 环境相关
│   ├── custom_env.py             # 自定义环境
│   └── wrappers.py               # 环境包装器
│
└── tutorials/                    # 教程文档
    ├── 01_q_learning_theory.md   # Q-Learning 理论
    ├── 02_dqn_theory.md          # DQN 理论
    ├── 03_policy_gradient.md     # 策略梯度理论
    └── 04_actor_critic.md        # Actor-Critic 理论
```

---

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- Gymnasium
- NumPy, Matplotlib

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/Carl-Creat/Introduction-to-Reinforcement-Learning-Basics.git
cd Introduction-to-Reinforcement-Learning-Basics

# 安装依赖
pip install -r requirements.txt

# 或安装为包
pip install -e .
```

### 运行示例

**Q-Learning 训练 CartPole**
```bash
python Q-Learning/cartpole_qlearning.py
```

**DQN 训练**
```bash
python DQN/train_cartpole.py
```

**PPO 训练**
```bash
python PPO/train.py --env CartPole-v1 --epochs 500
```

**SAC 训练**
```bash
python SAC/train.py --env Pendulum-v1
```

---

## 算法详解

### 1. Q-Learning

最基础的强化学习算法，使用表格存储 Q 值。

```python
# Q-Learning 更新公式
Q(s, a) = Q(s, a) + α * [r + γ * max(Q(s', a')) - Q(s, a)]
```

**特点**：
- 适用于离散状态和动作空间
- 需要探索-利用平衡（ε-贪心）
- 收敛性有保证

### 2. DQN (Deep Q-Network)

使用神经网络近似 Q 函数，处理高维状态空间。

**关键技术**：
- 经验回放 (Experience Replay)
- 目标网络 (Target Network)
- 奖励裁剪 (Reward Clipping)

### 3. Double DQN

解决 DQN 的 Q 值过估计问题。

```python
# Double DQN 目标值计算
action = argmax(Q_online(s'))
target = r + γ * Q_target(s', action)
```

### 4. Dueling DQN

将 Q 函数分解为状态价值 V(s) 和优势函数 A(s,a)。

```python
Q(s, a) = V(s) + A(s, a) - mean(A(s, :))
```

### 5. Rainbow DQN

集成 6 种改进：
1. Double DQN
2. Dueling DQN
3. Prioritized Experience Replay
4. Multi-step Learning
5. Distributional RL (C51)
6. Noisy Nets

### 6. REINFORCE

蒙特卡洛策略梯度方法。

```python
∇J(θ) = E[∇log π(a|s) * G_t]
```

### 7. Actor-Critic

结合值函数和策略梯度。

- **Actor**: 策略网络，选择动作
- **Critic**: 值网络，评估动作

### 8. A3C

异步优势 Actor-Critic，多线程并行训练。

### 9. PPO

近端策略优化，使用裁剪目标函数限制策略更新幅度。

```python
L_CLIP(θ) = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
```

### 10. TRPO

信任区域策略优化，使用 KL 散度约束策略更新。

### 11. SAC

软演员-评论家，最大熵强化学习。

**特点**：
- 自动调整温度参数
- 双 Q 网络减少过估计
- 重参数化技巧

---

## 可视化工具

### 训练过程可视化

```python
from utils.visualization import TrainingVisualizer

visualizer = TrainingVisualizer()
visualizer.plot_rewards(rewards, save_path='results/rewards.png')
visualizer.plot_losses(losses, save_path='results/losses.png')
```

### 结果对比

```python
from utils.plotter import compare_algorithms

compare_algorithms(
    algorithms=['DQN', 'DoubleDQN', 'DuelingDQN'],
    env='CartPole-v1',
    episodes=500
)
```

---

## 学习资源

### 推荐书籍
- 《Reinforcement Learning: An Introduction》- Sutton & Barto (圣经)
- 《动手学强化学习》- 张伟楠等
- 《深度强化学习》- 王树森

### 在线课程
- [David Silver RL Course](https://www.davidsilver.uk/teaching/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [CS285: Deep RL - Sergey Levine](http://rail.eecs.berkeley.edu/deeprlcourse/)

### 经典论文
- DQN: [Human-level control through deep RL](https://www.nature.com/articles/nature14236)
- Double DQN: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- Dueling DQN: [Dueling Network Architectures for Deep RL](https://arxiv.org/abs/1511.06581)
- Rainbow: [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298)
- PPO: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- SAC: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)

---

## 实验环境

### Gymnasium 环境

| 环境 | 描述 | 适用算法 |
|------|------|----------|
| CartPole-v1 | 倒立摆 | 所有算法 |
| Acrobot-v1 | 双摆 | 所有算法 |
| MountainCar-v0 | 山地车 | DQN, PPO |
| Pendulum-v1 | 摆锤 | 连续动作算法 |
| LunarLander-v2 | 月球着陆器 | DQN, PPO, SAC |

### Atari 游戏

```bash
# 安装 Atari 环境
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]

# 训练 DQN 玩 Pong
python DQN/train_atari.py --env PongNoFrameskip-v4
```

---

## 贡献指南

欢迎提交 PR 完善仓库！

1. Fork 本仓库
2. 创建新分支 (`git checkout -b feature/新算法`)
3. 添加代码和注释
4. 提交更改 (`git commit -m '添加 xxx 算法'`)
5. 推送到分支 (`git push origin feature/新算法`)
6. 提交 Pull Request

### 代码规范
- 遵循 PEP 8 规范
- 添加详细的文档字符串
- 包含运行示例
- 添加单元测试（可选）

---

## 更新日志

### 2026-03-21
- 新增 A3C、PPO、TRPO、Rainbow DQN、Dueling DQN 算法
- 添加可视化工具和日志系统
- 完善教程文档
- 重构项目结构

### 2026-02-13
- 初始化仓库
- 添加 Q-Learning、DQN、DDQN、SAC 基础实现

---

## License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

*持续更新中... Keep learning, keep coding!* 🚀
