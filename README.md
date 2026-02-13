# Introduction-to-Reinforcement-Learning-Basics
强化学习基础入门
# 强化学习与深度学习实践项目 🤖

本项目是我在强化学习（RL）与深度学习（DL）领域的学习与实践合集，包含从经典Q-Learning到现代SAC算法，以及深度学习在MNIST手写数字识别上的应用。所有代码均适配了最新的Gymnasium和PyTorch API，可直接运行。

---

## 📁 项目结构

```
.
├── DDQN/                  # Double DQN 算法实现
│   └── CartPole-v1 (倒立摆小车).py
├── DL/                    # 深度学习实践
│   ├── MNIST dataset loading and pre...py  # MNIST数据加载与预处理
│   └── number recognition(数字识别).py     # 基于PyTorch的MNIST数字识别
├── DQN/                   # DQN 算法实现
│   └── CartPole-v1 (倒立摆小车).py
├── Q-Learning/            # 经典Q-Learning算法
│   ├── CartPole-v1 (inverted pendulum...py
│   └── Simple Maze Game (DRL).py           # 网格迷宫导航
├── SAC/                   # Soft Actor-Critic 算法
│   ├── real-time inverted pendulum.py      # 实时倒立摆控制
│   └── 伪代码.py
└── README.md
```

---

## 🎯 项目亮点

### 1. 强化学习 (Reinforcement Learning)
- **Q-Learning**: 实现了表格型Q-Learning，用于解决网格迷宫和CartPole问题。
- **DQN**: 使用深度Q网络（DQN）解决CartPole-v1，通过经验回放和ε-贪婪策略提升训练稳定性。
- **DDQN**: 在DQN基础上引入双网络架构，有效解决了Q值高估问题。
- **SAC**: 实现了适用于连续动作空间的Soft Actor-Critic算法，用于实时控制倒立摆。

### 2. 深度学习 (Deep Learning)
- **MNIST数字识别**: 基于PyTorch实现了全连接神经网络，完成手写数字识别任务。
- **数据预处理**: 提供了适配新版scikit-learn的MNIST数据集标准化加载、归一化和可视化流程。

---

## 🚀 快速开始

### 环境依赖
- Python 3.8+
- PyTorch
- Gymnasium
- scikit-learn
- NumPy, Matplotlib

### 运行示例
1.  **Q-Learning 迷宫游戏**
    ```bash
    python Q-Learning/Simple\ Maze\ Game\ (DRL).py
    ```
2.  **DDQN 控制 CartPole**
    ```bash
    python DDQN/CartPole-v1\ \(倒立摆小车\).py
    ```
3.  **MNIST 数字识别**
    ```bash
    python DL/number\ recognition\(数字识别\).py
    ```

---

## 📚 学习路径

本项目按照从易到难的顺序组织，适合作为强化学习和深度学习的入门实践：
1.  **Q-Learning**: 从表格型方法开始，理解价值迭代和探索-利用平衡。
2.  **DQN/DDQN**: 学习如何用深度神经网络逼近价值函数。
3.  **SAC**: 掌握策略梯度方法，处理连续动作空间问题。
4.  **深度学习**: 应用深度学习解决经典的图像分类任务。

---

## 📝 许可证

本项目采用 MIT 许可证，欢迎学习和交流。

---
