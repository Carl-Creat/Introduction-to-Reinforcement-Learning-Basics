'''# 定义 SAC 超参数
alpha = 0.2  # 熵正则项系数
gamma = 0.99  # 折扣因子
tau = 0.005  # 目标网络软更新参数
lr = 3e-4  # 学习率

# 初始化 Actor、Critic、Target Critic 网络和优化器
actor = ActorNetwork()  # 策略网络 π(s)
critic1 = CriticNetwork()  # 第一个 Q 网络 Q1(s, a)
critic2 = CriticNetwork()  # 第二个 Q 网络 Q2(s, a)
target_critic1 = CriticNetwork()  # 目标 Q 网络 1
target_critic2 = CriticNetwork()  # 目标 Q 网络 2

# 将目标 Q 网络的参数设置为与 Critic 网络相同
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

# 初始化优化器
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=lr)
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=lr)

# 经验回放池（Replay Buffer）
replay_buffer = ReplayBuffer()

# SAC 训练循环
for each iteration:
    # Step 1: 从 Replay Buffer 中采样一个批次 (state, action, reward, next_state)
    batch = replay_buffer.sample()
    state, action, reward, next_state, done = batch

    # Step 2: 计算目标 Q 值 (y)
    with torch.no_grad():
        # 从 Actor 网络中获取 next_state 的下一个动作
        next_action, next_log_prob = actor.sample(next_state)

        # 目标 Q 值的计算：使用目标 Q 网络的最小值 + 熵项
        target_q1_value = target_critic1(next_state, next_action)
        target_q2_value = target_critic2(next_state, next_action)
        min_target_q_value = torch.min(target_q1_value, target_q2_value)

        # 目标 Q 值 y = r + γ * (最小目标 Q 值 - α * next_log_prob)
        target_q_value = reward + gamma * (1 - done) * (min_target_q_value - alpha * next_log_prob)

    # Step 3: 更新 Critic 网络
    # Critic 1 损失
    current_q1_value = critic1(state, action)
    critic1_loss = F.mse_loss(current_q1_value, target_q_value)

    # Critic 2 损失
    current_q2_value = critic2(state, action)
    critic2_loss = F.mse_loss(current_q2_value, target_q_value)

    # 反向传播并更新 Critic 网络参数
    critic1_optimizer.zero_grad()
    critic1_loss.backward()
    critic1_optimizer.step()

    critic2_optimizer.zero_grad()
    critic2_loss.backward()
    critic2_optimizer.step()

    # Step 4: 更新 Actor 网络
    # 通过 Actor 网络生成新的动作及其 log 概率
    new_action, log_prob = actor.sample(state)

    # 计算 Actor 的目标损失：L = α * log_prob - Q1(s, π(s))
    q1_value = critic1(state, new_action)
    actor_loss = (alpha * log_prob - q1_value).mean()

    # 反向传播并更新 Actor 网络参数
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Step 5: 软更新目标 Q 网络参数
    with torch.no_grad():
        for param, target_param in zip(critic1.parameters(), target_critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(critic2.parameters(), target_critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
'''
