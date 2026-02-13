import numpy as np
import gymnasium as gym


def q_learning():
    # 启用渲染模式（human模式会弹出窗口显示动画）
    env = gym.make('CartPole-v1', render_mode='human')
    action_size = env.action_space.n

    # 进行数据的离散化
    bins = 10
    bounds = [(-4.8, 4.8), (-3, 3), (-0.418, 0.418), (-4, 4)]

    def discretize(obs):
        block_nums = []
        for i in range(4):
            min_v, max_v = bounds[i]
            ratio = (obs[i] - min_v) / (max_v - min_v)
            block = int(max(0, min(bins - 1, ratio * bins)))
            block_nums.append(block)
        return block_nums[0] * bins ** 3 + block_nums[1] * bins ** 2 + block_nums[2] * bins + block_nums[3]

    # 初始化Q表
    Q = np.random.uniform(low=0.0, high=0.1, size=(bins ** 4, action_size))

    # 设置参数
    num_episodes = 1000
    alpha = 0.15
    gamma = 0.995
    epsilon = 0.4
    epsilon_decay = 0.998
    epsilon_min = 0.02

    # 控制可视化频率：只在测试时渲染，且每100轮测试一次（避免窗口闪烁）
    visualize_interval = 100

    for episode in range(num_episodes):
        # 训练环节（不渲染，加快训练速度）
        state = discretize(env.reset()[0])
        done = False
        episode_reward = 0

        while not done:
            # 训练时不渲染，只在测试时渲染
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_obs, reward, done, _, _ = env.step(action)
            next_state = discretize(next_obs)
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            state = next_state
            episode_reward += reward

        # 衰减探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 测试环节（每隔visualize_interval轮渲染一次，观察训练效果）
        total_reward = 0
        state_test = discretize(env.reset()[0])
        done_test = False
        max_test_steps = 500
        test_step = 0

        while not done_test and test_step < max_test_steps:
            # 测试时强制渲染，显示动画
            action_test = np.argmax(Q[state_test, :])
            next_obs_test, reward_test, done_test, _, _ = env.step(action_test)
            total_reward += reward_test
            state_test = discretize(next_obs_test)
            test_step += 1

            # 可选：添加微小延迟，让动画更易观察（如果觉得太快可以取消注释）
            # import time
            # time.sleep(0.01)

        # 打印日志和控制可视化频率
        if (episode + 1) % 50 == 0:
            print(f"训练次数:{episode + 1}, 训练奖励:{episode_reward:.0f}, 测试奖励:{total_reward:.0f}")

        # 每visualize_interval轮后，保持窗口显示1秒（方便观察）
        if (episode + 1) % visualize_interval == 0 and total_reward > 0:
            import time
            print(f"=== 第{episode + 1}轮测试完成，窗口将保持1秒 ===")
            time.sleep(1)

    # 训练结束后，进行一次最终测试（长时间渲染展示最终效果）
    print("\n=== 训练完成，展示最终效果（持续5秒）===")
    final_reward = 0
    state_final = discretize(env.reset()[0])
    done_final = False
    final_step = 0
    while not done_final and final_step < 500:
        action_final = np.argmax(Q[state_final, :])
        next_obs_final, reward_final, done_final, _, _ = env.step(action_final)
        final_reward += reward_final
        state_final = discretize(next_obs_final)
        final_step += 1
        import time
        time.sleep(0.02)  # 适当延迟，让动画更清晰
    print(f"最终测试奖励: {final_reward:.0f}")

    env.close()


if __name__ == "__main__":
    q_learning()
