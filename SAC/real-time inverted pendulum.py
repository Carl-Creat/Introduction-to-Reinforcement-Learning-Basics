import random
import gymnasium as gym  # 环境
import numpy as np
from tqdm import tqdm  # 添加实时可视化进度条
import torch
import torch.nn.functional as F  # 导入神经网络功能模块
from torch.distributions import Normal, Categorical  # 导入正态分布（Normal，选取连续的数） 和分类分布（Categorical，选取离散的类别）
import matplotlib.pyplot as plt
import time  # 用于控制渲染速度


class ReplayBuffer:
    """经验回放池"""

    '''构建空的经验回放池'''

    def __init__(self, capacity):  # 定义了一个有容量限制的循环缓冲区
        self.buffer = []  # 设定一个空回放池
        self.capacity = capacity  # 设定容量大小
        self.position = 0  # 设定更新位置

    '''指定存储及增删规则'''

    # 把智能体和环境交互产生的经验数据存入缓冲区
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # 进行预占位，将经验池初始化为capacity个位置
        self.buffer[self.position] = (state, action, reward, next_state, done)  # 从第position个位置进行数据的添加
        self.position = (self.position + 1) % self.capacity  # 在数量超过capacity时进行循环增删

    '''随机采样以供训练'''

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)  # 在经验池中随机无放回的抽取batch_size个数据以供训练使用
        state, action, reward, next_state, done = zip(*batch)  # 将原本随机抽取的数据进行解包，重新分成5组
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(
            done)  # 将这5组数据再转换为numpy格式以供深度学习时使用

    '''展示经验池中的容量'''

    def __len__(self):
        """返回当前经验池大小"""
        return len(self.buffer)  # 返回当前经验池里实际存储的经验数量


def train_off_policy_agent(env, env_render, agent, num_episodes, replay_buffer, minimal_size, batch_size,
                           render_speed=0.01):
    """离线策略训练函数"""
    return_list = []
    for i in range(10):  # 分10个进度条训练
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                # 初始化训练环境和渲染环境（保证初始状态一致）
                state, _ = env.reset(seed=0)
                state_render, _ = env_render.reset(seed=0)
                done = False
                truncated = False

                # 每一步都进行渲染
                step_count = 0
                while not done and not truncated:
                    step_count += 1
                    # 智能体选择动作
                    action = agent.take_action(state)

                    # 训练环境交互
                    next_state, reward, done, truncated, _ = env.step(action)
                    replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    # 渲染环境同步交互（每一步都渲染）
                    next_state_render, _, done_render, truncated_render, _ = env_render.step(action)
                    env_render.render()  # 实时渲染当前帧
                    time.sleep(render_speed)  # 控制渲染速度
                    state_render = next_state_render

                    # 经验池足够大时才开始训练
                    if len(replay_buffer) > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)

                return_list.append(episode_return)
                # 每10个episode更新一次进度条
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list


def moving_average(a, window_size):
    """滑动平均函数，对于普通的折线图进行平滑化处理"""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# ===================== 连续动作SAC（Pendulum-v1） =====================
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 第一层全连接，进行特征提取
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)  # 输出动作分布的mu值
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)  # 输出动作分布的std值
        self.action_bound = action_bound  # 规范动作的边界

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)  # 高斯分布
        normal_sample = dist.rsample()  # 重参数化采样，使得采样过程可导
        log_prob = dist.log_prob(normal_sample)  # 计算采样值的对数概率，用于SAC熵的计算
        action = torch.tanh(normal_sample)
        # 修正tanh后的对数概率
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)  # 第一层全连接，进行特征提取
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # 第二层全连接，加深网络，提升表达能力
        self.fc_out = torch.nn.Linear(hidden_dim, 1)  # 输出层，输出单个的Q值

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))  # 第一层特征提取，激活ReLU，引入非线性
        x = F.relu(self.fc2(x))  # 第二层加深网络
        return self.fc_out(x)


class SACContinuous:
    """处理连续动作的SAC算法"""

    '''初始化函数_int_'''

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy,
                 tau, gamma, device):
        # 搭建核心网络（策略网络+双Q网络+双目标网络）
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)  # 输出动作和对数概率
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 评估动作价值
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 避免高估
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 计算目标值
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)

        # 初始化目标网络参数，使其初始化为和当前Q网络完全一致
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # 温度系数α（用log值保证非负）
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy  # 目标熵，控制探索速度
        self.gamma = gamma  # 折扣因子，代表未来奖励的权重
        self.tau = tau  # 软更新系数，代表目标网络更新幅度
        self.device = device  # 训练设备

    '''动作选择take_action(和环境交互）'''

    def take_action(self, state):
        """选择动作（适配单个状态输入）"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 转换为张量并加上bach维度
        action, _ = self.actor(state)  # 策略网络输出动作
        return [action.item()]  # 转换为列表

    '''目标Q值计算calc_target(TD目标）'''

    def calc_target(self, rewards, next_states, dones):
        """计算目标Q值"""
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob  # 熵的定义：1）香农熵-∑p * log（p）；2）自信息-log(p);这里指后者；
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        """软更新目标网络"""
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        """更新网络参数"""
        # 转换为张量并移到指定设备
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 奖励重塑（Pendulum-v1的奖励范围是[-16.2, 0]，重塑到[0, 1]附近）
        rewards = (rewards + 8.0) / 8.0

        # ========== 更新双Q网络 ==========
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))  # Q1损失，当前Q1值和目标Q值的均方误差
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))  # Q2损失，当前Q2值和目标Q值的均方误差

        # 反向传播+优化处理
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ========== 更新策略网络 ==========
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== 更新温度系数α ==========
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # ========== 软更新目标网络 ==========
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


# ===================== 训练连续动作SAC（Pendulum-v1） =====================
print("===== 开始训练Pendulum-v1（连续动作SAC） =====")
env_name = 'Pendulum-v1'
# 创建两个环境：一个用于训练（无渲染），一个用于可视化（有渲染）
env = gym.make(env_name)  # 训练用环境
env_render = gym.make(env_name, render_mode="human")  # 可视化用环境

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值

# 随机种子
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# 超参数
actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 100
hidden_dim = 128
gamma = 0.99
tau = 0.005
buffer_size = 100000
minimal_size = 1000
batch_size = 64
target_entropy = -env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
render_speed = 0.005  # 渲染速度，值越小越快（0.005是比较流畅的速度）

# 初始化经验池和智能体
replay_buffer = ReplayBuffer(buffer_size)
agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)

# 训练并实时渲染每一步
return_list = train_off_policy_agent(env, env_render, agent, num_episodes,
                                     replay_buffer, minimal_size,
                                     batch_size, render_speed)

# 绘制训练结果
episodes_list = list(range(len(return_list)))
fig, ax = plt.subplots(figsize=(6, 6))

# 绘制原始曲线和滑动平均曲线
ax.plot(episodes_list, return_list, color='#1f77b4', alpha=0.3, linewidth=1.2, label='Raw Returns')
ax.plot(episodes_list, moving_average(return_list, 9), color='#ff7f0e', linewidth=2, label='Moving Average')

ax.set_xlabel('Episodes', fontsize=10, fontweight='bold')
ax.set_ylabel('Returns', fontsize=10, fontweight='bold')
ax.set_title(f'SAC (Continuous) on {env_name}', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', fontsize=10, frameon=True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# 关闭环境
env.close()
env_render.close()
