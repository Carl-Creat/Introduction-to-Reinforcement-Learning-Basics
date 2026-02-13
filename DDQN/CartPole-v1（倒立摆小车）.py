# 补充必要的基础导入
from random import random
import numpy as np
import matplotlib.pyplot as plt
# 替换为维护的 Gymnasium，兼容新版 API
import gymnasium as gym
from collections import namedtuple
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# 补充常量定义
ENV = 'CartPole-v1'  # 替换为新版，消除过时警告
GAMMA = 0.9
MAX_STEPS = 500  # v1 版本最大步数为500，匹配新版环境
NUM_EPISODES = 500
BATCH_SIZE = 32
CAPACITY = 10000

# 补充 Transition 具名元组定义
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)

# 补充 ReplayMemory 类定义
class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY    # memory 最大长度
        self.memory = []    # 存储过往经验
        self.index = 0  # 要保存的索引

    def push(self, state, action, state_next, reward):
        '''保存 transition 到经验池'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        '''随机采样小批量数据'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''返回当前 memory 长度'''
        return len(self.memory)

# 建立深度学习网络
class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

# 执行 DQN 的 Brain 类
class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # 取得 CartPole 的动作数 2

        # 创建存储经验的对象
        self.memory = ReplayMemory(CAPACITY)

        # 构建一个神经网络
        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)  # 主 Q 网络
        self.target_q_network = Net(n_in, n_mid, n_out)  # 目标 Q 网络
        print(self.main_q_network)  # 主网络形状

        # 最优化方法的设定
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        ''' 经验回放学习网络的连接参数 '''

        # 1. 检查内存大小
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. 创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3. 获取 Q(s_t,a_t) 作为监督信息
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 更新连接参数
        self.update_main_q_network()

    def decide_action(self, state, episode):
        '''根据当前状态确定动作'''
        # ε-贪婪法 逐步采用最佳动作
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # 主网络切换到推理模式
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
            # 获得网络输出最大值的索引 index = max(1)[1]
            # .view(1,1) 将 [torch.LongTensor of size 1]　转换为 size 1x1
        else:
            # 随机返回 0,1
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 随机返回
            # action 的形式为 [torch.LongTensor of size 1x1]

        return action

    def make_minibatch(self):
        ''' 2. 创建小批量数据 '''

        # 2.1 从经验池中取出小批量数据
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 将每个变量转换为对应格式
        batch = Transition(*zip(*transitions))

        # 2.3 将每个变量元素转化为对应于小批量的形式
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        ''' 3. 找到 Q(s_t,a_t) 值作为监督信息 '''

        # 3.1 两个网络切换为推理模式
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 求取网络输出的 Q(s_t, a_t)
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # 3.3 求max{Q(s_t+1, a)}。
        # 创建索引掩码以检查 cartpole 是否未完成且具有 next_state
        # 修复：新版 PyTorch 需转换为 bool 类型
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        # 首先全部设置为 0
        next_state_values = torch.zeros(BATCH_SIZE)

        # $a_m$
        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 从主网络中求取下一个状态中最大Q值的动作 a_m
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 仅过滤具有下一个状态的，并将 size 32 转换为 size 32*1
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 从目标 Q 网络中找到具有下一状态的 index 的动作 a_m 的 Q 值
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 根据Q学习公式，求出 Q(s_t, a_t)作为监督信息
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        ''' 4. 更新连接参数 '''

        # 4.1 主网络训练模式
        self.main_q_network.train()

        # 4.2 计算损失函数（smooth_l1_loss 是 Huberloss）
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        # 4.3 更新连接参数
        self.optimizer.zero_grad()  # 重置梯度
        loss.backward()  # 计算反向传播
        self.optimizer.step()  # 更新连接参数

    def update_target_q_function(self):  # 添加 DDQN
        ''' 让目标网络与主网络相同 '''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

# CartPole 上运行的智能体(agent)类，带有杆的小车
class Agent:
    def __init__(self, num_states, num_actions):
        '''设置任务状态和动作的数量'''
        self.brain = Brain(num_states, num_actions)  # 为智能体生成大脑来确定动作

    def update_q_function(self):
        '''更新 Q 函数'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''决定动作'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''将 state, action, state_next, reward 的内容保存在 memory 经验池中'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        ''' 将目标网络更新到与主网络相同 '''
        self.brain.update_target_q_function()

# 执行 CartPole 的环境类
class Environment:
    def __init__(self):
        # 修复：添加 render_mode 确保兼容新版 Gymnasium
        self.env = gym.make(ENV, render_mode='rgb_array')  # 设定任务
        num_states = self.env.observation_space.shape[0]  # 获得任务的状态变量数 4
        num_actions = self.env.action_space.n  # CartPole的动作数 2
        self.agent = Agent(num_states, num_actions)  # 创建 Agent 在环境中执行动作

    def run(self):
        '''执行'''
        episode_10_list = np.zeros(10)  # 存储 10 次试验的连续站立步数
        complete_episodes = 0  # 连续成功统计
        episode_final = False  # 最终尝试标记
        frames = []  # 存储图像的变量

        for episode in range(NUM_EPISODES):  # 重复试验次数
            # 修复：拆分 reset 返回的元组 (state, info)，解决 tuple 转 ndarray 错误
            observation, info = self.env.reset()  # 环境初始化

            state = observation  # 直接将观测值作为状态值使用
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  # NumPy 变量转换为 PyTorch Tensor
            state = torch.unsqueeze(state, 0)  # FloatTensor of size 4 转换为 size 1x4

            for step in range(MAX_STEPS):  # 每 1 轮循环（1 episode）
                # 不再绘制动画
                action = self.agent.get_action(state, episode)  # 求取动作

                # 修复：拆分 step 返回的 5 个值，兼容新版 Gymnasium API
                observation_next, reward, terminated, truncated, info = self.env.step(action.item())
                # 合并 terminated 和 truncated 为 done，保持原逻辑
                done = terminated or truncated

                # 给与奖励。对 episode是否结束以及是否有下一个状态进行判断
                if done:  # 如果 step 不超过 500，或杆子倾斜超过某个角度
                    state_next = None  # 没有下一个状态，存储 None

                    # 添加到最近的 10 episode 的步数列表中
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))

                    # 修复：v1 版本成功阈值为 475 步（对应 v0 的 195 步）
                    if step < 475:
                        reward = torch.FloatTensor(
                            [-1.0])  # 半途倒下，奖励 -1
                        complete_episodes = 0  # 重置连续成功记录
                    else:
                        reward = torch.FloatTensor([1.0])  # 一直站立直到结束时奖励 1
                        complete_episodes = complete_episodes + 1  # 更新连续记录
                else:
                    reward = torch.FloatTensor([0.0])  # 普通奖励 0
                    state_next = observation_next  # 将状态设置为观察值
                    state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor)  # numpy 变量 --> PyTorch Tensor 变量
                    state_next = torch.unsqueeze(state_next, 0)  # FloatTensor of size 4 扩展为 size 1x4

                # 向经验池中添加经验
                self.agent.memorize(state, action, state_next, reward)

                # 经验回放 Experience Replay，更新 Q 函数
                self.agent.update_q_function()

                # 更新观测值
                state = state_next

                # 结束处理
                if done:
                    print('%d Episode: Finished after %d steps：10 次试验平均 step 数 = %.1lf' % (
                        episode, step + 1, episode_10_list.mean()))

                    # 使用 DDQN 添加，每 2 轮试验复制一次主网络到目标网络
                    if (episode % 2 == 0):
                        self.agent.update_target_q_function()
                    break

            if episode_final is True:
                # 不再绘制动画
                break

            # 连续 10 轮成功
            if complete_episodes >= 10:
                print('10 轮连续成功')
                episode_final = True  # 标记下一次为最终试验

# 主函数（添加设备兼容，避免 CUDA/CPU 问题）
if __name__ == "__main__":
    cartpole_env = Environment()
    cartpole_env.run()
