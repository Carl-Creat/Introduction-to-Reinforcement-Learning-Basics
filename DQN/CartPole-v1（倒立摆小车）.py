# 1. 包
from random import random

import numpy as np
import matplotlib.pyplot as plt
# 修复：替换为维护的 Gymnasium，兼容新版 API
import gymnasium as gym

# 2. 动画
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import HTML, display


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    以gif格式显示关键帧列，具有控制
    """

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        img = patch.set_data(frames[i])
        return img  ## *** return是必须要有的 ***

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    anim.save('media/movie_cartpole_DQN.mp4')
    ## display(display_animation(anim, default_mode='loop'))  ## *** delete ***
    return HTML(anim.to_jshtml())  ## *** 返回一个HTML对象，以便被调用者显示。 ***



# 然后，实现一个 namedtuple（具名元组） 用例
# 这段代码使用的是 namedtuple
# 可以使用 namedtuple 与字段名称成对存储值
# 按字段名称访问值很方便
# 原书提供链接：https://docs.python.jp/3/library/collections.html#collections.namedtuple
# 中文文档链接：https://docs.python.org/zh-cn/3/library/collections.html#collections.namedtuple
# 以下是一个用法示例

from collections import namedtuple

Tr = namedtuple('tr', ('name_a', 'value_b'))
Tr_object = Tr('名称为A', 100)

print(Tr_object)  # 输出：tr(name_a='名称为A'，value_b=100)
print(Tr_object.value_b)  # 输出：100

# 使用 namedtuple 转换 Tr_object
# 键名 name_a，name_b
# 可以通过键名访问每个值
# 使用 namdtuple 转换每个步骤的 transition
# 以便实现 DQN 时更容易访问状态和动作值



# 3. 生成 namedtuple
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward')
)

# 4. 常量
# 修复：替换为新版 CartPole-v1，消除过时警告
ENV = 'CartPole-v1'
GAMMA = 0.9
MAX_STEPS = 500  # 修复：v1 版本最大步数为500，匹配新版环境
NUM_EPISODES = 500



# 5. ReplayMemory 存储经验数据
'''
为了实现小批量学习，定义内存类 ReplayMemory 来存储经验数据

push 函数，用于保存该步骤中的 transition 作为经验
sample 函数，随机选择 transition
len 函数，返回当前存储的 transition 数

如果存储的 transition 数大于常量 CAPACITY，则将索引返回到前面并覆盖旧内容
'''
class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY    # 下面 memory 的最大长度
        self.memory = []    # 存储过往经验
        self.index = 0  # 表示要保存的索引

    def push(self, state, action, state_next, reward):
        '''将 transition = (state, action, state_next, reward) 保存在存储器中'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 内存未满时添加

        # 使用具名元组对象 Transition 将值和字段名称保存为一对
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 索引加一

    def sample(self, batch_size):
        '''随机检索 Batch_size 大小的样本并返回'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''返回当前 memory 长度'''
        return len(self.memory)




# 执行 DQN 的 Brain 类
# 将 Q 函数定义为深度学习网络（而非一个表格）

# 包

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# 常量
BATCH_SIZE = 32
CAPACITY = 10000

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # CartPole 的两个动作

        # 创建存储经验的对象
        self.memory = ReplayMemory(CAPACITY)

        # 构建一个神经网络
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)  # 输出网络的形状

        # 最优化方法的设定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        '''通过 Experience Replay（经验回放） 学习网络的连接参数'''

        # -----------------------------------------
        # 1. 检查经验池的大小
        # -----------------------------------------
        # 1.1 经验池大小小于批量数据时不执行任何操作
        if len(self.memory) < BATCH_SIZE:
            return

        # -----------------------------------------
        # 2. 创建小批量数据
        # -----------------------------------------
        # 2.1 从经验池中获取小批量数据
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 将每个变量转换为与小批量
        # 得到的 transitions 存储了一个 BATCH_SIZE 大小的 (state, action, state_next, reward)
        # 即：BATCH_SIZE * (state, action, state_next, reward)
        # 想把它变成小批量数据，换句话说：
        # 转为 (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 2.3 将每个变量转换为与小批量数据对应的形式
        # 例如，对于 state，形状为 [torch.FloatTensor of size 1x4]
        # 将其转换为 torch.FloatTensor of size BATCH_SIZE * 4
        # cat 是指 Concatenates（连接）
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # 只收集下一个状态是否存在的变量：
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        # -----------------------------------------
        # 3. 求取 Q(s_t, a_t)值作为监督信号
        # -----------------------------------------
        # 3.1 将网络切换到推理模式
        self.model.eval()

        # 3.2 求取网络输出的 Q(s_t, a_t)
        # self.model(state_batch)输出左右两个 Q 值
        # 成为[torch.FloatTensor of size BATCH_SIZEx2]
        # 为了求得于此处执行的动作 a_t 对应的 Q 值，
        # 求取由 action_batch 执行的动作 a_t 是向右还是向左的 index
        # 用 gather 获得相应的 Q 值。
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # 3.3 求max{Q(s_t+1, a)}。
        # 需要注意下一个状态s_t+1，不存在下一个状态时为 0

        # 创建索引掩码以检查 cartple 是否未完场且具有 next_state
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        # 首先全部设置为 0
        next_state_values = torch.zeros(BATCH_SIZE)

        # 求取具有下一状态的 index 的最大 Q 值
        # 访问输出并通过 max() 求列方向最大值的 [value, index]
        # 并输出其 Q 值（index = 0）
        # 用 detach 取出该值
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        # 3.4 从 Q 公式中求取 Q(s_t, a_t) 值作为监督信息
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # -----------------------------------------
        # 4. 更新连接参数
        # -----------------------------------------
        # 4.1 切换到训练状态
        self.model.train()

        # 4.2 计算损失函数（smooth_l1_loss 是 Huberloss）
        # expected_state_action_values 的 size 是 [minbatch]，
        # 通过 unsqueeze 得到 [minibatch x 1]
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # 4.3 更新连接参数
        self.optimizer.zero_grad()  # 重置渐变
        loss.backward()  # 计算反向传播
        self.optimizer.step()  # 更新连接参数

    def decide_action(self, state, episode):
        '''根据当前状态确定动作'''
        # 使用 ε-贪婪法 逐步采用最佳动作
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # 将网络切换到推理模式
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
            # 获取网络输出最大值的 索引 index = max(1)[1]
            # .view(1,1) 将 [torch.LongTensor of size 1] 转换为 size 1x1 大小

        else:
            # 随即返回 0,1 的动作
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 随机返回 0,1 动作
            # action 的形式为 [torch.LongTensor of size 1x1]

        return action






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


# 执行 CartPole 的环境类


class Environment:

    def __init__(self):
        # 修复：添加 render_mode 确保渲染正常，兼容新版 Gymnasium
        self.env = gym.make(ENV, render_mode='rgb_array')  # 设定任务
        num_states = self.env.observation_space.shape[0]  # 获得任务的状态变量数 4
        num_actions = self.env.action_space.n  # CartPole的动作数 2
        self.agent = Agent(num_states, num_actions)  # 创建 Agent，在环境中执行动作

    def run(self):
        '''执行'''
        episode_10_list = np.zeros(10)  # 存储 10 次试验的连续站立步数，用于输出平均步数
        complete_episodes = 0  # 连续 195 步以上统计
        episode_final = False  # 最终尝试标记
        frames = []  # 存储图像的变量

        for episode in range(NUM_EPISODES):  # 最大重复试验次数
            # 修复1：拆分 reset 返回的元组 (state, info)，解决 tuple 转 ndarray 错误
            observation, info = self.env.reset()  # 环境初始化

            state = observation  # 直接将观测作为状态 state 使用
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  # NumPy 变量转换为 PyTorch Tensor
            state = torch.unsqueeze(state, 0)  # FloatTensor of size 4 转换为 size 1x4

            for step in range(MAX_STEPS):  # 1 轮循环（1 episode）

                if episode_final is True:  # 最终试验时添加各时刻图像
                    frames.append(self.env.render())  # 修复：新版 render 无需 mode 参数

                action = self.agent.get_action(state, episode)  # 求取动作

                # 修复2：拆分 step 返回的 5 个值，兼容新版 Gymnasium API
                # 通过执行动作 a_t 求 s_{t+1} 和 done 标志
                # 从 action 中指定 .item() 并获取内容
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
                    state_next = observation_next  # 保持观察不变
                    state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor)  # numpy 变量 --> PyTorch 变量
                    state_next = torch.unsqueeze(state_next, 0)  # FloatTensor of size 4 转为 size 1x4

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
                    break

            # 修复：v1 版本连续 10 轮达到 475 步即判定成功（对应 v0 的 195 步）
            if complete_episodes >= 10:
                print('10 轮连续成功')
                episode_final = True  # 标记下一次为最终试验

            if episode_final is True:
                # 保存并绘制动画
                self.env.close()  # 关闭环境
                html = display_frames_as_gif(frames)
                display(html)  # 修复：显式显示 HTML 动画
                break

# 主函数
if __name__ == "__main__":
    # 修复：添加设备兼容，避免 CUDA/CPU 问题
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cartpole_env = Environment()
    cartpole_env.run()
