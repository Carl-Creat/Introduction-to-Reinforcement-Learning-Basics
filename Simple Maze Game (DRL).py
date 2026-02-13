import numpy as np
import random
import matplotlib.pyplot as plt

#初始化设置
#网格环境初始化
grid_size=3                           #nxn网格
actions=[0, 1, 2, 3]                  #4个动作：0=上，1=下，2=左，3=右
obstacle=[(1,1),(0,2)]               #障碍物设置
goal=(grid_size-1, grid_size-1)       #终点设置
action_names=['上', '下', '左', '右']

#奖励矩阵初始化：
rewards=np.zeros((grid_size, grid_size))

#奖励规则设置
for r in range(grid_size):
    for c in range(grid_size):
        if (r, c)==goal:
            rewards[r, c]=10
        elif (r, c) in obstacle:
            rewards[r, c]=-5
        else:
            l=((r-goal[0])**2+(c-goal[1])**2)**0.5
            rewards[r, c]=-0.1*l

#Q表初始化
Q=np.zeros((grid_size, grid_size, len(actions)))

#参数设置
alpha=0.1             #学习率
gamma=0.9             #折扣因子
epsilon=0.2           #初始探索率
num_episodes=1000     #训练回合
epsilon_min=0.01      #探索率最低值
epsilon_decay=0.995   #每回合探索率降低0.001

#定义一个空间状态函数
def get_new_state(state, action):
    r,c=state
    if action==0:
        return max(0, r-1), c
    elif action==1:
        return min(grid_size-1, r+1), c
    elif action==2:
        return r, max(0, c-1)
    elif action==3:
        return r, min(grid_size-1, c+1)

#可视化设置,进行绘图
plt.ion()     #开启实时绘图
fig,(ax1, ax2)=plt.subplots(1, 2, figsize=(10,5))  #1行2列的图,左边用来绘制奖励函数，右边绘制运动轨迹

#左图：奖励值随训练次数的变化
reward_record=[]       #记录每回合的总奖励
avg_reward_record=[]   #记录每100次的奖励平均值
reward_line, = ax1.plot([], [], 'b-', linewidth=0.5)  #在左图中用蓝色细线进行绘制
#设置第一个图的标题及横纵坐标
ax1.set_title('total_reward_per_round', fontsize=16)
ax1.set_xlabel('training_round')
ax1.set_ylabel('reward')
ax1.grid(True, alpha=0.3)     #设置网格进行辅助
ax1.set_xlim(0, num_episodes) #设置x轴范围
ax1.set_ylim(-20, 20)         #设置y轴范围

#右图：网格和智能体位置动图
#先画静态网格：
for r in range(grid_size):
    for c in range(grid_size):
        if (r, c)==goal:
            color = 'yellow'  #终点黄色
        elif (r, c) in obstacle:
            color = 'black'   #障碍物黑色
        else:
            color = 'white'   #普通格子白色
        #画正方形格子
        ax2.fill([c, c+1, c+1, c], [grid_size-1-r, grid_size-1-r, grid_size-r, grid_size-r],
                 color, edgecolor='black', linewidth=2)

ax2.set_title('Real-Time Location of Intelligent Agent', fontsize=16)
ax2.set_aspect('equal')           #正方形网格
ax2.set_xticks([])                #隐藏坐标轴
ax2.set_yticks([])
agent_dot, = ax2.plot([], [], 'rs', markersize=15)  #红色方框代表智能体

#进行循环训练
print(f"开始训练（共{num_episodes}回合）")
for episode in range(num_episodes):
    current_state=(0,0)       #每次从起点开始
    prev_state=None           #记录上一个位置，用于判断是否后退，如果后退就给负奖励
    total_reward=0            #初始化总奖励，用于记录当前回合的总奖励
    #贪婪策略：
    #探索率衰减
    current_epsilon=max(epsilon_min, epsilon * (epsilon_decay ** episode))

    while current_state != goal:
        if random.uniform(0, 1) < current_epsilon:
            action = random.choice(actions)  # 随机探索
        else:
            action = np.argmax(Q[current_state[0], current_state[1]])  #选Q值最高的动作

        #执行动作，得到新位置
        new_state = get_new_state(current_state, action)

        #计算奖励：基础奖励+后退惩罚
        base_reward = rewards[new_state]            #格子本身的奖励（终点/障碍/距离奖励）
        back_penalty = 0                            #后退惩罚初始为0
        #判断是否返回上一位置（prev_state不为None时才判断，避免起点误判）
        if prev_state is not None and new_state==prev_state:
            back_penalty=-5  #说明智能体返回上一步，进行扣分
        total_reward += base_reward + back_penalty  #累计总奖励

        #更新Q表
        #新Q值=旧Q值+学习率×(当前总奖励+未来最好分数-旧分数)
        old_q = Q[current_state[0], current_state[1], action]
        best_future_q = np.max(Q[new_state[0], new_state[1]])          #未来能拿到的最高Q值
        Q[current_state[0], current_state[1], action] = old_q + alpha * ((base_reward + back_penalty) + gamma * best_future_q - old_q)

        #更新动图中智能体位置
        x = new_state[1] + 0.5                  #列→x轴（加0.5是为了在格子中心）
        y = grid_size - 1 - new_state[0] + 0.5  #行→y轴（网格坐标系转换）
        agent_dot.set_data([x], [y])

        #更新位置：当前位置变上一个位置，新位置变当前位置
        prev_state = current_state
        current_state = new_state

        #快速刷新画面
        plt.pause(0.0001)

    #记录当前回合奖励，更新奖励曲线
    reward_record.append(total_reward)
    reward_line.set_data(range(episode+1), reward_record)
    ax1.relim()          #自动调整坐标轴
    ax1.autoscale_view()
    fig.canvas.draw()

    #更新标题，显示当前进度
    ax2.set_title(f'round{episode+1} | exploration rate{current_epsilon:.3f}')

    # 每100回合打印一次进度
    if (episode+1) % 100 == 0:
        avg_reward = np.mean(reward_record[-100:])
        print(f"第{episode+1}回合--平均奖励：{avg_reward:.2f}")
        avg_reward_record.append(avg_reward)   #将当前100回合平均奖励存入列表

    plt.pause(0.001)

#训练结束，显示最优路径
plt.ioff()  # 关闭实时绘图
print("\n训练结束！绘制最优路径")

#重新画网格，显示最优路径
ax2.clear()
for r in range(grid_size):
    for c in range(grid_size):
        if (r, c) == goal:
            color = 'yellow'
        elif (r, c) in obstacle:
            color = 'black'
        else:
            color = 'white'
        ax2.fill([c, c+1, c+1, c], [grid_size-1-r, grid_size-1-r, grid_size-r, grid_size-r],
                 color, edgecolor='black', linewidth=2)

#计算最优路径：只选Q值最高的动作
optimal_path=[]
current_state=(0, 0)
optimal_path.append(current_state)
while current_state != goal:
    action = np.argmax(Q[current_state[0], current_state[1]])
    current_state = get_new_state(current_state, action)
    optimal_path.append(current_state)

#绘制最优路径：
x_path = [p[1] + 0.5 for p in optimal_path]
y_path = [grid_size - 1 - p[0] + 0.5 for p in optimal_path]
ax2.plot(x_path, y_path, 'r-', linewidth=3, marker='s', markersize=8, label='optimal path')
ax2.legend()
ax2.set_title('Optimal Path', fontsize=12)
ax2.set_aspect('equal')
ax2.set_xticks([])
ax2.set_yticks([])

#输出结果
print("\n学习结果如下：")
print("Q表（每个位置的4个动作Q值：[上、下、左、右]）：")
for r in range(grid_size):
    for c in range(grid_size):
        if (r, c) in obstacle:
            print(f"位置({r},{c})：障碍物")
        elif (r, c) == goal:
            print(f"位置({r},{c})：终点")
        else:
            q_scores = Q[r, c].round(2)
            best_action = action_names[np.argmax(Q[r, c])]
            print(f"位置({r},{c})：{q_scores} → 最优动作：{best_action}")

print(f"\n最优路径：{' → '.join([f'({r},{c})' for r, c in optimal_path])}")

#训练结束后弹出新窗口，绘制每100回合平均奖励曲线
plt.figure(figsize=(8, 5))                                    #新建窗口（独立于之前的双图窗口）
x_data = [i*100 for i in range(1, len(avg_reward_record)+1)]  #x轴
plt.plot(x_data, avg_reward_record, 'b-o', linewidth=2, markersize=6)
plt.title('Average Reward per 100 Episodes', fontsize=14)
plt.xlabel('Training Episodes', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.grid(True, alpha=0.3)

#显示最终图表
plt.tight_layout()
plt.show()
