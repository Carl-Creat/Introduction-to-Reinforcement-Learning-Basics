'''1.预处理数据'''
# 导入数据集，原书使用 fetch_mldata，新版本scikit-learn中已经没有这个包了
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True)

# 1.预处理数据（仅修改这2行！）
x = mnist.data.values / 255  # 加.values转为numpy数组，解决KeyError
y = mnist.target.values      # 加.values统一格式

# 可视化第一个 MNIST 数据（原代码完全不变）
import matplotlib.pyplot as plt

plt.imshow(x[0].reshape(28, 28), cmap='gray')
print("标签：{:}".format(y[0]))
plt.show()



'''2.创建DataLoader'''
# 2.创建 DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
# 2.1 将数据分成训练和测试（6:1）
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/7, random_state=0)

# 2.2 将数据转换为 tensor
x_train = torch.Tensor(x_train)
x_test = torch.Tensor(x_test)
y_train = torch.LongTensor(y_train.astype(float))
y_test = torch.LongTensor(y_test.astype(float))

# 2.3 使用一组数据和标签创建 Dataset
ds_train = TensorDataset(x_train, y_train)
ds_test = TensorDataset(x_test, y_test)

# 2.4 、使用小批量数据集创建 DataLoader
# 与 Chainer 中的 iterators.SerialIterator 类似
loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)


'''3.构建神经网络'''
# 构建网络
# Keras 风格
# 快速构建，使用 Sequential

from torch import nn
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))

print(model)



'''4.设置误差函数和优化方法'''
# 设置误差函数和优化方法
from torch import optim
# 误差函数，确定实际输出与期望输出之间的误差，这里使用 交叉熵
loss_fn = nn.CrossEntropyLoss() # 很多使用使用 criterion 作为变量名

# 学习权重参数时的优化方法，Adam 是一种梯度下降法
optimizer = optim.Adam(model.parameters(), lr=0.01)  #lr代表学习率

'''5.设置学习和推理'''
#设置学习和推理
# 5.1 定义学习 1 轮所做的事情

def train(epoch):
    model.train()   # 将网络切换到训练模式

    # 从数据加载器中取小批量数据进行计算
    for data, targets in loader_train:
        optimizer.zero_grad()   # 初始梯度设置为 0
        outputs = model(data)   # 输入数据并计算输出
        loss = loss_fn(outputs, targets)    # 计算输出和训练数据标签之间的误差
        loss.backward() # 对误差进行反向传播
        optimizer.step()    #更新权重

    print("epoch{}：结束\n".format(epoch))


# 5.2 定义 1 次推理中要做的事情
def test():
    model.eval()    # 将网络切换到推理模式
    correct = 0

    # 从数据加载器中取小批量数据进行计算
    with torch.no_grad():   # 输入数据并计算输出
        for data, targets in loader_test:
            outputs = model(data)   # 找到概率最高的标签

            # 推论
            _, predicted = torch.max(outputs.data, 1)   # 找到概率最高的标签
            correct += predicted.eq(targets.data.view_as(predicted)).sum()  # 如果计算结果和标签一致，计数加一

    data_num = len(loader_test.dataset) # 数据的总数
    print("\n 测试数据的准确率：{}/{}({:.0f}%)\n".format(correct, data_num, 100. * correct / data_num))



# 未训练时的准确率
test()  #用test函数计算正确答案的百分比；此时尚未进行连接参数的学习


'''6.执行学习和推理'''
# 执行学习和推理，对于60000个数据进行三轮学习
for epoch in range(3):
    train(epoch)

test()

# 推理特定图像数据
index = 800    # 举例，第2020个

model.eval()
data = x_test[index]
output = model(data)
_, predicted = torch.max(output.data, 0)
print("预测结果是{}".format(predicted))
x_text_show = (x_test[index]).numpy()
plt.imshow(x_text_show.reshape(28, 28), cmap='gray')
print("正确标签：{:}".format(y_test[index]))
plt.show()

# 如果要根据 Chainer 之类的输入实现灵活的计算
# 构建更灵活一些的网络

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)   # 与 Chainer 不同，PyTorch 中 None 不被接受
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        # forward 可以改变，以匹配输入 x
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

model = Net(n_in=28*28*1, n_mid=100, n_out=10)  # 创建网络对象
print(model)
