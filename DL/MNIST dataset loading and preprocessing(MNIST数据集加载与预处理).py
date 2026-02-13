# 导入必要的库（注意：不要导入多余的mnist模块）
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

# 验证sklearn版本
import sklearn
print(f"scikit-learn 版本：{sklearn.__version__}")

# 核心：先正确加载MNIST数据集，并重命名变量避免冲突
print("正在加载MNIST数据集...")
# 加载数据集，变量名用 mnist_dataset 而非 mnist，避免和模块名冲突
mnist_dataset = fetch_openml('mnist_784', data_home=".", parser="auto")

# 提取特征和标签，转为numpy数组（关键）
X = mnist_dataset.data.values  # 特征数据（70000, 784）
y = mnist_dataset.target.values  # 标签数据（70000,）

# 归一化像素值到0-1区间（修复你代码中的 x = mnist.data / 255 问题）
X = X.astype(np.float32) / 255.0
y = y.astype(np.int32)

# 验证数据
print("\n数据集基本信息：")
print(f"特征数据形状：{X.shape}")
print(f"标签数据形状：{y.shape}")

# 可视化前5张图片
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = X[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
