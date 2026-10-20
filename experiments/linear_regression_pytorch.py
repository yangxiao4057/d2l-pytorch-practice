import torch
from torch import nn
from torch.utils import data

# 固定随机种子，方便复现
torch.manual_seed(42)


# 1. 生成模拟数据
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)


# 2. 创建数据加载器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train
    )


batch_size = 10
data_iter = load_array((features, labels), batch_size)


# 3. 定义模型
# 每个样本输入两个特征，输出一个预测值
net = nn.Sequential(nn.Linear(2, 1))


# 4. 初始化模型参数
with torch.no_grad():
    net[0].weight.normal_(0, 0.01)
    net[0].bias.fill_(0)


# 5. 定义损失函数：默认返回平均平方误差
loss = nn.MSELoss()


# 6. 创建优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# 7. 训练模型
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        # 计算当前批次的平均损失
        l = loss(net(X), y)

        # 清理旧梯度
        trainer.zero_grad()

        # 计算当前梯度
        l.backward()

        # 更新参数
        trainer.step()

    # 查看全部训练数据的平均损失
    with torch.no_grad():
        train_l = loss(net(features), labels)
        print(
            f"epoch {epoch + 1}, "
            f"loss {train_l.item():.6f}"
        )


# 8. 查看学习结果
print("\n真实权重：", true_w)
print("学到的权重：", net[0].weight.detach().reshape(-1))
print("真实偏置：", true_b)
print("学到的偏置：", net[0].bias.item())


# 9. 使用模型预测
with torch.no_grad():
    new_X = torch.tensor([[1.0, 2.0]])
    prediction = net(new_X)
    print("\n输入：", new_X)
    print("预测结果：", prediction.item())