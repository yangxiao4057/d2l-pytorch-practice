import random
import torch

# 固定随机种子，方便复现
random.seed(42)
torch.manual_seed(42)


# 1. 生成模拟数据
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)


# 2. 分批读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


batch_size = 10


# 3. 初始化模型参数
w = torch.normal(
    0, 0.01, size=(2, 1), requires_grad=True
)
b = torch.zeros(1, requires_grad=True)


# 4. 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 5. 定义损失函数：返回每个样本的平方误差的一半
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 6. 定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 7. 训练模型
lr = 0.03
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # 计算每个样本的损失
        l = squared_loss(linreg(X, w, b), y)

        # 对损失求和，再计算梯度
        l.sum().backward()

        # 更新参数，并清空梯度
        sgd([w, b], lr, X.shape[0])

    # 查看全部训练数据的平均损失
    with torch.no_grad():
        train_l = squared_loss(
            linreg(features, w, b), labels
        )
        print(
            f"epoch {epoch + 1}, "
            f"loss {train_l.mean().item():.6f}"
        )


# 8. 查看学习结果
print("\n真实权重：", true_w)
print("学到的权重：", w.detach().reshape(-1))
print("真实偏置：", true_b)
print("学到的偏置：", b.item())


# 9. 使用模型预测
with torch.no_grad():
    new_X = torch.tensor([[1.0, 2.0]])
    prediction = linreg(new_X, w, b)
    print("\n输入：", new_X)
    print("预测结果：", prediction.item())