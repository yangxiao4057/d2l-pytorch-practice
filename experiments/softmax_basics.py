import torch
from torch import nn


# 1. 两个样本，每个样本有三个类别的原始分数（logits）
# 第0维是样本，第1维是类别；类别编号从0开始。
logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, 1.0],
], requires_grad=True)
labels = torch.tensor([0, 2], dtype=torch.long)

print("logits形状：", logits.shape)
print("真实类别：", labels)


# 2. 从公式实现Softmax
# 每行减去该行最大值，避免指数运算溢出；不改变概率。
def softmax(X):
    shifted = X - X.max(dim=1, keepdim=True).values
    exp_X = torch.exp(shifted)
    return exp_X / exp_X.sum(dim=1, keepdim=True)


probs = softmax(logits)
print("\n分类概率：\n", probs.detach())
print("每行概率之和：", probs.sum(dim=1).detach())
print("与torch.softmax一致：", torch.allclose(probs, torch.softmax(logits, dim=1)))


# 3. 最大概率对应预测类别；Softmax不改变分数的大小顺序
predictions = probs.argmax(dim=1)
accuracy = (predictions == labels).float().mean()
print("\n预测类别：", predictions)
print("准确率：", accuracy.item())


# 4. 给同一样本的所有分数加相同常数，概率保持不变
shifted_probs = softmax(logits + 10)
print("\n分数全部加10后概率不变：", torch.allclose(probs, shifted_probs))


# 5. 交叉熵：取真实类别对应的概率，再计算负对数
# 这里用于理解公式；实际训练优先使用下方数值稳定的CrossEntropyLoss。
true_class_probs = probs[torch.arange(labels.shape[0]), labels]
per_sample_loss = -torch.log(true_class_probs)
manual_loss = per_sample_loss.mean()
print("\n真实类别对应的概率：", true_class_probs.detach())
print("每个样本的交叉熵：", per_sample_loss.detach())
print("平均交叉熵：", manual_loss.item())


# 6. CrossEntropyLoss直接接收logits，不要先做Softmax
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)
print("PyTorch交叉熵：", loss.item())
print("两种损失一致：", torch.allclose(manual_loss, loss))


# 7. 反向传播：这里只观察logits梯度，还没有训练模型参数
loss.backward()
print("\nlogits梯度：\n", logits.grad)
