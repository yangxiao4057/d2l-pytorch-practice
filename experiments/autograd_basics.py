import torch


# 1. 创建需要求导的标量
x = torch.tensor(2.0, requires_grad=True)
y = 3 * x ** 2 + 2 * x

print("x.shape:", x.shape)
print("y:", y.item())
print("y.requires_grad:", y.requires_grad)
print("y.grad_fn:", type(y.grad_fn).__name__)
print("反向传播前:", x.grad)


# 2. 第一次反向传播
y.backward()

print("第一次梯度:", x.grad.item())
print("x 仍然是:", x.item())


# 3. 重新进行前向计算，再反向传播
# 没有清零，所以梯度会累积
y2 = 3 * x ** 2 + 2 * x
y2.backward()

print("累积后的梯度:", x.grad.item())


# 4. 清零梯度
x.grad.zero_()

print("清零后:", x.grad.item())


# 5. 重新计算，验证清零的效果
y3 = 3 * x ** 2 + 2 * x
y3.backward()

print("重新计算的梯度:", x.grad.item())


# 6. 不记录计算图
with torch.no_grad():
    z = 3 * x ** 2 + 2 * x

print("z.requires_grad:", z.requires_grad)