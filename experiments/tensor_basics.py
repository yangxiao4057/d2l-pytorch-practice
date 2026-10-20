import torch


# 1. 创建一个两行三列的 Tensor
X = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6]],
    dtype=torch.float32,
)

print("=== 基本信息 ===")
print("X:\n", X)
print("shape:", X.shape)
print("dtype:", X.dtype)
print("device:", X.device)
print("维度数量:", X.ndim)
print("元素数量:", X.numel())


# 2. 索引与切片
print("\n=== 索引与切片 ===")

first_row = X[0]
first_row_matrix = X[0:1]
first_column = X[:, 0]

print("X[0]:", first_row)
print("X[0].shape:", first_row.shape)

print("X[0:1]:", first_row_matrix)
print("X[0:1].shape:", first_row_matrix.shape)

print("X[:, 0]:", first_column)
print("X[:, 0].shape:", first_column.shape)


# 3. 逐元素运算
print("\n=== 逐元素运算 ===")
print("X + 10:\n", X + 10)
print("X * 2:\n", X * 2)
print("X * X:\n", X * X)


# 4. 矩阵乘法
print("\n=== 矩阵乘法 ===")

w = torch.tensor(
    [[1.0],
     [2.0],
     [3.0]]
)

result = X @ w

print("X.shape:", X.shape)
print("w.shape:", w.shape)
print("X @ w:\n", result)
print("结果形状:", result.shape)


# 5. 广播：给每一行加上同一个向量
print("\n=== 广播 ===")

offset = torch.tensor([10.0, 20.0, 30.0])
broadcast_result = X + offset

print("offset.shape:", offset.shape)
print("X + offset:\n", broadcast_result)
print("结果形状:", broadcast_result.shape)


# 6. 改变形状
print("\n=== 改变形状 ===")

flat = X.reshape(-1)

print("展平后的 Tensor:", flat)
print("展平后的形状:", flat.shape)
print("原来的 X.shape:", X.shape)