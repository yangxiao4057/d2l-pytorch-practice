import numpy as np
import torch


numbers = [1, 2, 3]
array = np.array(numbers)
tensor = torch.tensor(numbers)

print("=== 类型 ===")
print("列表:", type(numbers))
print("NumPy:", type(array))
print("Tensor:", type(tensor))

print("\n=== 自己加自己 ===")
print("列表:", numbers + numbers)
print("NumPy:", array + array)
print("Tensor:", tensor + tensor)

print("\n=== 乘以 2 ===")
print("列表:", numbers * 2)
print("NumPy:", array * 2)
print("Tensor:", tensor * 2)

print("\n=== 形状与数据类型 ===")
print("列表长度:", len(numbers))
print("NumPy:", array.shape, array.dtype)
print("Tensor:", tensor.shape, tensor.dtype, tensor.device)

print("\n=== 转换成浮点数 ===")
float_array = array.astype(np.float32)
float_tensor = tensor.to(torch.float32)

print("NumPy:", float_array, float_array.dtype)
print("Tensor:", float_tensor, float_tensor.dtype)