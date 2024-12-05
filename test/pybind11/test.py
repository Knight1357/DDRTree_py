# 导入 pybind11 模块
import example
import numpy as np

result = example.add(3, 5)
print(f"The result of adding 3 and 5 is: {result}")


# 创建一个随机的 2D numpy 数组
input_array = np.random.rand(100, 50).astype(np.float64)

# 调用测试函数
result = example.test_array_input(input_array)

# 打印结果
print("Test result:", result)
