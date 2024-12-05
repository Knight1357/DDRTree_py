import numpy as np
import ddr_tree  # 替换为实际绑定的模块名

# 准备测试数据
R_a = np.array([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]])

R_b = np.array([[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]])

# 调用绑定函数
result = ddr_tree.sqdist(R_a, R_b)

# 打印结果
print("Squared Distance Matrix:")
print(result)
