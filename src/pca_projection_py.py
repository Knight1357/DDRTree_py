import numpy as np
import ddr_tree  # 替换为实际绑定的模块名

# 准备测试数据
# 示例矩阵 R_C，维度为 (N, D)，其中 N 是样本数，D 是特征数
R_C = np.array([[2.5, 0.5, 2.2, 1.9],
                [1.5, 1.7, 3.0, 1.6],
                [3.2, 3.1, 1.6, 2.7],
                [4.5, 2.9, 1.9, 1.5]])

# 设置降维目标维度
dimensions = 2

# 调用绑定的 PCA 投影函数
result = ddr_tree.pca_projection(R_C, dimensions)

# 打印结果
print("Input Matrix R_C:")
print(R_C)
print(f"Projected Matrix (Reduced to {dimensions} dimensions):")
print(result)
