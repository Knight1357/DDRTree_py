import ddr_tree
import numpy as np

# 示例数据
X = np.random.rand(10, 5)  # 10个样本，5个特征
Z = np.random.rand(10, 3)
Y = np.random.rand(10, 2)
W = np.random.rand(10, 2)

# 调用 DDRTree 减维函数
result = ddr_tree.DDRTree_reduce_dim(X, Z, Y, W, dimensions=3, maxiter=100, num_clusters=3, sigma=0.5, lambda_=0.1, gamma=0.1, eps=1e-6, verbose=True)

# 输出结果
print(result["W"])
