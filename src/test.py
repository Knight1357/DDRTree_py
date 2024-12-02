import numpy as np
import ddr_tree

def check_array(name, arr, expected_shape):
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be a numpy array.")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D. Got: {arr.ndim}D.")
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}. Got: {arr.shape}.")
    if arr.dtype != np.float64:
        raise TypeError(f"{name} must have dtype numpy.float64. Got: {arr.dtype}.")

# 模拟输入数据
X = np.random.rand(5, 8).astype(np.float64)  # 输入数据
Z = np.random.rand(10, 8).astype(np.float64)  # 初始中心点
Y = np.random.rand(10, 2).astype(np.float64)  # 降维表示
W = np.random.rand(5, 8).astype(np.float64)  # 权重矩阵

# 检查数据
check_array("X", X, (5, 8))
check_array("Z", Z, (10, 8))
check_array("Y", Y, (10, 2))
check_array("W", W, (5, 8))

# 参数
dimensions = 2
maxiter = 20
num_clusters = 10
sigma = 1e-3
lambda_ = 5.0
gamma = 10.0
eps = 1e-3
verbose = True

# 调用函数
result = ddr_tree.DDRTree_reduce_dim(
    X, Z, Y, W,
    dimensions, maxiter, num_clusters,
    sigma, lambda_, gamma, eps,
    verbose
)

# 查看结果
print("Projection matrix W:\n", result["W"])
print("Reduced dimensions Z:\n", result["Z"])
print("Spanning tree:\n", result["stree"])
print("The End")
