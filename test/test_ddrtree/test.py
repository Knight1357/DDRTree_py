import numpy as np
from ddr_tree import DDRTree_reduce_dim

# 设置随机种子
np.random.seed(42)

# 参数设置
n_samples = 50      # 样本数量（N）
n_features = 10     # 原始特征维度（D）
dimensions = 2      # 降维后的维度（d）
num_clusters = 3    # 聚类数量（K）
maxiter = 10        # 最大迭代次数
sigma = 1.0         # 高斯核参数（调整为较大值，避免数值问题）
lambda_ = 0.1       # 正则化参数
gamma = 0.1         # 权重参数
eps = 1e-5          # 收敛阈值
verbose = True      # 是否输出详细信息

# 生成可控的随机数据
R_X = np.random.rand(n_features, n_samples)          # (D x N)
R_Z = np.random.rand(dimensions, n_samples)          # (d x N)
R_Y = np.random.rand(dimensions, num_clusters)       # (d x K)
R_W = np.random.rand(n_features, dimensions)         # (D x d)

# 检查输入数据是否有异常值
for name, matrix in zip(['R_X', 'R_Z', 'R_Y', 'R_W'], [R_X, R_Z, R_Y, R_W]):
    assert not np.any(np.isnan(matrix)), f"{name} contains NaN values"
    assert not np.any(np.isinf(matrix)), f"{name} contains Inf values"

# 调用函数
result = DDRTree_reduce_dim(
    R_X, R_Z, R_Y, R_W,
    dimensions, maxiter, num_clusters,
    sigma, lambda_, gamma, eps, verbose
)

# 输出结果
print("-------------------------------------------------------------")
print("W shape:", result['W'].shape)
print("Z shape:", result['Z'].shape)
print("stree shape:", result['stree'].shape)
print("Y shape:", result['Y'].shape)
print("Q shape:", result['Q'].shape)
print("R shape:", result['R'].shape)
print("Objective values:", result['objective_vals'])
