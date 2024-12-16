import numpy as np
from py_ddr_tree import DDRTree

np.random.seed(42)
# 假设你已经有一个用于测试的高维数据矩阵 X
# 例如：生成一个 100 个样本，10 个特征的随机数据\
N = 5000  # 样本数量
D = 2000   # 特征数量
X = np.random.rand(D, N)  # D=10, N=100

# 定义要传递给DDRTree的参数
dimensions = 2  # 这里确保 dimensions <= D
maxIter = 20  # 最大迭代次数
sigma = 1e-3  # 带宽参数
lambda_param = 0.1  # 正则化参数
ncenter = 1000  # 聚类中心的数量
param_gamma = 10  # k-means的正则化参数
tol = 1e-3  # 相对目标差异
verbose = True  # 输出调试信息

# 调用 DDRTree 函数
result = DDRTree(
    X, 
    dimensions=dimensions,
    maxIter=maxIter,
    sigma=sigma,
    lambda_param=lambda_param,
    ncenter=ncenter,
    param_gamma=param_gamma,
    tol=tol,
    verbose=verbose
)

# 打印返回的结果
print("降维的特征 W:\n", result['W'])
print("降维后的数据 Z:\n", result['Z'])
print("树结构 stree:\n", result['stree'])
print("聚类中心 Y:\n", result['Y'])
print("原始输入的维数:\n", result['X'].shape)
print("附加输出 R:\n", result['R'])
print("附加输出 Q:\n", result['Q'])
print("目标函数值:\n", result['objective_vals'])
