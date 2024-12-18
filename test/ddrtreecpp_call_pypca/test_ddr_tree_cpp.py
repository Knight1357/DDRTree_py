import numpy as np
import sys
import os
from loguru import logger
try:
    # 获取当前脚本路径，添加共享库所在的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(current_dir, "DDRTree_py")
    sys.path.append(so_path)
    import DDRTree_cpp
    from utils import time_func
except ImportError:
    print("无法导入 ddr_tree 模块或 DDRTree_reduce_dim 函数，请检查模块是否安装正确及函数名是否准确。")
    raise


@time_func
def test_ddr_tree_cpp():

    # 设置随机种子
    np.random.seed(42)

    # 参数设置
    n_samples = 30      # 样本数量（N）
    n_features = 10    # 原始特征维度（D）
    num_clusters = 2   # 聚类数量（K）
    dimensions = 5      # 降维后的维度（d）
    maxiter = 20        # 最大迭代次数
    sigma = 1e-3        # 高斯核参数
    lambda_ = 0.1       # 正则化参数
    gamma = 10         # 权重参数
    eps = 1e-3          # 收敛阈值
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
    logger.warning(f"开始调用 DDRTree_reduce_dim 函数")
    result = DDRTree_cpp.DDRTree_reduce_dim(
        R_X, R_Z, R_Y, R_W,
        dimensions, maxiter, num_clusters,
        sigma, lambda_, gamma, eps, verbose
    )

    logger.warning(f"开始调用 DDRTree_reduce_dim 函数结束")
    # 输出结果
    print("-------------------------------------------------------------")
    print("W shape:", result['W'].shape)
    print("Z shape:", result['Z'].shape)
    print("stree shape:", result['stree'].shape)
    print("Y shape:", result['Y'].shape)
    print("X shape:", result['X'].shape)
    print("Q shape:", result['Q'].shape)
    print("R shape:", result['R'].shape)
    print("Objective values:", result['objective_vals'])


if __name__ == "__main__":
    test_ddr_tree_cpp()
