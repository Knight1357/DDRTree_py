import numpy as np
from scipy.linalg import eigh  # 用于进行特征分解
from scipy.stats import norm   # 用于计算正态分布分位数
from scipy.sparse.linalg import svds  # 近似奇异值分解，类似于R中的irlba
from sklearn.cluster import KMeans
from utils import time_func
from loguru import logger

@time_func
def pca_projection_python(C, L):
    logger.warning(f"开始python pca降维")
    # C: 用于PCA的数据矩阵
    # L: 要计算的主成分的数量
    num_features, num_samples = C.shape
    
    # 判断L是否大于等于矩阵C的最小维度
    if L >= min(num_features, num_samples):
        # 计算矩阵C的特征值和特征向量
        if num_features < num_samples:
            cov_matrix = np.cov(C.T)
        else:
            cov_matrix = np.cov(C)
            
        eigen_values, eigen_vectors = eigh(cov_matrix)
        
        # 按特征值降序排序，获取排序的索引
        sorted_indices = np.argsort(eigen_values)[::-1]
        
        # 获取前L个对应最大特征值的特征向量
        W = eigen_vectors[:, sorted_indices[:L]]
        return W
    else:
        # 确保生成的初始向量具有正确的大小
        n_features = C.shape[1]
        initial_v = norm.ppf(np.linspace(1 / (n_features + 1), 1, n_features))
        
        # 使用稀疏SVD进行近似PCA，当L小于维度时
        u, s, vt = svds(C, k=L, v0=initial_v)

        logger.warning(f"结束python pca降维")
        # 返回前L个右奇异向量（在R中是V）
        return vt.T


def DDRTree(X,
            dimensions=2,
            initial_method=None,
            maxIter=20,
            sigma=1e-3,
            lambda_param=None,
            ncenter=None,
            param_gamma=10,
            tol=1e-3,
            verbose=False,
            **kwargs):
    """
    执行 DDRTree 构建.
    
    参数:
    X: 需要进行 DDRTree 构建的 D x N 矩阵.
    dimensions: 降维的维数.
    initial_method: 当为 None 时，使用 PCA 降维。否则使用提供的方法降维.
    maxIter: 最大迭代次数.
    sigma: 带宽参数.
    lambda_param: 逆图嵌入的正则化参数.
    ncenter: 正则化图中允许的节点数.
    param_gamma: k-means 的正则化参数.
    tol: 相对目标差异.
    verbose: 输出调试信息.
    **kwargs: 传递给初始方法的其他参数.
    
    返回:
    包含 W, Z, stree, Y, history 的字典.
    """
    
    D, N = X.shape

    # 初始化
    W = pca_projection_python(np.dot(X, X.T), dimensions)
    if initial_method is None:
        Z = np.dot(W.T, X)  # 矩阵乘法（转置）
    else:
        # 使用提供的初始方法
        tmp = initial_method(X, **kwargs)
        if tmp.shape[1] > D or tmp.shape[0] > N:
            raise ValueError('降维方法返回的维度不正确')
        Z = tmp[:, :dimensions].T  # 转置成维度正确的格式

    if ncenter is None:
        K = N
        Y = Z[:, :K]
    else:
        K = ncenter
        if K > Z.shape[1]:
            raise ValueError("错误: ncenter 必须大于等于 ncol(X)")
        centers = Z[:, np.linspace(0, Z.shape[1]-1, K, dtype=int)].T
        kmeans = KMeans(n_clusters=K, init=centers)
        kmeans.fit(Z.T)
        
        Y = kmeans.cluster_centers_.T  # 转置成 D x K

    # 默认 lambda_param
    if lambda_param is None:
        lambda_param = 5 * N

    from ddr_tree import DDRTree_reduce_dim
    # 修改为调用 C++ 绑定的函数
    ddrtree_res = DDRTree_reduce_dim(X, Z, Y, W, dimensions, maxIter, K, sigma, lambda_param, param_gamma, tol, verbose)
    
    return {
        'W': ddrtree_res['W'],
        'Z': ddrtree_res['Z'],
        'stree': ddrtree_res['stree'],
        'Y': ddrtree_res['Y'],
        'X': ddrtree_res['X'],
        'R': ddrtree_res['R'],
        'Q': ddrtree_res['Q'],
        'objective_vals': ddrtree_res['objective_vals'],
        'history': None
    }

# 示例调用
# import pandas as pd
# from sklearn.datasets import load_iris

# iris = load_iris()
# subset_iris_mat = iris.data[[0, 1, 51, 102], :].T  # 子集数据
# DDRTree_res = DDRTree(subset_iris_mat, dimensions=2, maxIter=5, sigma=1e-2, lambda_param=1, ncenter=3, param_gamma=10, tol=1e-2, verbose=False)
# Z = DDRTree_res['Z']
# Y = DDRTree_res['Y']
# stree = DDRTree_res['stree']

# 绘图部分可以使用 matplotlib 进行可视化
