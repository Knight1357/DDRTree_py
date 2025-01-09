import numpy as np
import os
import sys
from scipy.linalg import eigh  # 用于进行特征分解
from scipy.stats import norm  # 用于计算正态分布分位数
from scipy.sparse.linalg import splu, svds, spsolve  # 近似奇异值分解，类似于R中的irlba
from sklearn.cluster import KMeans
from loguru import logger
from utils import time_func
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix

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


@time_func
def sqdist_python(a, b):
    """
    计算矩阵 a 和矩阵 b 之间的平方距离矩阵。

    参数：
        a (np.ndarray): 矩阵 a (形状为 m x n) 。
        b (np.ndarray): 矩阵 b (形状为 m x p) 。

    返回：
        W (np.ndarray): 矩阵 a 和矩阵 b 的平方距离矩阵（形状为 n x p) 。
    """
    # 计算矩阵 a 和 b 的列范数的平方
    # a 的列范数平方 (n,)
    aa = np.sum(np.square(a), axis=0)  # 每列的平方和
    # b 的列范数平方 (p,)
    bb = np.sum(np.square(b), axis=0)  # 每列的平方和

    # 计算矩阵乘积 a^T * b (n x p)
    ab = np.dot(a.T, b)

    # 使用广播计算平方距离矩阵
    # W 的每个元素 W[i, j] = ||a[:, i] - b[:, j]||^2
    W = np.expand_dims(aa, axis=1) + np.expand_dims(bb, axis=0) - 2 * ab

    # 确保平方距离矩阵中没有负值，数值误差可能导致极小负值
    W = np.abs(W)  # 保证矩阵中所有值为非负

    return W


@time_func
def DDRTree_reduce_dim_python(
    X_in,
    Z_in,
    Y_in,
    W_in,
    dimensions,
    maxIter,
    num_clusters,
    sigma,
    lambda_,
    gamma,
    eps,
    verbose,
):
    """
    DDRTree_reduce_dim_cpp 函数的 Python 实现版本。

    参数：
        X_in (np.ndarray): 输入数据矩阵。
        Z_in (np.ndarray): 另一个输入数据矩阵。
        Y_in (np.ndarray): 输入数据矩阵（通常是嵌入结果的初始化）。
        W_in (np.ndarray): 权重矩阵。
        dimensions (int): 目标维度。
        maxIter (int): 最大迭代次数。
        num_clusters (int): 聚类数量。
        sigma (float): Sigma 参数。
        lambda_ (float): Lambda 参数。
        gamma (float): Gamma 参数。
        eps (float): 收敛判定阈值。
        verbose (bool): 是否打印调试信息。

    返回：
        Y_out (np.ndarray): 降维后的数据矩阵。
        stree (csr_matrix): 稀疏树结构矩阵。
        Z_out (np.ndarray): 更新后的 Z 矩阵。
        W_out (np.ndarray): 更新后的 W 矩阵。
        Q (np.ndarray): 正交矩阵 Q。
        R (np.ndarray): 上三角矩阵 R。
        objective_vals (list): 每次迭代的目标函数值列表。
    """
    # 初始化输出变量
    Y_out = Y_in.copy()  # 初始化降维后的结果矩阵
    Z_out = Z_in.copy()  # 初始化 Z 矩阵
    W_out = W_in.copy()  # 初始化 W 矩阵

    objective_vals = []

    N_cells = X_in.shape[1]  # X_in 的列数
    if verbose:
        logger.info(f"初始化：细胞数量 N_cells = {N_cells}")

    # 使用 NetworkX 初始化无向图，替代 Boost Graph
    g = nx.Graph()
    g.add_nodes_from(range(Y_in.shape[1]))  # 添加所有节点

    # 构造完全图（每个节点对都有边），跳过对角线
    for j in range(Y_in.shape[1]):
        for i in range(j):  # 仅遍历上三角部分
            if i != j:
                g.add_edge(i, j)  # 添加边

    if verbose:
        logger.info("构造完全图完成")

    # 权重矩阵的初始化为零矩阵
    B = np.zeros((Y_in.shape[1], Y_in.shape[1]))  # Y_in 列数为图的节点数

    # 初始化用于存储最小生成树的结构
    old_spanning_tree = [None] * g.number_of_nodes()  # 占位符，长度等于节点数量

    # 初始化目标矩阵
    distsqMU = None  # 用于存储距离平方
    L = None  # 用于存储拉普拉斯矩阵

    # Z 和 Y 之间的距离矩阵
    distZY = np.zeros((X_in.shape[1], num_clusters))  # X_in 列数 x 聚类数量
    min_dist = np.zeros((X_in.shape[1], num_clusters))  # 最小距离矩阵
    tmp_distZY = np.zeros((X_in.shape[1], num_clusters))  # 临时距离矩阵

    # 初始化权重矩阵 R
    tmp_R = np.zeros((X_in.shape[1], num_clusters))  # X_in 列数 x 聚类数量
    R = np.zeros((tmp_R.shape[1], num_clusters))  # 稀疏矩阵 R 的占位符

    # 初始化 Gamma 矩阵
    Gamma = np.zeros((R.shape[1], R.shape[1]))  # R 的列数 x 列数

    # 稀疏矩阵形式
    tmp = csr_matrix(Gamma.shape)  # 稀疏矩阵
    tmp_dense = np.zeros((Gamma.shape[0], Gamma.shape[1]))  # 稠密矩阵

    # 初始化 Q 矩阵
    Q = np.zeros((tmp_dense.shape[0], R.shape[0]))  # Q 的形状取决于 Gamma 和 R 的行数

    # 初始化其他临时变量
    C = np.zeros((X_in.shape[0], Q.shape[1]))  # X_in 行数 x Q 列数
    tmp1 = np.zeros((C.shape[0], X_in.shape[0]))  # 临时矩阵，形状为 C 行数 x X_in 行数

    if verbose:
        logger.info("所有矩阵初始化完成")

    for iter in range(maxIter):
        if verbose:
            logger.info("**************************************")
            logger.info(f"Iteration: {iter}")

        distsqMU = sqdist_python(Y_out, Y_out)

        if verbose:
            logger.info(f"distsqMU: {distsqMU}")

        if verbose:
            logger.info("正在更新图中的边权重...")

        # 遍历图中所有边
        for edge in g.edges():
            u, v = edge  # 边的起点和终点
            if u != v:  # 检查是否为非自环边
                # 从 distsqMU 矩阵中获取权重值
                weight = distsqMU[u, v]
                # 更新边权重
                g[u][v]["weight"] = weight
                # 如果需要调试信息，打印更新的权重
                # if verbose:
                #     logger.info(f"边 ({u}, {v}) 的权重更新为: {weight}")

        if verbose:
            logger.info("计算最小生成树 (MST)")

        # 使用 networkx 提供的 Prim 算法计算 MST
        mst = nx.minimum_spanning_tree(g, algorithm="prim")  # 最小生成树
        spanning_tree = list(nx.to_dict_of_lists(mst).values())  # 转换为简单的MST结构

        if verbose:
            logger.info("Refreshing B matrix")

        # 更新邻接矩阵 B，先清除旧的边
        for ei in range(len(old_spanning_tree)):
            source = ei
            target = old_spanning_tree[ei]
            B[source, target] = 0
            B[target, source] = 0

        # 添加新的 MST 边
        for ei in range(len(spanning_tree)):
            source = ei
            target = spanning_tree[ei]
            if source != target:
                B[source, target] = 1
                B[target, source] = 1

        if verbose:
            logger.info(f"   B : ({B.shape[0]} x {B.shape[1]})")

        # 保存当前的MST结构
        old_spanning_tree = spanning_tree

        # 计算拉普拉斯矩阵 L
        L = np.diag(B.sum(axis=1))
        L = L - B

        if verbose:
            logger.info(f"   拉普拉斯矩阵 L: ({L.shape[0]} x {L.shape[1]})")

        # 检查 Z_out 和 Y_out 是否有 NaN
        if verbose:
            logger.info(
                f"   Z_out: ({Z_out.shape[0]} x {Z_out.shape[1]}), 最大值: {np.max(Z_out)}"
            )
            logger.info(
                f"   Y_out: ({Y_out.shape[0]} x {Y_out.shape[1]}), 最大值: {np.max(Y_out)}"
            )

        # 计算 Z 和 Y 之间的平方距离矩阵
        distZY = sqdist_python(Z_out, Y_out)  # 使用之前实现的 sq_dist_py 函数

        if verbose:
            logger.info(
                f"   distZY: ({distZY.shape[0]} x {distZY.shape[1]}), 最大值: {np.max(distZY)}"
            )

        if verbose:
            logger.info(f"   min_dist: ({min_dist.shape[0]} x {min_dist.shape[1]})")

        # 计算每一行的最小值
        distZY_minCoeff = np.min(distZY, axis=1)  # 每行的最小值

        if verbose:
            logger.info("distZY_minCoeff:")
            logger.info(distZY_minCoeff)

        # 将最小值复制到 min_dist 的每一列
        for i in range(min_dist.shape[1]):
            min_dist[:, i] = distZY_minCoeff

        if verbose:
            logger.info("min_dist:")
            logger.info(min_dist)

        # 计算 tmp_distZY = distZY - min_dist
        tmp_distZY = distZY - min_dist

        if verbose:
            logger.info("tmp_distZY:")
            logger.info(tmp_distZY)

        # 计算 tmp_R = exp(-tmp_distZY / sigma)
        tmp_R = np.exp(-tmp_distZY / sigma)

        if verbose:
            logger.info(f"tmp_R: ({tmp_R.shape[0]} x {tmp_R.shape[1]})")
            logger.info(tmp_R)

        # 数值检查：确保 tmp_R 中没有 NaN 或 Inf
        if not np.all(np.isfinite(tmp_R)):
            raise ValueError("tmp_R contains NaN or Inf!")

        # 计算 R = tmp_R / row-wise sum of tmp_R
        row_sums = np.sum(tmp_R, axis=1, keepdims=True)  # 按行求和并保持二维
        R = tmp_R / row_sums

        if verbose:
            logger.info(f"R: ({R.shape[0]} x {R.shape[1]})")
            logger.info(R)

        # 计算 tmp_R 的行和并扩展到 R 的每列
        tmp_R_rowsums = np.sum(tmp_R, axis=1, keepdims=True)
        R = tmp_R / tmp_R_rowsums

        if verbose:
            logger.info(f"R: ({R.shape[0]} x {R.shape[1]})")
            logger.info(R)

        # 更新 Gamma 矩阵，对角元素是 R 的列和
        Gamma = np.zeros((R.shape[1], R.shape[1]))
        np.fill_diagonal(Gamma, np.sum(R, axis=0))

        if verbose:
            logger.info(f"Gamma: ({Gamma.shape[0]} x {Gamma.shape[1]})")
            logger.info(Gamma)

        # 计算目标函数的第一部分 obj1
        x1 = np.log(np.sum(np.exp(-distZY / sigma), axis=1))
        obj1 = -sigma * np.sum(x1 - min_dist[:, 0] / sigma)

        if verbose:
            logger.info(f"obj1: {obj1}")

        if verbose:
            logger.info(f"   X : ( { X_in.shape[0] } x { X_in.shape[1] } )")
            logger.info(f"   W : ( { W_out.shape[0] } x { W_out.shape[1] } )")
            logger.info(f"   Z : ( { Z_out.shape[0] } x { Z_out.shape[1] } )")

        # 计算目标函数的第二部分 obj2
        norm_diff = np.linalg.norm(X_in - np.dot(W_out, Z_out)) ** 2
        obj2 = (
            norm_diff
            + lambda_ * np.sum(np.diagonal(np.dot(Y_out, np.dot(L, Y_out.T))))
            + gamma * obj1
        )

        if verbose:
            logger.info(f"obj2: {obj2}")

        if verbose:
            logger.info(f"   L : ( {L.shape[0]} x {L.shape[1]} )")

        # 更新目标函数值列表
        objective_vals.append(obj2)

        if verbose:
            logger.info("Checking termination criterion")

        # 检查收敛条件
        if iter >= 1:
            delta_obj = abs(objective_vals[iter] - objective_vals[iter - 1])
            delta_obj /= abs(objective_vals[iter - 1])

            if verbose:
                logger.info(f"delta_obj: {delta_obj}")
                
            if delta_obj < eps:
                if verbose:
                    logger.warning("Termination criterion met. Stopping iteration.")
                break

        if verbose:
            logger.info("Computing tmp")

        if verbose:
            logger.info("... stage 1")
        # Step 1: 构造稀疏矩阵 tmp
        tmp = (Gamma + (L * (lambda_ / gamma))) * ((gamma + 1.0) / gamma)

        # 转换为稀疏矩阵格式
        tmp = csr_matrix(tmp)

        if verbose:
            logger.info(f"稀疏矩阵 tmp: {tmp.shape}, 非零元素数: {tmp.nnz}")

        # Step 2: 构造 R 的稀疏矩阵形式
        R_sp = csr_matrix(R)
        R_sp = csc_matrix(R_sp)

        if verbose:
            logger.info("tmp: tmp = tmp - (R.T @ R)")
        # 更新 tmp: tmp = tmp - (R.T @ R)
        # t = R_sp.T @ R_sp
        tmp = tmp - R_sp.T @ R_sp

        if verbose:
            logger.info(
                f"更新后的稀疏矩阵 tmp: {tmp.shape}, 非零元素数: {tmp.nnz}"
            )

        # Step 3: 稀疏矩阵分解和线性方程求解
        try:
            # 使用稀疏矩阵求解器 splu 代替 SimplicialLLT
            lu_solver = splu(tmp)
            tmp_dense = lu_solver.solve(R.T).T
        except Exception as e:
            if verbose:
                logger.info("稀疏矩阵分解失败，切换到密集求解。")
                logger.info(f"错误信息: {e}")
            # 如果稀疏分解失败，回退到密集矩阵计算
            tmp_dense = np.linalg.solve(tmp.toarray(), R.T).T

        if verbose:
            logger.info(f"tmp_dense: ( {tmp_dense.shape[0]} x {tmp_dense.shape[1]} )")

        if verbose:
            logger.info(f"Computing Q: ( {Q.shape[0]} x {Q.shape[1]} )")

        # 计算 Q 矩阵
        Q = (X_in + np.dot(np.dot(X_in, tmp_dense), R.T)) / (gamma + 1.0)

        if verbose:
            logger.info(f"gamma: {gamma}")
            logger.info(f"X_in: {X_in.shape}")
            logger.info(f"Q: {Q.shape}")

        # 计算 C 矩阵
        C = Q

        # 计算临时矩阵 tmp1
        tmp1 = np.dot(Q, X_in.T)

        if verbose:
            logger.info("Computing W")

        # PCA 投影：计算 W 矩阵
        W = pca_projection_python((tmp1 + tmp1.T) / 2, dimensions)

        # 更新 W_out
        W_out = W

        if verbose:
            logger.info("Computing Z")

        # 计算 Z 矩阵
        Z_out = np.dot(W_out.T, C)

        if verbose:
            logger.info("Computing Y")

        # 计算 Y 矩阵
        # Y_out = t(solve((lambda / gamma * L + Gamma), t(Z %*% R)))
        Y_out = (lambda_ / gamma) * L + Gamma
        Y_out = spsolve(Y_out, np.dot(Z_out, R).T).T  # 稀疏矩阵解法

    if verbose:
        logger.info("Clearing MST sparse matrix")

    # 初始化稀疏矩阵（这里使用 LIL 格式，便于构造）
    stree = lil_matrix((N_cells, N_cells))

    if verbose:
        logger.info(f"Setting up MST sparse matrix with {len(old_spanning_tree)} edges")

    # 遍历最小生成树的边并填充三元组列表
    for ei in range(len(old_spanning_tree)):
        # 获取节点索引
        source = ei
        target = old_spanning_tree[ei]

        # 确保节点索引不同，避免自环
        if source != target:
            # 计算边的权重
            weight_1 = distsqMU[source, target]
            weight_2 = distsqMU[target, source]  # 对称边的权重

            # 添加无向图的双向边
            stree[source, target] = weight_1
            stree[target, source] = weight_2


    # 转换为 CSR 格式以便高效运算
    stree = stree.tocsr()

    if verbose:
        logger.info(f"MST sparse matrix constructed with shape {stree.shape}")

    return {
        "W": W,
        "Z": Z_out,
        "stree": stree,
        "Y": Y_out,
        "X": X_in,
        "R": R,
        "Q": Q,
        "objective_vals": objective_vals,
        "history": None,
    }


def DDRTree_python(
    X,
    dimensions=2,
    initial_method=None,
    maxIter=20,
    sigma=1e-3,
    lambda_param=None,
    ncenter=None,
    param_gamma=10,
    tol=1e-3,
    verbose=False,
    **kwargs,
):
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
            raise ValueError("降维方法返回的维度不正确")
        Z = tmp[:, :dimensions].T  # 转置成维度正确的格式

    if ncenter is None:
        K = N
        Y = Z[:, :K]
    else:
        K = ncenter
        if K > Z.shape[1]:
            raise ValueError("错误: ncenter 必须大于等于 ncol(X)")
        centers = Z[:, np.linspace(0, Z.shape[1] - 1, K, dtype=int)].T
        kmeans = KMeans(n_clusters=K, init=centers)
        kmeans.fit(Z.T)

        Y = kmeans.cluster_centers_.T  # 转置成 D x K

    # 默认 lambda_param
    if lambda_param is None:
        lambda_param = 5 * N

    # 修改为调用 C++ 绑定的函数
    ddrtree_res = DDRTree_reduce_dim_python(
        X,
        Z,
        Y,
        W,
        dimensions,
        maxIter,
        K,
        sigma,
        lambda_param,
        param_gamma,
        tol,
        verbose,
    )

    return {
        "W": ddrtree_res["W"],
        "Z": ddrtree_res["Z"],
        "stree": ddrtree_res["stree"],
        "Y": ddrtree_res["Y"],
        "X": ddrtree_res["X"],
        "R": ddrtree_res["R"],
        "Q": ddrtree_res["Q"],
        "objective_vals": ddrtree_res["objective_vals"],
        "history": None,
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
