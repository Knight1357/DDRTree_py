import numpy as np
from DDRTree import sqdist, pca_projection_python

def expose_sqdist(a, b):
    """
    封装 sqdist 函数，便于 C++ 调用。
    参数:
        a: numpy.ndarray, D x N 矩阵。
        b: numpy.ndarray, D x N 矩阵。
    返回:
        矩阵: a 和 b 之间的平方距离。
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("Input matrices must be numpy arrays.")
    return sqdist(a, b)

def expose_pca_projection(C, L):
    """
    封装 pca_projection_python 函数，便于 C++ 调用。
    参数:
        C: numpy.ndarray, 数据矩阵。
        L: int, 要计算的主成分数量。
    返回:
        矩阵: PCA 降维后的主成分矩阵。
    """
    if not isinstance(C, np.ndarray):
        raise ValueError("Input matrix C must be a numpy array.")
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be a positive integer.")
    return pca_projection_python(C, L)