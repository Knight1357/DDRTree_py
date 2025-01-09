import numpy as np
from scipy.sparse import csr_matrix
from sparse_mult import sparse_matrix_multiplication

# 构造稀疏矩阵 R
R = csr_matrix(np.array([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]
], dtype=np.float64))

# 获取稀疏矩阵的 CSR 格式
indices = R.indices.tolist()
indptr = R.indptr.tolist()
data = R.data.tolist()
shape = R.shape

# 调用 Rust 模块进行稀疏矩阵乘法
indptr_result, indices_result, data_result, shape_result = sparse_matrix_multiplication(
    indices, indptr, data, shape
)

# 将结果转换为 SciPy 稀疏矩阵
result_matrix = csr_matrix((data_result, indices_result, indptr_result), shape=shape_result)

print("Resultant Sparse Matrix (CSR):")
print(result_matrix)
