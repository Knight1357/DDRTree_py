import numpy as np
from matrix_mult import multiply_dense_matrices

def test_dense_matrix_multiply():
    # 创建两个密集矩阵
    matrix_a = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    matrix_b = np.array([[1, 0], [0, 1], [1, 1]])

    # 调用 Rust 的乘法函数
    result = multiply_dense_matrices(matrix_a, matrix_b)

    # 打印结果
    print("Result (data):", result['data'])
    print("Result (indices):", result['indices'])
    print("Result (indptr):", result['indptr'])
    print("Result (shape):", result['shape'])

if __name__ == "__main__":
    test_dense_matrix_multiply()
