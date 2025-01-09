import numpy as np
from numba import njit, prange

@njit(parallel=True)
def dense_matrix_multiply(A, B, threads):
    """
    Perform dense matrix multiplication using Numba and parallelization.

    Parameters:
        A (numpy.ndarray): The first dense matrix with shape (m, n).
        B (numpy.ndarray): The second dense matrix with shape (n, p).
        threads (int): Number of threads to use for parallel computation.

    Returns:
        numpy.ndarray: The resulting dense matrix with shape (m, p).
    """
    # Ensure matrix dimensions are compatible
    m, n = A.shape
    n_b, p = B.shape
    if n != n_b:
        raise ValueError("Matrix dimensions do not match for multiplication.")

    # Initialize the result matrix
    C = np.zeros((m, p), dtype=A.dtype)

    # Perform parallelized matrix multiplication
    for i in prange(m):  # Parallelize over rows of A
        for j in range(p):  # Iterate over columns of B
            temp = 0.0
            for k in range(n):  # Dot product of row A[i] and column B[:, j]
                temp += A[i, k] * B[k, j]
            C[i, j] = temp

    return C

# Example usage
if __name__ == "__main__":
    # Example matrices
    np.random.seed(42)
    A = np.random.rand(500, 300).astype(np.float32)
    B = np.random.rand(300, 400).astype(np.float32)

    # Number of threads for parallel computation
    threads = 4

    # Perform matrix multiplication
    result = dense_matrix_multiply(A, B, threads)

    # Print the result shape
    print("Result shape:", result.shape)
