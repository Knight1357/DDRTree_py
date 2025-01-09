import numpy as np
from scipy.sparse import random, csr_matrix
import numba_sparce_mult


def test_csr_matrix_multiply():
    """
    Test the csr_matrix_multiply function with random sparse matrices.
    """
    # Generate two random sparse matrices
    np.random.seed(42)  # For reproducibility
    rows_a, cols_a = 100, 200  # Shape of matrix A
    rows_b, cols_b = 200, 150  # Shape of matrix B

    density_a = 0.05  # Sparsity level of matrix A
    density_b = 0.05  # Sparsity level of matrix B

    # Generate random sparse matrices in CSR format
    matrix_a = random(rows_a, cols_a, density=density_a, format='csr', dtype=np.float32, random_state=42)
    matrix_b = random(rows_b, cols_b, density=density_b, format='csr', dtype=np.float32, random_state=42)

    # Perform CSR matrix multiplication
    result = numba_sparce_mult.csr_matrix_multiply(matrix_a, matrix_b, threads=4)

    # Print results
    print("Matrix A shape:", matrix_a.shape)
    print("Matrix B shape:", matrix_b.shape)
    print("Result shape:", result.shape)
    print("Result (dense format):\n", result.toarray())

    return result

# Example usage
if __name__ == "__main__":
    result = test_csr_matrix_multiply()
