import numpy as np
from scipy.sparse import csr_matrix, random
from sparse_module import multiply_csr_matrices

def generate_random_sparse_matrix(rows, cols, density=0.01, format='csr'):
    """Generates a random sparse matrix with specified dimensions and density."""
    return random(rows, cols, density=density, format=format, dtype=np.float64)

def test_csr_matrix_multiply_with_random_matrices():
    np.random.seed(42)  # Set a seed for reproducibility

    # Define dimensions for matrices A and B
    rows_a = np.random.randint(5, 20)  # Random number of rows for matrix A
    cols_a_rows_b = np.random.randint(5, 20)  # Shared dimension for A (columns) and B (rows)
    cols_b = np.random.randint(5, 20)  # Random number of columns for matrix B

    # Generate random sparse matrices A and B
    matrix_a = generate_random_sparse_matrix(rows_a, cols_a_rows_b)
    matrix_b = generate_random_sparse_matrix(cols_a_rows_b, cols_b)

    # Print matrix shapes
    print("Matrix A shape:", matrix_a.shape)
    print("Matrix B shape:", matrix_b.shape)

    # Call Rust function for CSR matrix multiplication
    rust_result = multiply_csr_matrices(matrix_a, matrix_b)

    # Compute the expected result using SciPy's dot method
    python_result = matrix_a.dot(matrix_b)

    # Convert Rust's result back to a CSR matrix for comparison
    rust_csr_result = csr_matrix(
        (rust_result['data'], rust_result['indices'], rust_result['indptr']),
        shape=rust_result['shape']
    )

    # Compare the results
    if np.allclose(rust_csr_result.toarray(), python_result.toarray(), atol=1e-8):
        print("Test passed: The multiplication results are equal.")
    else:
        print("Test failed: The multiplication results differ.")
        # Provide additional details
        diff = rust_csr_result - python_result
        print("Difference matrix (non-zero elements):", diff.nnz)
        print(diff)

if __name__ == "__main__":
    # Run the test case
    test_csr_matrix_multiply_with_random_matrices()
