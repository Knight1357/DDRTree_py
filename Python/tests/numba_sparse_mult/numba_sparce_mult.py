import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange

@njit(parallel=True)
def parallel_csr_multiply(data_a, indices_a, indptr_a, shape_a,
                          data_b, indices_b, indptr_b, shape_b):
    """
    Perform CSR matrix multiplication using Numba and parallelization.

    Parameters:
        data_a, indices_a, indptr_a: CSR components of matrix A
        data_b, indices_b, indptr_b: CSR components of matrix B
        shape_a: Shape of matrix A
        shape_b: Shape of matrix B

    Returns:
        (data_result, indices_result, indptr_result): CSR components of the result matrix
    """
    if shape_a[1] != shape_b[0]:
        raise ValueError("Matrix dimensions do not match for multiplication.")
    
    n_rows = shape_a[0]
    indptr_result = np.zeros(n_rows + 1, dtype=np.int32)
    local_data_list = []
    local_indices_list = []

    for i in prange(n_rows):
        row_values = {}
        for a_index in range(indptr_a[i], indptr_a[i + 1]):
            col_a = indices_a[a_index]
            val_a = data_a[a_index]

            for b_index in range(indptr_b[col_a], indptr_b[col_a + 1]):
                col_b = indices_b[b_index]
                val_b = data_b[b_index]

                if col_b not in row_values:
                    row_values[col_b] = 0.0
                row_values[col_b] += val_a * val_b

        for col, val in row_values.items():
            local_indices_list.append(col)
            local_data_list.append(val)
        indptr_result[i + 1] = len(local_data_list)

    return (
        np.array(local_data_list, dtype=np.float32),
        np.array(local_indices_list, dtype=np.int32),
        indptr_result
    )

def csr_matrix_multiply(csr_a, csr_b, threads):
    """
    Multiply two CSR matrices using Numba and parallelization.

    Parameters:
        csr_a: scipy.sparse.csr_matrix
        csr_b: scipy.sparse.csr_matrix
        threads: Number of parallel threads to use

    Returns:
        scipy.sparse.csr_matrix: Result of the multiplication
    """
    if not isinstance(csr_a, csr_matrix) or not isinstance(csr_b, csr_matrix):
        raise ValueError("Both inputs must be CSR matrices.")
    
    # Extract CSR components
    data_a, indices_a, indptr_a = csr_a.data, csr_a.indices, csr_a.indptr
    data_b, indices_b, indptr_b = csr_b.data, csr_b.indices, csr_b.indptr
    
    # Set the number of threads for Numba
    import os
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    
    # Multiply matrices
    data_result, indices_result, indptr_result = parallel_csr_multiply(
        data_a, indices_a, indptr_a, csr_a.shape,
        data_b, indices_b, indptr_b, csr_b.shape
    )
    
    # Create and return the result CSR matrix
    return csr_matrix((data_result, indices_result, indptr_result), shape=(csr_a.shape[0], csr_b.shape[1]))

# Example usage
if __name__ == "__main__":
    # Define two sparse matrices in CSR format
    a = csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    b = csr_matrix([[0, 1, 0], [4, 0, 0], [0, 0, 5]])
    
    # Multiply the matrices with 4 parallel threads
    result = csr_matrix_multiply(a, b, threads=4)
    
    # Print the result as a dense array
    print(result.toarray())
