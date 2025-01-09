#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

// Import required libraries
use pyo3::prelude::*; // Import pre-defined items from the PyO3 module
use pyo3::types::PyDict; // Import PyDict type, used for creating Python dictionaries
use sprs::CsMat; // Import the definition for sparse matrix CsMat
use pyo3::Py; // Import Py type
use pyo3::PyAny; // Import PyAny type, representing any Python object
use pyo3::PyResult; // Import PyResult type, used to represent potential errors
use pyo3::Python; // Import Python type, used to obtain the Python GIL (Global Interpreter Lock)

/// Matrix multiplication function that takes two sparse matrices as input and returns their product
///
/// # Parameters
///
/// * `a`: The first sparse matrix
/// * `b`: The second sparse matrix
///
/// # Returns
///
/// Returns the resulting sparse matrix
fn csr_matrix_multiply(a: &CsMat<f64>, b: &CsMat<f64>) -> CsMat<f64> {
    assert_eq!(a.cols(), b.rows(), "Incompatible matrix dimensions for multiplication.");

    let mut triplet = Vec::new(); // 用于存储 (row, col, value)

    for i in 0..a.rows() {
        if let Some(a_row) = a.outer_view(i) {
            for j in 0..b.cols() {
                if let Some(b_col) = b.outer_view(j) {
                    let value = a_row.dot(&b_col);
                    if value != 0.0 {
                        triplet.push((i, j, value));
                    }
                }
            }
        }
    }

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut data = Vec::new();

    for (r, c, v) in triplet {
        rows.push(r);
        cols.push(c);
        data.push(v);
    }

    CsMat::new_csc((a.rows(), b.cols()), rows, cols, data)
}

/// Python interface function for multiplication
///
/// # Parameters
///
/// * `matrix_a`: The first sparse matrix
/// * `matrix_b`: The second sparse matrix
///
/// # Returns
///
/// Returns a Python dictionary containing the multiplication result
#[pyfunction]
fn multiply_csr_matrices(
    matrix_a: Py<PyAny>, // The first matrix, using PyAny type to accept any Python object
    matrix_b: Py<PyAny>, // The second matrix
) -> PyResult<Py<PyAny>> {
    // Obtain the Python GIL
    Python::with_gil(|py| {
        // Extract relevant data from the first matrix
        let data_a: Vec<f64> = matrix_a.getattr(py, "data")?.extract(py)?;
        let indices_a: Vec<usize> = matrix_a.getattr(py, "indices")?.extract(py)?;
        let indptr_a: Vec<usize> = matrix_a.getattr(py, "indptr")?.extract(py)?;
        let shape_a: (usize, usize) = matrix_a.getattr(py, "shape")?.extract(py)?;

        // Check if the length of indptr_a matches the expected value
        assert_eq!(indptr_a.len(), shape_a.0 + 1, "Indptr length does not match dimension for matrix A.");

        // Print debug information
        println!("Matrix A: Data length: {}, Indices length: {}, Indptr length: {}", 
                  data_a.len(), indices_a.len(), indptr_a.len());
        println!("Shape A: {:?}", shape_a);

        // Extract relevant data from the second matrix
        let data_b: Vec<f64> = matrix_b.getattr(py, "data")?.extract(py)?;
        let indices_b: Vec<usize> = matrix_b.getattr(py, "indices")?.extract(py)?;
        let indptr_b: Vec<usize> = matrix_b.getattr(py, "indptr")?.extract(py)?;
        let shape_b: (usize, usize) = matrix_b.getattr(py, "shape")?.extract(py)?;

        // Construct sparse matrices from the extracted data
        let csr_a = CsMat::new_csc((shape_a.0, shape_a.1), indptr_a, indices_a, data_a);
        let csr_b = CsMat::new_csc((shape_b.0, shape_b.1), indptr_b, indices_b, data_b);
        
        // Call the matrix multiplication function to get the result
        let csr_c = csr_matrix_multiply(&csr_a, &csr_b);

        // Extract data from the multiplication result
        let (data_c, indices_c, indptr_c) = (csr_c.data().to_vec(), csr_c.indices().to_vec(), csr_c.indptr().to_vec());
        let shape_c = (csr_c.rows(), csr_c.cols());

        // Create a Python dictionary to store the result
        let result = PyDict::new(py);
        result.set_item("data", data_c)?;
        result.set_item("indices", indices_c)?;
        result.set_item("indptr", indptr_c)?;
        result.set_item("shape", shape_c)?;

        // Return the Python dictionary
        Ok(result.into())
    })
}

// Define the Python module
#[pymodule]
fn sparse_module(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add the multiplication function to the module
    m.add_function(wrap_pyfunction!(multiply_csr_matrices, m)?)?;
    Ok(())
}
