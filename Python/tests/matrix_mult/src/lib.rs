#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use sprs::CsMat;

/// 将 Python 密集矩阵转换为 Rust 的稀疏矩阵
fn dense_to_csr(dense: &PyAny) -> PyResult<CsMat<f64>> {
    Python::with_gil(|py| {
        // 从 Python 对象提取 shape
        let shape: &PyAny = dense.getattr(py, PyString::new(py, "shape"))?;
        let rows: usize = shape.get_item(0)?.extract()?;
        let cols: usize = shape.get_item(1)?.extract()?;

        let mut data = vec![];
        let mut row_indices = vec![];
        let mut col_indices = vec![];

        // 遍历密集矩阵以填充数据
        for i in 0..rows {
            let row: Vec<f64> = dense.get_item(i)?.extract()?;
            for j in 0..cols {
                if row[j] != 0.0 {
                    data.push(row[j]);
                    row_indices.push(i);
                    col_indices.push(j);
                }
            }
        }

        Ok(CsMat::new_csc((rows, cols), row_indices, col_indices, data))
    })
}

/// 矩阵乘法函数，接受两个密集矩阵作为参数，返回它们的稀疏矩阵乘积
#[pyfunction]
fn multiply_dense_matrices(matrix_a: &PyAny, matrix_b: &PyAny) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        // 将密集矩阵转换为稀疏矩阵
        let csr_a = dense_to_csr(matrix_a)?;
        let csr_b = dense_to_csr(matrix_b)?;

        // 确保矩阵相乘的合法性
        if csr_a.cols() != csr_b.rows() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Incompatible shapes for matrix multiplication.",
            ));
        }

        // 执行稀疏矩阵乘法
        let csr_c = csr_a * csr_b;

        // 提取乘法结果中的数据
        let (data_c, indices_c, indptr_c) = (
            csr_c.data().to_vec(),
            csr_c.indices().to_vec(),
            csr_c.indptr().to_vec(),
        );
        let shape_c = (csr_c.rows(), csr_c.cols());

        // 创建一个 Python 字典来存储结果
        let result = PyDict::new(py);
        result.set_item("data", data_c)?;
        result.set_item("indices", indices_c)?;
        result.set_item("indptr", indptr_c)?;
        result.set_item("shape", shape_c)?;

        // 返回 Python 字典
        Ok(result.into())
    })
}

/// 定义 Python 模块
#[pymodule]
fn sparse_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(multiply_dense_matrices, m)?)?;
    Ok(())
}
