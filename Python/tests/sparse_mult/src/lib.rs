use pyo3::prelude::*;
use sprs::{CsMat, CsMatView};

/// 稀疏矩阵乘法：计算 A^T * A
#[pyfunction]
fn sparse_matrix_multiplication(
    indices: Vec<usize>,
    indptr: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
) -> (Vec<usize>, Vec<usize>, Vec<f64>, (usize, usize)) {
    // 构造稀疏矩阵 A
    let a = CsMat::new((shape.0, shape.1), indptr, indices, data);

    // 计算 A^T * A
    let ata = a.transpose_view().unwrap() * &a;

    // 将结果转换回 CSR 格式
    let (indptr, indices, data) = ata.into_raw_storage();
    let shape = ata.shape();

    (indptr, indices, data, shape)
}

/// 创建 Python 模块
#[pymodule]
fn sparse_mult(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sparse_matrix_multiplication, m)?)?;
    Ok(())
}
