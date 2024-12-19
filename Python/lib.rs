use pyo3::prelude::*;
use numpy::{PyReadonlyArrayDyn,IntoPyArray,PyReadonlyArray2,PyArray2};
use rayon::prelude::*;
use ndarray::prelude::*;
// use hdf5::File;
use sprs::CsMat;
// 进度条
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::ParallelIterator;
// use anyhow::{anyhow,Result};
// use hdf5::types::VarLenUnicode;

// 计算皮尔逊相关性
fn pearson_correlation(x: &ArrayView1<f32>, y: &ArrayView1<f32>, mean_x: f32, mean_y: f32, std_x: f32, std_y: f32, n: f32) -> f32 {
    let denominator = std_x * std_y * n;
    if denominator == 0.0 {
        f32::NAN
    } else {
        let numerator: f32 = x.iter().zip(y.iter())
        .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
        .sum();
        numerator / denominator
    }
}


// fn h5ad2csr(filename: &str) -> Result<CsMat<f32>> {
//     let file = File::open(filename)?;
//     let group = file.group("X")?;
//     let shape: Vec<usize> = group.attr("shape")?.read_1d()?.to_vec();
//     // println!("shape: {:?}",&shape);

//     // 尝试读取编码类型属性，如果不存在则读取 h5sparse_format 属性
//     let encoding_type = if let Ok(encoding) = group.attr("encoding-type") {
//         encoding.read_scalar::<VarLenUnicode>()?.as_str().to_string()
//     } else if let Ok(h5sparse_format) = group.attr("h5sparse_format") {
//         h5sparse_format.read_scalar::<VarLenUnicode>()?.as_str().to_string()
//     } else {
//         return Err(anyhow!("Neither 'encoding-type' nor 'h5sparse_format' found"));
//     };
//     // println!("编码类型: {:?}", &encoding_type);

//     let data: Vec<f32> = file.dataset("X/data")?.read_1d()?.to_vec();
//     let indices: Vec<usize> = file.dataset("X/indices")?.read_1d()?.to_vec();
//     let indptr: Vec<usize> = file.dataset("X/indptr")?.read_1d()?.to_vec();
//     let mtx: CsMat<f32>;
//     // 根据编码类型处理CSR和CSC
//     if encoding_type.starts_with("csr") {
//         // println!("编码类型以'csr'开始");
//         mtx = CsMat::new((shape[0], shape[1]), indptr, indices, data);
//     } else if encoding_type.starts_with("csc") {
//         // println!("编码类型以'csc'开始");
//         mtx = CsMat::new_csc((shape[0], shape[1]), indptr, indices, data).to_other_storage();
//     } else {
//         return Err(anyhow!("Unsupported encoding type: {:?}",encoding_type));
//     }
//     Ok(mtx)
// }

/// 对给定的一行数据进行排名，使用 'first' 方法处理相同值
/// 使用 u32 减少内存
fn rank_first(row: ArrayView1<f32>) -> Array1<u32> {
    let mut ranks = vec![0u32; row.len()];
    let mut values_indices: Vec<(f32, usize)> = row.iter().cloned().zip(0..row.len()).collect();
    values_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (rank, &(_, index)) in values_indices.iter().enumerate() {
        ranks[index] = (rank + 1) as u32;
    }
    Array1::from(ranks)
}

/// 对一个二维数组的每一行进行排名
fn rank_2d_array(array: ArrayView2<f32>) -> Array2<u32> {
    let shape = array.raw_dim();
    let mut ranked_array = Array2::<u32>::zeros(shape);

    ranked_array.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(row_idx, mut row)| {
        let rank_row = rank_first(array.row(row_idx));
        row.assign(&rank_row.view());
    });

    ranked_array
}


// 稀疏矩阵行迭代进行 rank
// fn rank_csr_matrix(mtx: CsMat<f32>) -> Array2<u32> {
//     let (rows, cols) = mtx.shape();
//     let mut ranked_array = Array2::<u32>::zeros((rows, cols).f());

//     ranked_array.axis_iter_mut(Axis(0)).into_par_iter().enumerate().for_each(|(row_idx, mut row)| {
//         let mtx_row = mtx.outer_view(row_idx).unwrap().to_dense();
//         let rank_row = rank_first(mtx_row.view());
//         row.assign(&rank_row.view());
//     });

//     ranked_array
// }

#[pyfunction]
fn rank_matrix(py: Python, matrix: PyReadonlyArray2<f32>) -> Py<PyArray2<u32>> {
    let matrix = matrix.as_array();
    let ranked_matrix = rank_2d_array(matrix);
    ranked_matrix.into_pyarray_bound(py).into()
}

// #[pyfunction]
// fn rank_matrix_from_file(py: Python, filename: &str) -> Py<PyArray2<u32>> {
//     // 从 h5ad 文件中读取数据，减少内存
//     let mtx = h5ad2csr(filename).expect("Rust `h5ad2csr` 函数出错");
//     let ranked_matrix = rank_csr_matrix(mtx);
//     ranked_matrix.into_pyarray_bound(py).into()
// }

// 计算皮尔逊相关性
// 不提前计算均值，标准差
// fn pearson_correlation2(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
//     let mean_x = x.mean().unwrap();
//     let mean_y = y.mean().unwrap();
//     let std_x = x.std(0.);
//     let std_y = y.std(0.);
//     let denominator = std_x * std_y;

//     if denominator == 0.0 {
//         f64::NAN
//     } else {
//         // Calculate the terms needed for the Pearson coefficient
//         let diff_x = x - mean_x;
//         let diff_y = y - mean_y;
//         let numerator = diff_x.dot(&diff_y);

//         numerator / denominator / (x.len() as f64)
//     }
// }
fn pearson_correlation2(x: sprs::CsVecView<f64>, y: sprs::CsVecView<f64>) -> f64 {
    fn sum_x_x2(x: sprs::CsVecView<f64>) -> (f64, f64) {
        x.data()
            .iter()
            .fold((0.0, 0.0), |(x0, x1), &x| (x0 + x, x1 + x.powi(2)))
    }

    assert_eq!(x.dim(), y.dim());

    let (sum_x, sum_x2) = sum_x_x2(x);
    let (sum_y, sum_y2) = sum_x_x2(y);

    let sum_xy = x.dot(y);

    let n = x.dim() as f64;
    let numerator = n * sum_xy - sum_x * sum_y;

    let denominator_x = (n * sum_x2 - sum_x.powi(2)).sqrt();
    let denominator_y = (n * sum_y2 - sum_y.powi(2)).sqrt();

    let denominator = denominator_x * denominator_y;
    if denominator == 0.0 {
        f64::NAN
    } else {
        numerator / denominator
    }
}

// 使用稀疏矩阵，减少内存占用
#[pyfunction]
fn corr_by_index_rs2(py: Python<'_>, mtx: &Bound<'_, PyAny>, col_idx_pairs: PyReadonlyArrayDyn<'_, usize>) -> PyResult<PyObject> {
    let col_idx_pairs = col_idx_pairs.as_array();
    // csc 转为 sprs
    let data: Vec<f64> = mtx.getattr("data")?.extract()?;
    let indices: Vec<usize> = mtx.getattr("indices")?.extract()?;
    let indptr: Vec<usize> = mtx.getattr("indptr")?.extract()?;
    let shape: (usize, usize) = mtx.getattr("shape")?.extract()?;

    // 创建sprs矩阵
    let mtx = CsMat::new_csc((shape.0, shape.1), indptr, indices, data);
    println!("转化到 Rust 完成，开始计算相关性。");

    let style = ProgressStyle::default_bar()
        .template("{wide_bar} {pos}/{len}    {per_sec} iter/s {msg}").unwrap();
    // 使用 Rayon 的 par_iter 来处理索引对并行计算相关性
    let corr_values: Vec<f64> = col_idx_pairs.axis_iter(ndarray::Axis(0))
        .into_par_iter()
        .progress_with_style(style)
        .map(|row| {
            let idx1 = row[0];
            let idx2 = row[1];
            let col1 = mtx.outer_view(idx1).unwrap();
            let col2 = mtx.outer_view(idx2).unwrap();

            pearson_correlation2(col1, col2)
        })
        .collect();

    // You would then iterate through the index pairs and compute correlations, storing them in a vector
    // For now, we just return a placeholder string
    Ok(corr_values.into_pyarray_bound(py).into())
}


#[pyfunction]
fn corr_by_index_rs(py: Python<'_>, mtx: PyReadonlyArrayDyn<'_, f32>, col_idx_pairs: PyReadonlyArrayDyn<'_, usize>, mean: PyReadonlyArrayDyn<'_, f32>, stdv: PyReadonlyArrayDyn<'_, f32>) -> PyResult<PyObject> {
    let mtx = mtx.as_array();
    let col_idx_pairs = col_idx_pairs.as_array();
    let mean = mean.as_array();
    let stdv = stdv.as_array();

    // 使用 Rayon 的 par_iter 来处理索引对并行计算相关性
    let corr_values: Vec<f32> = col_idx_pairs.axis_iter(ndarray::Axis(0))
        .into_par_iter()
        .map(|row| {
            let idx1 = row[0];
            let idx2 = row[1];
            let col1 = mtx.slice(s![.., idx1]);
            let col2 = mtx.slice(s![.., idx2]);
            let n = col1.len() as f32;

            pearson_correlation(&col1, &col2,mean[idx1], mean[idx2], stdv[idx1], stdv[idx2], n)
        })
        .collect();

    // You would then iterate through the index pairs and compute correlations, storing them in a vector
    // For now, we just return a placeholder string
    Ok(corr_values.into_pyarray_bound(py).into())
}

/// A Python module implemented in Rust.
#[pymodule]
fn scenic_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(corr_by_index_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rank_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(corr_by_index_rs2, m)?)?;
    // m.add_function(wrap_pyfunction!(rank_matrix_from_file, m)?)?;
    Ok(())
}