// 导入需要的库以支持Python交互、数值计算和并行处理
use pyo3::prelude::*; // 用于创建Python模块
use numpy::{PyReadonlyArrayDyn, IntoPyArray, PyReadonlyArray2, PyArray2}; // 用于处理Python中的NumPy数组
use rayon::prelude::*; // 用于并行迭代
use ndarray::prelude::*; // 用于多维数组运算
use sprs::CsMat; // 支持稀疏矩阵的操作
use indicatif::{ParallelProgressIterator, ProgressStyle}; // 用于展示进度条
use rayon::iter::ParallelIterator; // 用于并行迭代

// 计算皮尔逊相关系数的函数，接受两个一维数组和它们的均值、标准差以及长度
fn pearson_correlation(
    x: &ArrayView1<f32>,
    y: &ArrayView1<f32>,
    mean_x: f32,
    mean_y: f32,
    std_x: f32,
    std_y: f32,
    n: f32,
) -> f32 {
    // 计算分母
    let denominator = std_x * std_y * n;
    if denominator == 0.0 {
        f32::NAN // 如果分母为0，返回NaN表示不适用
    } else {
        // 计算分子：两个向量的协方差
        let numerator: f32 = x.iter()
            .zip(y.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum();
        numerator / denominator // 返回皮尔逊相关系数
    }
}

/// `rank_first`函数为一维数组排名，使用 'first' 方法处理相同值
fn rank_first(row: ArrayView1<f32>) -> Array1<u32> {
    let mut ranks = vec![0u32; row.len()]; // 初始化排名向量
    let mut values_indices: Vec<(f32, usize)> = row.iter().cloned().zip(0..row.len()).collect();
    values_indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // 降序排序
    for (rank, &(_, index)) in values_indices.iter().enumerate() {
        ranks[index] = (rank + 1) as u32; // 根据排序结果赋予排名
    }
    Array1::from(ranks) // 返回排名数组
}

/// 函数对二维数组的每一行进行排名
fn rank_2d_array(array: ArrayView2<f32>) -> Array2<u32> {
    let shape = array.raw_dim(); // 获取数组维度
    let mut ranked_array = Array2::<u32>::zeros(shape); // 初始化排名数组

    // 并行迭代每一行进行排名
    ranked_array.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(row_idx, mut row)| {
            let rank_row = rank_first(array.row(row_idx)); // 使用rank_first函数对行排名
            row.assign(&rank_row.view()); // 将排名结果赋值到对应行
        });

    ranked_array // 返回排名二维数组
}

/// 将二维数组的行根据其元素值排名，并将结果转化为Python的ndarray返回
#[pyfunction]
fn rank_matrix(py: Python, matrix: PyReadonlyArray2<f32>) -> Py<PyArray2<u32>> {
    let matrix = matrix.as_array(); // 转换为ndarray
    let ranked_matrix = rank_2d_array(matrix); // 进行排名
    ranked_matrix.into_pyarray_bound(py).into() // 转换成Python对象返回
}

/// 计算两个稀疏向量之间的皮尔逊相关系数
fn pearson_correlation2(x: sprs::CsVecView<f64>, y: sprs::CsVecView<f64>) -> f64 {
    fn sum_x_x2(x: sprs::CsVecView<f64>) -> (f64, f64) {
        x.data().iter().fold((0.0, 0.0), |(x0, x1), &x| (x0 + x, x1 + x.powi(2)))
    }

    assert_eq!(x.dim(), y.dim()); // 确保向量维度相同

    let (sum_x, sum_x2) = sum_x_x2(x); // 计算x的和及平方和
    let (sum_y, sum_y2) = sum_x_x2(y); // 计算y的和及平方和

    let sum_xy = x.dot(&y); // 计算x与y的点积

    let n = x.dim() as f64; // 向量的长度
    let numerator = n * sum_xy - sum_x * sum_y; // 计算皮尔逊相关系数的分子

    // 计算分母部分
    let denominator_x = (n * sum_x2 - sum_x.pow(2)).sqrt();
    let denominator_y = (n * sum_y2 - sum_y.pow(2)).sqrt();

    let denominator = denominator_x * denominator_y;
    if denominator == 0.0 {
        f64::NAN // 如果分母为0，返回NaN
    } else {
        numerator / denominator // 返回皮尔逊相关系数
    }
}

/// 使用稀疏矩阵计算给定列索引对之间的相关系数
#[pyfunction]
fn corr_by_index_rs2(
    py: Python<'_>,
    mtx: &PyAny,
    col_idx_pairs: PyReadonlyArrayDyn<usize>,
) -> PyResult<PyObject> {
    let col_idx_pairs = col_idx_pairs.as_array();

    // 从Python对象中提取稀疏矩阵的组成部分并构造`sprs`矩阵
    let data: Vec<f64> = mtx.getattr("data")?.extract()?;
    let indices: Vec<usize> = mtx.getattr("indices")?.extract()?;
    let indptr: Vec<usize> = mtx.getattr("indptr")?.extract()?;
    let shape: (usize, usize) = mtx.getattr("shape")?.extract()?;

    let mtx = CsMat::new_csc((shape.0, shape.1), indptr, indices, data);
    println!("转化到 Rust 完成，开始计算相关性。");

    // 进度条配置
    let style = ProgressStyle::default_bar()
        .template("{wide_bar} {pos}/{len}    {per_sec} iter/s {msg}")?;

    // 并行计算每一个列对的皮尔逊相关系数
    let corr_values: Vec<f64> = col_idx_pairs
        .axis_iter(ndarray::Axis(0))
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

    Ok(corr_values.into_pyarray_bound(py).into()) // 返回结果转成Python对象
}

/// 计算密集矩阵中给定列对的皮尔逊相关系数
#[pyfunction]
fn corr_by_index_rs(
    py: Python<'_>,
    mtx: PyReadonlyArrayDyn<f32>,
    col_idx_pairs: PyReadonlyArrayDyn<usize>,
    mean: PyReadonlyArrayDyn<f32>,
    stdv: PyReadonlyArrayDyn<f32>,
) -> PyResult<PyObject> {
    let mtx = mtx.as_array();
    let col_idx_pairs = col_idx_pairs.as_array();
    let mean = mean.as_array();
    let stdv = stdv.as_array();

    // 并行计算每一个列对的皮尔逊相关系数
    let corr_values: Vec<f32> = col_idx_pairs
        .axis_iter(ndarray::Axis(0))
        .into_par_iter()
        .map(|row| {
            let idx1 = row[0];
            let idx2 = row[1];
            let col1 = mtx.slice(s![.., idx1]);
            let col2 = mtx.slice(s![.., idx2]);
            let n = col1.len() as f32;

            pearson_correlation(&col1, &col2, mean[idx1], mean[idx2], stdv[idx1], stdv[idx2], n)
        })
        .collect();

    Ok(corr_values.into_pyarray_bound(py).into()) // 返回结果转成Python对象
}

/// 定义Python模块`scenic_rs`
#[pymodule]
fn scenic_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(corr_by_index_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rank_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(corr_by_index_rs2, m)?)?;
    Ok(())
}
