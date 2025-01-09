use sprs::{CsMat, CsMatBase};
use rand::Rng;

// 矩阵乘法函数
pub fn csr_multiply(a: &CsMat<f64>, b: &CsMat<f64>) -> CsMat<f64> {
    // 检查矩阵维度是否匹配
    assert_eq!(a.cols(), b.rows(), "Matrix dimensions do not match for multiplication");
    // 直接使用 `sprs` 提供的乘法功能
    a * b
}

// 随机生成 CSR 稀疏矩阵
pub fn generate_random_csr(rows: usize, cols: usize, density: f64) -> CsMat<f64> {
    let mut rng = rand::thread_rng();

    // 创建稀疏矩阵的值、行和列索引
    let mut values = Vec::new();
    let mut row_indices = Vec::new();
    let mut col_indices = Vec::new();

    for row in 0..rows {
        for col in 0..cols {
            if rng.gen::<f64>() < density {
                values.push(rng.gen_range(1.0..10.0));
                row_indices.push(row);
                col_indices.push(col);
            }
        }
    }

    // 从三元组格式（COO）创建CSR矩阵
    CsMat::new((rows, cols), row_indices, col_indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_multiply() {
        let rows = 5;
        let cols = 5;
        let density = 0.3;

        // 随机生成两个稀疏矩阵
        let a = generate_random_csr(rows, cols, density);
        let b = generate_random_csr(cols, rows, density);

        // 执行矩阵乘法
        let c = csr_multiply(&a, &b);

        // 打印矩阵及结果
        println!("Matrix A:\n{:?}", a);
        println!("Matrix B:\n{:?}", b);
        println!("Matrix C (A * B):\n{:?}", c);

        // 验证矩阵结果的形状
        assert_eq!(c.rows(), a.rows());
        assert_eq!(c.cols(), b.cols());
    }
}
