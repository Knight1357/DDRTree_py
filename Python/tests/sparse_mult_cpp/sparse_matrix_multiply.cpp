#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <omp.h>
#include <iostream>

// 使用Eigen命名空间
using namespace Eigen;

// 定义稀疏矩阵乘法函数，使用CSR格式和多线程
SparseMatrix<double> sparseMatrixMultiply(const SparseMatrix<double> &A, const SparseMatrix<double> &B, int numThreads)
{
    // 检查矩阵维度是否匹配
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    std::cout << "设置OpenMP的线程数量" << std::endl;
    // 设置OpenMP的线程数量
    omp_set_num_threads(numThreads);

    std::cout << "转置B以便访问列变为按行存储，提高计算效率" << std::endl;
    // 转置B以便访问列变为按行存储，提高计算效率
    SparseMatrix<double> B_transposed = B.transpose();

    // 准备结果矩阵
    SparseMatrix<double> C(A.rows(), B.cols());

    // 用来存储稀疏矩阵的值
    std::vector<Triplet<double>> triplets;
    std::cout << "使用OpenMP并行化外层循环" << std::endl;
// 使用OpenMP并行化外层循环
#pragma omp parallel
    {
        // 每个线程独立存储triplets，避免竞争
        std::vector<Triplet<double>> localTriplets;

#pragma omp for schedule(dynamic)
        for (int i = 0; i < A.outerSize(); ++i)
        { // 遍历A的每一行
            for (SparseMatrix<double>::InnerIterator itA(A, i); itA; ++itA)
            { // 遍历A当前行的非零元素
                int k = itA.col();
                double valA = itA.value();

                for (SparseMatrix<double>::InnerIterator itB(B_transposed, k); itB; ++itB)
                { // 遍历B转置中当前列的非零元素
                    int j = itB.col();
                    double valB = itB.value();
                    localTriplets.emplace_back(i, j, valA * valB);
                }
            }
        }

// 合并线程的triplets
#pragma omp critical
        {
            triplets.insert(triplets.end(), localTriplets.begin(), localTriplets.end());
        }
    }

    // 构造稀疏矩阵C
    C.setFromTriplets(triplets.begin(), triplets.end());

    return C;
}

// 辅助函数，用于生成指定范围内的随机整数
int random_int(int min_val, int max_val)
{
    return min_val + std::rand() % (max_val - min_val + 1);
}

// 主函数示例
int main()
{
    // 设置第一个稀疏矩阵的相关参数（可按需调整）
    int rows_1 = 100000;                              // 行数
    int cols_1 = 2000;                                // 列数
    int num_nonzeros_1 = int(rows_1 * 0.30 * cols_1); // 非零元素个数

    // 用于存储第一个稀疏矩阵的三元组（行索引、列索引、值）
    std::vector<Eigen::Triplet<double>> triplets_1;
    // 随机生成第一个稀疏矩阵的三元组信息
    for (int i = 0; i < num_nonzeros_1; ++i)
    {
        int row = random_int(0, rows_1 - 1);
        int col = random_int(0, cols_1 - 1);
        double value = static_cast<double>(rand()) / RAND_MAX; // 生成0到1之间的随机double值
        triplets_1.push_back(Eigen::Triplet<double>(row, col, value));
    }
    // 创建第一个CSR格式的稀疏矩阵对象
    Eigen::SparseMatrix<double> sparse_matrix_1(rows_1, cols_1);
    sparse_matrix_1.setFromTriplets(triplets_1.begin(), triplets_1.end());

    // 设置第二个稀疏矩阵的相关参数（可按需调整）
    int rows_2 = 2000;
    int cols_2 = 100000;
    int num_nonzeros_2 = int(rows_2 * 0.30 * cols_2);

    // 用于存储第二个稀疏矩阵的三元组
    std::vector<Eigen::Triplet<double>> triplets_2;
    // 随机生成第二个稀疏矩阵的三元组信息
    for (int i = 0; i < num_nonzeros_2; ++i)
    {
        int row = random_int(0, rows_2 - 1);
        int col = random_int(0, cols_2 - 1);
        double value = static_cast<double>(rand()) / RAND_MAX;
        triplets_2.push_back(Eigen::Triplet<double>(row, col, value));
    }
    // 创建第二个CSR格式的稀疏矩阵对象
    Eigen::SparseMatrix<double> sparse_matrix_2(rows_2, cols_2);
    sparse_matrix_2.setFromTriplets(triplets_2.begin(), triplets_2.end());

    // // 输出第一个稀疏矩阵（以密集矩阵形式输出便于查看，实际中不一定需要转换）
    // std::cout << "第一个CSR稀疏矩阵（以密集矩阵形式展示）：" << std::endl;
    // std::cout << sparse_matrix_1 << std::endl;

    // // 输出第二个稀疏矩阵（以密集矩阵形式输出便于查看，实际中不一定需要转换）
    // std::cout << "第二个CSR稀疏矩阵（以密集矩阵形式展示）：" << std::endl;
    // std::cout << sparse_matrix_2 << std::endl;

    // // 输出矩阵A
    // std::cout << "Matrix sparse_matrix_1:" << std::endl;
    // std::cout << sparse_matrix_1 << std::endl;

    // // 输出矩阵B
    // std::cout << "Matrix sparse_matrix_2:" << std::endl;
    // std::cout << sparse_matrix_2 << std::endl;

    std::cout << "Start calculate ... " << std::endl;
    // 调用稀疏矩阵乘法函数
    int numThreads = 4; // 使用4个线程
    SparseMatrix<double> C = sparseMatrixMultiply(sparse_matrix_1, sparse_matrix_2, numThreads);

    // 输出结果矩阵
    for (int k = 0; k < C.outerSize(); ++k)
    {
        for (SparseMatrix<double>::InnerIterator it(C, k); it; ++it)
        {
            std::cout << "C(" << it.row() << ", " << it.col() << ") = " << it.value() << std::endl;
        }
    }

    return 0;
}
