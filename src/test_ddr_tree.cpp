#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "DDRTree.h"

int main() {
    using namespace Eigen;
    using namespace std;

    // 输入数据初始化
    MatrixXd X_in = MatrixXd::Random(10, 5);  // 10 样本, 5 特征
    MatrixXd Z_in = MatrixXd::Random(5, 3);   // 5 特征降至 3 维
    MatrixXd Y_in = MatrixXd::Random(10, 3);  // 初始 Y
    MatrixXd W_in = MatrixXd::Random(3, 3);   // 初始 W

    int dimensions = 2;         // 降维目标维度
    int maxIter = 10;           // 最大迭代次数
    int num_clusters = 3;       // 聚类数量
    double sigma = 0.5;         // 参数 sigma
    double lambda = 0.1;        // 参数 lambda
    double gamma = 0.1;         // 参数 gamma
    double eps = 1e-5;          // 终止条件
    bool verbose = true;        // 是否打印详细信息

    // 输出变量初始化
    MatrixXd Y_out;
    SparseMatrix<double> stree;
    MatrixXd Z_out;
    MatrixXd W_out;
    MatrixXd Q;
    MatrixXd R;
    vector<double> objective_vals;

    // 调用函数
    DDRTree_reduce_dim_cpp(X_in, Z_in, Y_in, W_in, dimensions, maxIter, num_clusters,
                           sigma, lambda, gamma, eps, verbose,
                           Y_out, stree, Z_out, W_out, Q, R, objective_vals);

    // 输出结果
    cout << "Y_out:\n" << Y_out << endl;
    cout << "Z_out:\n" << Z_out << endl;
    cout << "W_out:\n" << W_out << endl;
    cout << "Objective Values:\n";
    for (double obj : objective_vals) {
        cout << obj << " ";
    }
    cout << endl;

    return 0;
}
