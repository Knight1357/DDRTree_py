#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // 用于 Eigen 类型
#include "DDRTree.h"
#include <iostream>
namespace py = pybind11;

// pca_projection 函数绑定
py::array_t<double> pca_projection_py(py::array_t<double> R_C, int dimensions)
{
    Eigen::Map<Eigen::MatrixXd> C(R_C.mutable_data(), R_C.shape(0), R_C.shape(1));
    Eigen::MatrixXd W;
    pca_projection_cpp(C, dimensions, W);
    return py::array_t<double>({W.rows(), W.cols()}, W.data());
}

// sqdist 函数绑定
py::array_t<double> sqdist_py(py::array_t<double> R_a, py::array_t<double> R_b)
{
    Eigen::Map<Eigen::MatrixXd> a(R_a.mutable_data(), R_a.shape(0), R_a.shape(1));
    Eigen::Map<Eigen::MatrixXd> b(R_b.mutable_data(), R_b.shape(0), R_b.shape(1));
    Eigen::MatrixXd W;
    sq_dist_cpp(a, b, W);
    return py::array_t<double>({W.rows(), W.cols()}, W.data());
}

// DDRTree_reduce_dim 函数绑定
py::dict DDRTree_reduce_dim_py(py::array_t<double> R_X, py::array_t<double> R_Z, py::array_t<double> R_Y, py::array_t<double> R_W, int dimensions, int maxiter, int num_clusters, double sigma, double lambda_, double gamma, double eps, bool verbose)
{

    if (R_X.ndim() != 2 || R_X.dtype() != py::dtype::of<double>())
    {
        throw std::invalid_argument("Input X must be a 2D numpy array of type float64.");
    }
    if (R_Z.ndim() != 2 || R_Z.dtype() != py::dtype::of<double>())
    {
        throw std::invalid_argument("Input Z must be a 2D numpy array of type float64.");
    }
    if (R_Y.ndim() != 2 || R_Y.dtype() != py::dtype::of<double>())
    {
        throw std::invalid_argument("Input Y must be a 2D numpy array of type float64.");
    }
    if (R_W.ndim() != 2 || R_W.dtype() != py::dtype::of<double>())
    {
        throw std::invalid_argument("Input W must be a 2D numpy array of type float64.");
    }
    py::print("X shape:", R_X.shape(0), R_X.shape(1));
    py::print("Z shape:", R_Z.shape(0), R_Z.shape(1));
    py::print("Y shape:", R_Y.shape(0), R_Y.shape(1));
    py::print("W shape:", R_W.shape(0), R_W.shape(1));

    Eigen::Map<Eigen::MatrixXd> X(R_X.mutable_data(), R_X.shape(0), R_X.shape(1));
    Eigen::Map<Eigen::MatrixXd> Z(R_Z.mutable_data(), R_Z.shape(0), R_Z.shape(1));
    Eigen::Map<Eigen::MatrixXd> Y(R_Y.mutable_data(), R_Y.shape(0), R_Y.shape(1));
    Eigen::Map<Eigen::MatrixXd> W(R_W.mutable_data(), R_W.shape(0), R_W.shape(1));

    Eigen::MatrixXd Y_res, Z_res, W_out, Q, R;
    Eigen::SparseMatrix<double> stree_res;
    std::vector<double> objective_vals;

    DDRTree_reduce_dim_cpp(X, Z, Y, W, dimensions, maxiter, num_clusters, sigma, lambda_, gamma, eps, verbose, Y_res, stree_res, Z_res, W_out, Q, R, objective_vals);

    py::dict result;
    result["W"] = py::array_t<double>({W_out.rows(), W_out.cols()}, W_out.data());
    result["Z"] = py::array_t<double>({Z_res.rows(), Z_res.cols()}, Z_res.data());
    // 将稀疏矩阵转换为稠密矩阵
    Eigen::MatrixXd stree_dense = Eigen::MatrixXd(stree_res);
    // 获取稠密矩阵的形状
    std::vector<std::size_t> shape = {static_cast<std::size_t>(stree_dense.rows()), static_cast<std::size_t>(stree_dense.cols())};
    // 将稠密矩阵的指针传给 py::array_t
    result["stree"] = py::array_t<double>(shape, stree_dense.data());

    result["Y"] = py::array_t<double>({Y_res.rows(), Y_res.cols()}, Y_res.data());
    result["Q"] = py::array_t<double>({Q.rows(), Q.cols()}, Q.data());
    result["R"] = py::array_t<double>({R.rows(), R.cols()}, R.data());
    result["objective_vals"] = py::array_t<double>({static_cast<pybind11::size_t>(objective_vals.size())}, objective_vals.data());
    return result;
}

PYBIND11_MODULE(ddr_tree, m)
{
    m.def("pca_projection", &pca_projection_py, "Perform PCA projection");
    m.def("sqdist", &sqdist_py, "Compute squared distances");
    m.def("DDRTree_reduce_dim", &DDRTree_reduce_dim_py, "Reduce dimensions using DDRTree");
}
