#include "DDRTree.h"

#include <boost/functional/hash.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <iostream>

// using namespace boost;
// using boost::functional;
using namespace Eigen;

typedef Eigen::SparseMatrix<double> SpMat;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

Eigen::MatrixXd pca_projection(const Eigen::MatrixXd &C, int dimensions)
{
    EigenSolver<MatrixXd> es(C, true);

    MatrixXd eVecs = es.eigenvectors().real();
    VectorXd eVals = es.eigenvalues().real();

    // 排序并选择前几个特征
    std::vector<std::pair<double, int>> D;
    for (int i = 0; i < eVals.size(); i++)
    {
        D.emplace_back(eVals[i], i);
    }
    std::sort(D.rbegin(), D.rend());

    MatrixXd W(C.rows(), dimensions);
    for (int i = 0; i < dimensions; i++)
    {
        W.col(i) = eVecs.col(D[i].second);
    }
    return W;
}

void pca_projection_cpp(const MatrixXd &C, int dimensions, MatrixXd &W)
{
    EigenSolver<MatrixXd> es(C, true);

    MatrixXd eVecs = es.eigenvectors().real();
    VectorXd eVals = es.eigenvalues().real();

    // Sort by ascending eigenvalues:
    std::vector<std::pair<double, MatrixXd::Index>> D;
    D.reserve(eVals.size());
    for (MatrixXd::Index i = 0; i < eVals.size(); i++)
        D.push_back(std::make_pair<double, MatrixXd::Index>((double)eVals.coeff(i, 0), (long)i));
    std::sort(D.rbegin(), D.rend());
    MatrixXd sortedEigs;
    sortedEigs.resize(eVecs.rows(), dimensions);
    for (int i = 0; i < eVals.size() && i < dimensions; i++)
    {
        eVals.coeffRef(i, 0) = D[i].first;
        sortedEigs.col(i) = eVecs.col(D[i].second);
    }
    W = sortedEigs;
}

Eigen::MatrixXd sqdist(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
    VectorXd aa = A.array().square().colwise().sum();
    VectorXd bb = B.array().square().colwise().sum();
    MatrixXd ab = A.transpose() * B;

    MatrixXd dist = (aa.replicate(1, B.cols()) + bb.transpose().replicate(A.cols(), 1) - 2 * ab).array().abs();
    return dist;
}

void sq_dist_cpp(const MatrixXd &a, const MatrixXd &b, MatrixXd &W)
{
    // 计算 a 和 b 的列范数的平方
    VectorXd aa = a.colwise().squaredNorm();
    VectorXd bb = b.colwise().squaredNorm();

    // 计算矩阵乘积
    MatrixXd ab = a.transpose() * b;

    // 使用广播计算平方距离矩阵
    W = aa.replicate(1, bb.size()) + bb.transpose().replicate(aa.size(), 1) - 2 * ab;
    W = W.array().abs();
}

void DDRTree_reduce_dim_cpp(const Eigen::MatrixXd &X_in,
                            const Eigen::MatrixXd &Z_in,
                            const Eigen::MatrixXd &Y_in,
                            const Eigen::MatrixXd &W_in,
                            int dimensions,
                            int maxIter,
                            int num_clusters,
                            double sigma,
                            double lambda,
                            double gamma,
                            double eps,
                            bool verbose,
                            Eigen::MatrixXd &Y_out,
                            Eigen::SparseMatrix<double> &stree,
                            Eigen::MatrixXd &Z_out,
                            Eigen::MatrixXd &W_out,
                            Eigen::MatrixXd &Q,
                            Eigen::MatrixXd &R,
                            std::vector<double> &objective_vals)
{
    Y_out = Y_in;
    W_out = W_in;
    Z_out = Z_in;

    int N_cells = X_in.cols();
    /*
        typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
        typedef boost::adjacency_matrix<
                                      boost::undirectedS, boost::no_property,
                                      EdgeWeightProperty> Graph;
        typedef boost::graph_traits < Graph >::edge_descriptor Edge;
        typedef boost::graph_traits < Graph >::vertex_descriptor Vertex;

        if (verbose)
            std::cout << "setting up adjacency matrix" << std::endl;
        Graph g(Y_in.cols());
        for (std::size_t j = 0; j < Y_in.cols(); ++j) {
            for (std::size_t i = 0; i < Y_in.cols() && i <= j ; ++i) {
                Edge e; bool inserted;
                tie(e, inserted) = add_edge(i, j, g);
            }
        }
    */
    using namespace boost;
    typedef boost::property<boost::edge_weight_t, double> EdgeWeightProperty;
    typedef boost::adjacency_list<vecS, vecS, undirectedS,
                                  property<vertex_distance_t, double>, property<edge_weight_t, double>>
        Graph;

    typedef boost::graph_traits<Graph>::edge_descriptor Edge;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_iterator edge_iter;

    Graph g(Y_in.cols());
    // property_map<Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
    for (std::size_t j = 0; j < Y_in.cols(); ++j)
    {
        for (std::size_t i = 0; i < Y_in.cols() && i <= j; ++i)
        {
            if (i != j)
            {
                Edge e;
                bool inserted;
                tie(e, inserted) = add_edge(i, j, g);
            }
        }
    }

    boost::property_map<Graph, boost::edge_weight_t>::type EdgeWeightMap = get(boost::edge_weight_t(), g);

    MatrixXd B = MatrixXd::Zero(Y_in.cols(), Y_in.cols());

    std::vector<graph_traits<Graph>::vertex_descriptor>
        old_spanning_tree(num_vertices(g));

    // std::vector<double> objective_vals;

    MatrixXd distsqMU;
    MatrixXd L;
    MatrixXd distZY;
    distZY.resize(X_in.cols(), num_clusters);

    MatrixXd min_dist;
    min_dist.resize(X_in.cols(), num_clusters);

    MatrixXd tmp_distZY;
    tmp_distZY.resize(X_in.cols(), num_clusters);

    // SpMat tmp_R(X_in.cols(), num_clusters);
    MatrixXd tmp_R;
    tmp_R.resize(X_in.cols(), num_clusters);

    // SpMat R(X_in.cols(), num_clusters);
    R.resize(tmp_R.rows(), num_clusters);

    // SpMat Gamma(R.cols(), R.cols());
    MatrixXd Gamma = MatrixXd::Zero(R.cols(), R.cols());

    SpMat tmp(Gamma.rows(), Gamma.cols());

    MatrixXd tmp_dense;
    tmp_dense.resize(Gamma.rows(), Gamma.cols());

    // SpMat Q;
    Q.resize(tmp_dense.rows(), R.rows());

    MatrixXd C;
    C.resize(X_in.rows(), Q.cols());

    MatrixXd tmp1;
    tmp1.resize(C.rows(), X_in.rows());

    // 移除 Rcpp 环境依赖，直接调用本地的 PCA 函数

    for (int iter = 0; iter < maxIter; ++iter)
    {
        if (verbose)
            std::cout << "************************************** " << std::endl;
        if (verbose)
            std::cout << "Iteration: " << iter << std::endl;

        sq_dist_cpp(Y_out, Y_out, distsqMU);
        // std::cout << "distsqMU: " << distsqMU<< std::endl;
        std::pair<edge_iter, edge_iter> edgePair;
        if (verbose)
            std::cout << "updating weights in graph" << std::endl;
        for (edgePair = edges(g); edgePair.first != edgePair.second; ++edgePair.first)
        {
            if (source(*edgePair.first, g) != target(*edgePair.first, g))
            {
                // std::cout << "edge: " << source(*edgePair.first,g) << " " << target(*edgePair.first,g) << " : " << distsqMU(source(*edgePair.first,g), target(*edgePair.first,g)) << std::endl;
                EdgeWeightMap[*edgePair.first] = distsqMU(source(*edgePair.first, g), target(*edgePair.first, g));
            }
        }

        std::vector<graph_traits<Graph>::vertex_descriptor>
            spanning_tree(num_vertices(g));

        if (verbose)
            std::cout << "Finding MST" << std::endl;
        prim_minimum_spanning_tree(g, &spanning_tree[0]);

        if (verbose)
            std::cout << "Refreshing B matrix" << std::endl;
        // update the adjacency matrix. First, erase the old edges
        for (size_t ei = 0; ei < old_spanning_tree.size(); ++ei)
        {
            // if (ei != old_spanning_tree[ei]){
            B(ei, old_spanning_tree[ei]) = 0;
            B(old_spanning_tree[ei], ei) = 0;
            //            }
        }

        // now add the new edges
        for (size_t ei = 0; ei < spanning_tree.size(); ++ei)
        {
            if (ei != spanning_tree[ei])
            {
                B(ei, spanning_tree[ei]) = 1;
                B(spanning_tree[ei], ei) = 1;
            }
        }
        // std::cout << "B: " << std::endl << B << std::endl;
        if (verbose)
            std::cout << "   B : (" << B.rows() << " x " << B.cols() << ")" << std::endl;

        old_spanning_tree = spanning_tree;

        L = B.colwise().sum().asDiagonal();
        L = L - B;
        // std::cout << "   Z_out nan check : (" << Z_out.rows() << "x" << Z_out.cols() << ", " << Z_out.maxCoeff() << " )" << std::endl;

        // std::cout << "   Y_out nan check : (" << Y_out.rows() << "x" << Y_out.cols() << ", " << Y_out.maxCoeff() << " )" << std::endl;

        sq_dist_cpp(Z_out, Y_out, distZY);
        // std::cout << "   distZY nan check : (" << distZY.maxCoeff() << " )" << std::endl;
        if (verbose)
            std::cout << "   distZY : (" << distZY.rows() << " x " << distZY.cols() << ")" << std::endl;

        if (verbose)
            std::cout << "   min_dist : (" << min_dist.rows() << " x " << min_dist.cols() << ")" << std::endl;
        // min_dist <- matrix(rep(apply(distZY, 1, min), times = K), ncol = K, byrow = F)

        VectorXd distZY_minCoeff = distZY.rowwise().minCoeff();
        if (verbose)
            std::cout << "distZY_minCoeff = " << std::endl;
        for (int i = 0; i < min_dist.cols(); i++)
        {
            min_dist.col(i) = distZY_minCoeff;
        }
        // std::cout << min_dist << std::endl;

        // tmp_distZY <- distZY - min_dist
        tmp_distZY = distZY - min_dist;
        // std::cout << tmp_distZY << std::endl;

        if (verbose)
            std::cout << "   tmp_R : (" << tmp_R.rows() << " x " << tmp_R.cols() << ")" << std::endl;
        // tmp_R <- exp(-tmp_distZY / params$sigma)
        tmp_R = tmp_distZY.array() / (-1.0 * sigma);
        // std::cout << tmp_R << std::endl;

        tmp_R = tmp_R.array().exp().matrix();

        // 数值检查
        if (!tmp_R.allFinite())
        {
            std::cerr << "tmp_R contains NaN or Inf at iteration " << iter << std::endl;
            return;
        }

        if (verbose)
            std::cout << "   R : (" << R.rows() << " x " << R.cols() << ")" << std::endl;
        // R <- tmp_R / matrix(rep(rowSums(tmp_R), times = K), byrow = F, ncol = K)

        VectorXd tmp_R_rowsums = tmp_R.rowwise().sum();
        for (int i = 0; i < R.cols(); i++)
        {
            R.col(i) = tmp_R_rowsums;
        }
        // std::cout << R << std::endl;
        // std::cout << "&&&&&" << std::endl;
        R = (tmp_R.array() / R.array()).matrix();
        // std::cout << R << std::endl;

        if (verbose)
            std::cout << "   Gamma : (" << Gamma.rows() << " x " << Gamma.cols() << ")" << std::endl;
        // Gamma <- matrix(rep(0, ncol(R) ^ 2), nrow = ncol(R))
        Gamma = MatrixXd::Zero(R.cols(), R.cols());
        // diag(Gamma) <- colSums(R)
        Gamma.diagonal() = R.colwise().sum();
        // std::cout << Gamma << std::endl;

        // termination condition
        // obj1 <- - params$sigma * sum(log(rowSums(exp(-tmp_distZY / params$sigma))) - min_dist[, 1] / params$sigma)
        VectorXd x1 = (tmp_distZY.array() / -sigma).exp().rowwise().sum().log();
        // std::cout << "Computing x1 " << x1.transpose() << std::endl;
        double obj1 = -sigma * (x1 - min_dist.col(0) / sigma).sum();
        // std::cout << obj1 << std::endl;

        // obj2 <- (norm(X - W %*% Z, '2'))^2 + params$lambda * sum(diag(Y %*% L %*% t(Y))) + params$gamma * obj1 #sum(diag(A))
        // Rcpp:Rcout << X_in - W_out * Z_out << std::endl;

        if (verbose)
        {
            std::cout << "   X : (" << X_in.rows() << " x " << X_in.cols() << ")" << std::endl;
            std::cout << "   W : (" << W_out.rows() << " x " << W_out.cols() << ")" << std::endl;
            std::cout << "   Z : (" << Z_out.rows() << " x " << Z_out.cols() << ")" << std::endl;
        }

        double obj2 = (X_in - W_out * Z_out).norm();
        // std::cout << "norm = " << obj2 << std::endl;
        obj2 = obj2 * obj2;

        if (verbose)
        {
            std::cout << "   L : (" << L.rows() << " x " << L.cols() << ")" << std::endl;
        }

        obj2 = obj2 + lambda * (Y_out * L * Y_out.transpose()).diagonal().sum() + gamma * obj1;
        // std::cout << obj2 << std::endl;
        // std::cout << "obj2 = " << obj2 << std::endl;
        objective_vals.push_back(obj2);

        if (verbose)
            std::cout << "Checking termination criterion" << std::endl;
        if (iter >= 1)
        {
            double delta_obj = std::abs(objective_vals[iter] - objective_vals[iter - 1]);
            delta_obj /= std::abs(objective_vals[iter - 1]);
            if (verbose)
                std::cout << "delta_obj: " << delta_obj << std::endl;
            if (delta_obj < eps)
            {
                break;
            }
        }

        // std::cout << "L" << std::endl;
        // std::cout << L << std::endl;
        if (verbose)
            std::cout << "Computing tmp" << std::endl;
        // tmp <- t(solve( ( ( (params$gamma + 1) / params$gamma) * ((params$lambda / params$gamma) * L + Gamma) - t(R) %*% R), t(R)))

        if (verbose)
            std::cout << "... stage 1" << std::endl;
        tmp = ((Gamma + (L * (lambda / gamma))) * ((gamma + 1.0) / gamma)).sparseView();
        // std::cout << tmp << std::endl;
        if (verbose)
        {
            std::cout << "... stage 2" << std::endl;
            // std::cout << R.transpose() << std::endl;
        }

        SparseMatrix<double> R_sp = R.sparseView();
        tmp = tmp - (R_sp.transpose() * R_sp);
        // tmp = tmp_dense.sparseView();

        if (verbose)
        {
            std::cout << "Pre-computing LLT analysis" << std::endl;
            std::cout << "tmp is (" << tmp.rows() << "x" << tmp.cols() << "), " << tmp.nonZeros() << " non-zero values" << std::endl;
        }

        // std::cout << tmp << std::endl;
        SimplicialLLT<SparseMatrix<double>, Lower, AMDOrdering<int>> solver;
        solver.compute(tmp);
        if (solver.info() != Success)
        {
            // decomposition failed
            std::cout << "Error!" << std::endl;
            tmp_dense = tmp;
            tmp_dense = tmp_dense.partialPivLu().solve(R.transpose()).transpose();
            std::cout << tmp_dense << std::endl;
        }
        else
        {
            if (verbose)
                std::cout << "Computing LLT" << std::endl;
            tmp_dense = solver.solve(R.transpose()).transpose();
            if (solver.info() != Success)
            {
                // solving failed
                std::cout << "Error!" << std::endl;
            }
        }

        // tmp_dense = tmp_dense.llt().solve(R.transpose()).transpose();
        if (verbose)
            std::cout << "tmp_dense " << tmp_dense.rows() << "x" << tmp_dense.cols() << ") " << std::endl;

        if (verbose)
            std::cout << "Computing Q " << Q.rows() << "x" << Q.cols() << ") " << std::endl;
        // Q <- 1 / (params$gamma + 1) * (diag(1, N) + tmp %*% t(R))
        // tmp = tmp_dense.sparseView();
        // std::cout << "tmp_dense is (" << tmp_dense.rows() << "x" << tmp_dense.cols() <<"), " << tmp_dense.nonZeros() << " non-zero values" << std::endl;
        // std::cout << "R_sp is (" << R_sp.rows() << "x" << R_sp.cols() <<"), " << R_sp.nonZeros() << " non-zero values" << std::endl;

        /////////////////////////
        /*
        double gamma_coeff = 1.0 / (1 + gamma);

        SpMat Q_id(tmp_dense.rows(), R.rows());
        Q_id.setIdentity();

        tmp1 = gamma_coeff * (X_in * tmp_dense.sparseView());
        if (verbose)
            std::cout << "First tmp1 product complete: " << tmp1.rows() << "x" << tmp1.cols() <<"), " << tmp1.nonZeros() << " non-zero values" << std::endl;
        tmp1 = tmp1 * R_sp.transpose();
        if (verbose)
            std::cout << "Second tmp1 product complete: " << tmp1.rows() << "x" << tmp1.cols() <<"), " << tmp1.nonZeros() << " non-zero values" << std::endl;
        tmp1 += gamma_coeff * X_in;
        if (verbose)
            std::cout << "Third tmp1 product complete: " << tmp1.rows() << "x" << tmp1.cols() <<"), " << tmp1.nonZeros() << " non-zero values" << std::endl;
        tmp1 = tmp1 * X_in.transpose();
        if (verbose)
            std::cout << "Final tmp1 product complete: " << tmp1.rows() << "x" << tmp1.cols() <<"), " << tmp1.nonZeros() << " non-zero values" << std::endl;
        */
        ///////////////////////////
        /*
                 Q = ((MatrixXd::Identity(X_in.cols(), X_in.cols()) + (tmp_dense * R.transpose()) ).array() / (gamma + 1.0));

                 if (verbose){
                std::cout << "gamma: " << gamma << std::endl;
                std::cout << "   X_in : (" << X_in.rows() << " x " << X_in.cols() << ")" << std::endl;
                std::cout << "   Q : (" << Q.rows() << " x " << Q.cols() << ")" << std::endl;
                //std::cout << Q << std::endl;
                 }

                // C <- X %*% Q
                C = X_in * Q;
                if (verbose)
                std::cout << "   C : (" << C.rows() << " x " << C.cols() << ")" << std::endl;
                tmp1 =  C * X_in.transpose();
        */
        /////////////////////////

        Q = (X_in + ((X_in * tmp_dense) * R.transpose())).array() / (gamma + 1.0);

        if (verbose)
        {
            std::cout << "gamma: " << gamma << std::endl;
            std::cout << "   X_in : (" << X_in.rows() << " x " << X_in.cols() << ")" << std::endl;
            std::cout << "   Q : (" << Q.rows() << " x " << Q.cols() << ")" << std::endl;
            // std::cout << Q << std::endl;
        }

        // C <- X %*% Q
        // C = X_in * Q;
        C = Q;

        tmp1 = Q * X_in.transpose();

        /////////////////////////

        // std::cout << tmp1 << std::endl;

        // std::cout << tmp1 << std::endl;

        if (verbose)
        {
            std::cout << "Computing W" << std::endl;
            // std::cout << "tmp1 = " << std::endl;
            // std::cout << tmp1 << std::endl;
            // std::cout << (tmp1 + tmp1.transpose()) / 2 << std::endl;
        }

        // W <- pca_projection_R((tmp1 + t(tmp1)) / 2, params$dim)

        Eigen::MatrixXd W = pca_projection((tmp1 + tmp1.transpose()) / 2, dimensions);

        W_out = W;
        // pca_projection_cpp((tmp1 + tmp1.transpose()) / 2, dimensions, W_out);
        // std::cout << W_out << std::endl;

        if (verbose)
            std::cout << "Computing Z" << std::endl;
        // Z <- t(W) %*% C
        Z_out = W_out.transpose() * C;
        // std::cout << Z_out << std::endl;

        if (verbose)
            std::cout << "Computing Y" << std::endl;
        // Y <- t(solve((params$lambda / params$gamma * L + Gamma), t(Z %*% R)))
        Y_out = L * (lambda / gamma) + Gamma;
        Y_out = Y_out.llt().solve((Z_out * R).transpose()).transpose();

        // std::cout << Y_out << std::endl;
    }

    if (verbose)
        std::cout << "Clearing MST sparse matrix" << std::endl;
    stree.setZero();

    if (verbose)
    {
        std::cout << "Setting up MST sparse matrix with " << old_spanning_tree.size() << std::endl;
    }
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(2 * old_spanning_tree.size());
    // Send back the weighted MST as a sparse matrix
    for (size_t ei = 0; ei < old_spanning_tree.size(); ++ei)
    {
        // stree.insert(source(*ei, g), target(*ei, g)) = 1;//distsqMU(source(*ei, g), target(*ei, g));
        tripletList.push_back(T(ei, old_spanning_tree[ei], distsqMU(ei, old_spanning_tree[ei])));
        tripletList.push_back(T(old_spanning_tree[ei], ei, distsqMU(old_spanning_tree[ei], ei)));
    }
    stree = SpMat(N_cells, N_cells);
    stree.setFromTriplets(tripletList.begin(), tripletList.end());

    // Q = Q / X;
}

std::map<std::string, Eigen::MatrixXd> DDRTree_reduce_dim(
    const Eigen::MatrixXd &X, const Eigen::MatrixXd &Z, const Eigen::MatrixXd &Y,
    const Eigen::MatrixXd &W, int dimensions, int maxiter, int num_clusters,
    double sigma, double lambda, double gamma, double eps, bool verbose)
{

    // std::cout << "Mapping verbose" << std::endl;

    if (verbose)
    {
        std::cout << "X dimensions: " << X.rows() << "x" << X.cols() << std::endl;
        std::cout << "Z dimensions: " << Z.rows() << "x" << Z.cols() << std::endl;
        std::cout << "Y dimensions: " << Y.rows() << "x" << Y.cols() << std::endl;
        std::cout << "W dimensions: " << W.rows() << "x" << W.cols() << std::endl;
        std::cout << "Other parameters: dimensions=" << dimensions
                  << ", maxiter=" << maxiter
                  << ", num_clusters=" << num_clusters
                  << ", sigma=" << sigma
                  << ", lambda=" << lambda
                  << ", gamma=" << gamma
                  << ", eps=" << eps << std::endl;
    }

    const int X_n = X.rows(), X_p = X.cols();

    if (verbose)
        std::cout << "Mapping Z" << std::endl;

    const int Z_n = Z.rows(), Z_p = Z.cols();

    if (verbose)
        std::cout << "Mapping Y" << std::endl;

    const int Y_n = Y.rows(), Y_p = Y.cols();

    if (verbose)
        std::cout << "Mapping W" << std::endl;

    const int W_n = W.rows(), W_p = W.cols();

    MatrixXd Y_res;
    SpMat stree_res;
    MatrixXd Z_res;
    MatrixXd W_out;
    std::vector<double> objective_vals; // a vector for the value for the objective function at each iteration
    MatrixXd Q;
    MatrixXd R;

    DDRTree_reduce_dim_cpp(X, Z, Y, W, dimensions, maxiter, num_clusters, sigma, lambda, gamma, eps, verbose, Y_res, stree_res, Z_res, W_out, Q, R, objective_vals);

    std::map<std::string, Eigen::MatrixXd> result;
    result["W"] = W_out;
    result["Z"] = Z_res;
    result["Y"] = Y_res;
    result["Q"] = Q;
    result["R"] = R;
    result["objective_vals"] = Eigen::MatrixXd::Map(objective_vals.data(), objective_vals.size(), 1); // 将 vector 转为列矩阵

    // 注意：Eigen 不支持直接存储 SparseMatrix，你可以跳过或者处理为稠密矩阵
    result["stree"] = Eigen::MatrixXd(stree_res); // 将稀疏矩阵转换为稠密矩阵返回

    return result;
}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically
// run after the compilation.
//
