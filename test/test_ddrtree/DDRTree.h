#ifndef _DDRTree_DDRTREE_H
#define _DDRTree_DDRTREE_H

#include <Rcpp.h>
#include <Eigen/Dense> // 修改为标准 Eigen
#include <Eigen/Sparse>

using namespace Eigen;
// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

void pca_projection_cpp(const MatrixXd &R_C, int dimensions, MatrixXd &W);
void sq_dist_cpp(const MatrixXd &a, const MatrixXd &b, MatrixXd &W);
// DDRTree.h 中的声明
void DDRTree_reduce_dim_cpp(const Eigen::MatrixXd& X_in,
                            const Eigen::MatrixXd& Z_in,
                            const Eigen::MatrixXd& Y_in,
                            const Eigen::MatrixXd& W_in,
                            int dimensions,
                            int maxIter,
                            int num_clusters,
                            double sigma,
                            double lambda,
                            double gamma,
                            double eps,
                            bool verbose,
                            Eigen::MatrixXd& Y_out,
                            Eigen::SparseMatrix<double>& stree,
                            Eigen::MatrixXd& Z_out,
                            Eigen::MatrixXd& W_out,
                            Eigen::MatrixXd& Q,
                            Eigen::MatrixXd& R,
                            std::vector<double>& objective_vals);


#endif