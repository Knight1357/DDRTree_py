#ifndef _DDRTree_DDRTREE_H
#define _DDRTree_DDRTREE_H

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Rcpp;
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

void pca_projection_cpp(const MatrixXd& R_C, int dimensions,  MatrixXd& W);
void sq_dist_cpp(const MatrixXd& a, const MatrixXd& b,  MatrixXd& W);
void DDRTree_reduce_dim_cpp(const MatrixXd& X_in,
                            const MatrixXd& Z_in,
                            const MatrixXd& Y_in,
                            const MatrixXd& W_in,
                            int dimensions,
                            int maxIter,
                            int num_clusters,
                            double sigma,
                            double lambda,
                            double gamma,
                            double eps,
                            bool verbose,
                            MatrixXd& Y_out,
                            SpMat& stree,
                            MatrixXd& Z_out,
                            MatrixXd& W_out,
                            MatrixXd& Q,
                            MatrixXd& R,
                            std::vector<double>& objective_vals);

#endif