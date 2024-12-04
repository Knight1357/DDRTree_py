g++ -o test_ddr_tree test_ddr_tree.cpp DDRTree.cpp \
-I/opt/miniconda/envs/r42/lib/R/library/Rcpp/include \
-I/opt/miniconda/envs/r42/lib/R/library/RcppEigen/include \
-I/opt/miniconda/envs/r42/lib/R/include \
-I/opt/miniconda/envs/r42/include \
-I/opt/miniconda/envs/r42/include/python3.12 \
-march=native -std=c++17 -DEIGEN_DONT_VECTORIZE -Wno-ignored-attributes > compile_log.txt 2>&1
