#pragma once
#include "matrix.h"

// OpenMP并行版GKH迭代
bool gkh_svd_from_bidiagonal_omp(Matrix &U, Matrix &B, Matrix &V,
                                  int max_iter, double tol, int num_threads);
