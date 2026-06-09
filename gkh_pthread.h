#pragma once
#include "matrix.h"

// Pthread并行版GKH迭代
bool gkh_svd_from_bidiagonal_pthread(Matrix &U, Matrix &B, Matrix &V,
                                      int max_iter, double tol, int num_threads);
