#pragma once
#include "matrix.h"

// OpenMP并行版上双对角化
// num_threads: 使用的线程数
Matrix to_bidiagonal_omp(const Matrix &A, Matrix &U, Matrix &V, int num_threads);
