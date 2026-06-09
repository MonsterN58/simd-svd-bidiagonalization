#pragma once
#include "matrix.h"

// Pthread并行版上双对角化
Matrix to_bidiagonal_pthread(const Matrix &A, Matrix &U, Matrix &V, int num_threads);
