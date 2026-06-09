#pragma once
#include "matrix.h"

// MPI并行版上双对角化
// 所有进程都必须调用此函数。
// rank 0 负责初始化和最终输出，各进程按行分配参与 Householder 变换。
Matrix to_bidiagonal_mpi(const Matrix &A, Matrix &U, Matrix &V);
