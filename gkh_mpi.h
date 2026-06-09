#pragma once
#include "matrix.h"

// MPI并行版GKH迭代（主从式任务池 + 改变迭代方式）
// 要求在 MPI_Init 之后调用，所有进程都必须调用此函数。
// rank 0 为主进程，负责任务分发与结果收集；
// rank 1..size-1 为从进程，负责对子矩阵执行 bulge chase。
// 单进程时退化为串行版本。
bool gkh_svd_from_bidiagonal_mpi(Matrix &U, Matrix &B, Matrix &V,
                                  int max_iter = 6000,
                                  double tol = 1e-12);
