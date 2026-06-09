// bench_mpi.cpp
// MPI 并行 SVD benchmark 程序
// 测试不同矩阵规模、不同进程数下的性能
// 输出 CSV 格式结果
//
// 使用方式：mpiexec -n <num_procs> ./bench_mpi <size> [seed] [reps]

#include "matrix.h"
#include "gkh_mpi.h"
#include "bidiag_mpi.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>

#ifdef _WIN32
#include <windows.h>
#endif

static Matrix transpose(const Matrix &A)
{
    Matrix T(A.cols(), A.rows());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            T.at(j, i) = A.at(i, j);
    return T;
}

static double fro_norm(const Matrix &A)
{
    double s = 0.0;
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            s += A.at(i, j) * A.at(i, j);
    return std::sqrt(s);
}

static double orth_error(const Matrix &Q)
{
    Matrix I = transpose(Q) * Q;
    const int n = I.rows();
    for (int i = 0; i < n; ++i)
        I.at(i, i) -= 1.0;
    return fro_norm(I);
}

static double reconstruction_error(const Matrix &A, const Matrix &U,
                                     const Matrix &S, const Matrix &V)
{
    Matrix R = U * S * transpose(V);
    double s = 0.0;
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
        {
            const double d = A.at(i, j) - R.at(i, j);
            s += d * d;
        }
    return std::sqrt(s);
}

struct RunResult
{
    double bidiag_ms;
    double gkh_ms;
    double total_ms;
    double recon_err_rel;
    double orth_u_err;
    double orth_v_err;
    bool converged;
};

static RunResult run_once(const Matrix &A, int rank)
{
    Matrix U, V;

    double t1 = MPI_Wtime();
    Matrix B = to_bidiagonal_mpi(A, U, V);
    double t2 = MPI_Wtime();

    double t3 = MPI_Wtime();
    bool converged = gkh_svd_from_bidiagonal_mpi(U, B, V, 6000, 1e-12);
    double t4 = MPI_Wtime();

    RunResult res;
    res.bidiag_ms = (t2 - t1) * 1000.0;
    res.gkh_ms = (t4 - t3) * 1000.0;
    res.total_ms = res.bidiag_ms + res.gkh_ms;
    res.converged = converged;

    if (rank == 0)
    {
        double err = reconstruction_error(A, U, B, V);
        res.recon_err_rel = err / (fro_norm(A) + 1.0);
        res.orth_u_err = orth_error(U);
        res.orth_v_err = orth_error(V);
    }
    else
    {
        res.recon_err_rel = 0;
        res.orth_u_err = 0;
        res.orth_v_err = 0;
    }

    return res;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

#ifdef _WIN32
    if (rank == 0)
        SetConsoleOutputCP(65001);
#endif

    if (argc < 2)
    {
        if (rank == 0)
        {
            std::cerr << "用法: mpiexec -n <num_procs> " << argv[0]
                      << " <size> [seed] [reps]" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int size = std::stoi(argv[1]);
    long long seed = (argc >= 3) ? std::stoll(argv[2]) : 20260408LL;
    int reps = (argc >= 4) ? std::stoi(argv[3]) : 3;

    Matrix A = Matrix::random(size, size, -1.0, 1.0, seed);

    RunResult best;
    best.total_ms = 1e30;

    for (int r = 0; r < reps; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        RunResult res = run_once(A, rank);
        if (rank == 0 && res.total_ms < best.total_ms)
            best = res;
    }

    if (rank == 0)
    {
        // CSV 输出
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "mode,size,procs,bidiag_ms,gkh_ms,total_ms,"
                  << "recon_err_rel,orth_u_err,orth_v_err,converged" << std::endl;
        std::cout << "mpi,"
                  << size << ","
                  << mpi_size << ","
                  << best.bidiag_ms << ","
                  << best.gkh_ms << ","
                  << best.total_ms << ","
                  << best.recon_err_rel << ","
                  << best.orth_u_err << ","
                  << best.orth_v_err << ","
                  << (best.converged ? 1 : 0) << std::endl;

        // 人类可读
        std::cerr << "\n=== MPI " << size << "x" << size
                  << " " << mpi_size << "P ===" << std::endl;
        std::cerr << "  bidiag: " << best.bidiag_ms << " ms" << std::endl;
        std::cerr << "  gkh:    " << best.gkh_ms << " ms" << std::endl;
        std::cerr << "  total:  " << best.total_ms << " ms" << std::endl;
        std::cerr << "  recon:  " << best.recon_err_rel << std::endl;
        std::cerr << "  orthU:  " << best.orth_u_err << std::endl;
        std::cerr << "  orthV:  " << best.orth_v_err << std::endl;
        std::cerr << "  conv:   " << (best.converged ? "yes" : "no") << std::endl;
    }

    MPI_Finalize();
    return (rank == 0 && best.converged) ? 0 : 0;
}
