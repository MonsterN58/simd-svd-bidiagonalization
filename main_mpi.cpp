// main_mpi.cpp
// MPI 版本主测试程序
// 与 main.cpp 的测试用例保持一致（不修改 main.cpp）
// 使用方式：mpiexec -n <num_procs> ./main_mpi [seed]

#include "matrix.h"
#include "gkh_mpi.h"
#include "bidiag_mpi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
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

static double diagonal_structure_error(const Matrix &S)
{
    double max_abs = 0.0;
    for (int i = 0; i < S.rows(); ++i)
        for (int j = 0; j < S.cols(); ++j)
            if (i != j)
                max_abs = std::max(max_abs, std::fabs(S.at(i, j)));
    return max_abs;
}

static double order_error(const Matrix &S)
{
    const int n = S.cols();
    double worst = 0.0;
    for (int i = 0; i < n - 1; ++i)
    {
        double cur = S.at(i, i);
        double nxt = S.at(i + 1, i + 1);
        if (cur < nxt)
            worst = std::max(worst, nxt - cur);
    }
    return worst;
}

static bool nonnegative_diag(const Matrix &S)
{
    for (int i = 0; i < S.cols(); ++i)
        if (S.at(i, i) < -1e-12)
            return false;
    return true;
}

static bool run_case(const std::string &name, const Matrix &A,
                     double &sum_bidiag_ms, double &sum_gkh_ms, int rank)
{
    if (rank == 0)
        std::cout << "=== " << name << " ===" << std::endl;

    Matrix U, V;

    double t1 = MPI_Wtime();
    Matrix B = to_bidiagonal_mpi(A, U, V);
    double t2 = MPI_Wtime();

    double t3 = MPI_Wtime();
    const bool converged = gkh_svd_from_bidiagonal_mpi(U, B, V, 6000, 1e-12);
    double t4 = MPI_Wtime();

    double time_bidiag_ms = (t2 - t1) * 1000.0;
    double time_gkh_ms = (t4 - t3) * 1000.0;

    if (rank == 0)
    {
        sum_bidiag_ms += time_bidiag_ms;
        sum_gkh_ms += time_gkh_ms;

        const double err_recon = reconstruction_error(A, U, B, V);
        const double err_recon_rel = err_recon / (fro_norm(A) + 1.0);
        const double err_u = orth_error(U);
        const double err_v = orth_error(V);
        const double err_diag = diagonal_structure_error(B);
        const double err_order = order_error(B);
        const bool ok_nonneg = nonnegative_diag(B);

        std::cout << "  converged                 : " << (converged ? "yes" : "no") << "\n";
        std::cout << "  ||A-U*S*V^T||_F           : " << err_recon << "\n";
        std::cout << "  relative recon error      : " << err_recon_rel << "\n";
        std::cout << "  ||U^T U-I||_F             : " << err_u << "\n";
        std::cout << "  ||V^T V-I||_F             : " << err_v << "\n";
        std::cout << "  diagonal structure error  : " << err_diag << "\n";
        std::cout << "  descending order error    : " << err_order << "\n";
        std::cout << "  nonnegative diagonal      : " << (ok_nonneg ? "yes" : "no") << "\n";
        std::cout << "  time bidiagonalization(ms): " << time_bidiag_ms << "\n";
        std::cout << "  time gkh iteration(ms)    : " << time_gkh_ms << "\n";

        const double tol_recon_rel = 1e-8;
        const double tol_orth = 1e-7;
        const double tol_diag = 1e-10;
        const double tol_order = 1e-12;

        const bool pass = converged &&
                          (err_recon_rel < tol_recon_rel) &&
                          (err_u < tol_orth) &&
                          (err_v < tol_orth) &&
                          (err_diag < tol_diag) &&
                          (err_order < tol_order) &&
                          ok_nonneg;

        std::cout << "  结果: " << (pass ? "PASS" : "FAIL") << "\n\n";
        return pass;
    }

    return true;
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

    const long long base_seed = (argc >= 2) ? std::stoll(argv[1]) : 20260408LL;

    int total = 0;
    int passed = 0;
    double sum_bidiag_ms = 0.0;
    double sum_gkh_ms = 0.0;

    if (rank == 0)
    {
        std::cout << "MPI SVD 测试 (" << mpi_size << " 进程)" << std::endl;
        std::cout << std::endl;
    }

    // 样例1：5x5 固定值矩阵
    {
        Matrix A(5, 5);
        A.at(0, 0) = 4.0;  A.at(0, 1) = -1.0; A.at(0, 2) = 2.0;
        A.at(0, 3) = 0.5;  A.at(0, 4) = 3.0;
        A.at(1, 0) = 0.0;  A.at(1, 1) = 5.0;  A.at(1, 2) = -2.0;
        A.at(1, 3) = 1.0;  A.at(1, 4) = -1.5;
        A.at(2, 0) = 1.0;  A.at(2, 1) = 0.5;  A.at(2, 2) = 3.0;
        A.at(2, 3) = -4.0; A.at(2, 4) = 2.0;
        A.at(3, 0) = -2.0; A.at(3, 1) = 1.0;  A.at(3, 2) = 0.0;
        A.at(3, 3) = 6.0;  A.at(3, 4) = 1.0;
        A.at(4, 0) = 3.0;  A.at(4, 1) = -2.0; A.at(4, 2) = 1.0;
        A.at(4, 3) = 2.0;  A.at(4, 4) = 4.0;
        ++total;
        if (run_case("固定值 5x5", A, sum_bidiag_ms, sum_gkh_ms, rank))
            ++passed;
    }

    // 样例2：8x8 随机矩阵
    {
        Matrix A = Matrix::random(8, 8, -3.0, 3.0, base_seed + 1);
        ++total;
        if (run_case("随机 8x8", A, sum_bidiag_ms, sum_gkh_ms, rank))
            ++passed;
    }

    // 样例3：近秩亏损 10x8 矩阵
    {
        Matrix A = Matrix::random(10, 8, -2.0, 2.0, base_seed + 2);
        for (int i = 0; i < A.rows(); ++i)
            A.at(i, 2) = A.at(i, 0) + 1e-8 * (i + 1);
        ++total;
        if (run_case("近秩亏损 10x8", A, sum_bidiag_ms, sum_gkh_ms, rank))
            ++passed;
    }

    // 样例4：10x8 随机矩阵
    {
        Matrix A = Matrix::random(10, 8, -4.0, 4.0, base_seed + 3);
        ++total;
        if (run_case("随机 10x8", A, sum_bidiag_ms, sum_gkh_ms, rank))
            ++passed;
    }

    // 样例5：大规模 1000x1000 随机矩阵
    {
        Matrix A = Matrix::random(1000, 1000, -1.0, 1.0, base_seed + 4);
        ++total;
        if (run_case("随机 1000x1000", A, sum_bidiag_ms, sum_gkh_ms, rank))
            ++passed;
    }

    if (rank == 0)
    {
        std::cout << "==============================" << std::endl;
        std::cout << "MPI 进程数: " << mpi_size << std::endl;
        std::cout << "随机种子基值: " << base_seed << std::endl;
        std::cout << "总上二对角化耗时(ms): " << sum_bidiag_ms << std::endl;
        std::cout << "总GKH迭代耗时(ms): " << sum_gkh_ms << std::endl;
        std::cout << "通过: " << passed << " / " << total << std::endl;
    }

    MPI_Finalize();
    return (passed == total) ? 0 : 1;
}
