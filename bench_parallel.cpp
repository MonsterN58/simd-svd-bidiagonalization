#include "matrix.h"
#include "bidiagonalization.h"
#include "gkh.h"

#ifdef USE_OPENMP
#include "bidiag_omp.h"
#include "gkh_omp.h"
#endif

#ifdef USE_PTHREAD
#include "bidiag_pthread.h"
#include "gkh_pthread.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

// ---- 辅助函数 ----

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

static double reconstruction_error(const Matrix &A, const Matrix &U, const Matrix &S, const Matrix &V)
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

// ---- 多次运行取最优 ----

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

static RunResult run_once(const std::string &mode, const Matrix &A, int num_threads)
{
    using Clock = std::chrono::high_resolution_clock;

    Matrix U, V, B;
    bool converged;

    auto t1 = Clock::now();
    if (mode == "serial")
        B = to_bidiagonal(A, U, V);
#ifdef USE_OPENMP
    else if (mode == "omp")
        B = to_bidiagonal_omp(A, U, V, num_threads);
#endif
#ifdef USE_PTHREAD
    else if (mode == "pthread")
        B = to_bidiagonal_pthread(A, U, V, num_threads);
#endif
    else
        throw std::runtime_error("unknown mode: " + mode);
    auto t2 = Clock::now();

    auto t3 = Clock::now();
    if (mode == "serial")
        converged = gkh_svd_from_bidiagonal(U, B, V, 6000, 1e-12);
#ifdef USE_OPENMP
    else if (mode == "omp")
        converged = gkh_svd_from_bidiagonal_omp(U, B, V, 6000, 1e-12, num_threads);
#endif
#ifdef USE_PTHREAD
    else if (mode == "pthread")
        converged = gkh_svd_from_bidiagonal_pthread(U, B, V, 6000, 1e-12, num_threads);
#endif
    auto t4 = Clock::now();

    RunResult res;
    res.bidiag_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    res.gkh_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    res.total_ms = res.bidiag_ms + res.gkh_ms;

    double err_recon = reconstruction_error(A, U, B, V);
    res.recon_err_rel = err_recon / (fro_norm(A) + 1.0);
    res.orth_u_err = orth_error(U);
    res.orth_v_err = orth_error(V);
    res.converged = converged;

    return res;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    if (argc < 4)
    {
        std::cerr << "用法: " << argv[0] << " <mode:serial|omp|pthread> <size> <num_threads> [seed] [reps]\n";
        std::cerr << "  mode: serial, omp, pthread\n";
        std::cerr << "  size: 矩阵维度\n";
        std::cerr << "  num_threads: 线程数\n";
        std::cerr << "  seed: 随机种子 (默认20260408)\n";
        std::cerr << "  reps: 重复次数，取最优 (默认3)\n";
        return 1;
    }

    std::string mode = argv[1];
    int size = std::stoi(argv[2]);
    int num_threads = std::stoi(argv[3]);
    long long seed = (argc >= 5) ? std::stoll(argv[4]) : 20260408LL;
    int reps = (argc >= 6) ? std::stoi(argv[5]) : 3;

    if (mode != "serial"
#ifdef USE_OPENMP
        && mode != "omp"
#endif
#ifdef USE_PTHREAD
        && mode != "pthread"
#endif
    )
    {
        std::cerr << "未知模式: " << mode << "\n";
        return 1;
    }

    Matrix A = Matrix::random(size, size, -1.0, 1.0, seed);

    // 多次运行取最优
    RunResult best;
    best.total_ms = 1e30;

    for (int r = 0; r < reps; ++r)
    {
        RunResult res = run_once(mode, A, num_threads);
        if (res.total_ms < best.total_ms)
            best = res;
    }

    // 输出CSV
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "mode,size,threads,bidiag_ms,gkh_ms,total_ms,recon_err_rel,orth_u_err,orth_v_err,converged\n";
    std::cout << mode << ","
              << size << ","
              << num_threads << ","
              << best.bidiag_ms << ","
              << best.gkh_ms << ","
              << best.total_ms << ","
              << best.recon_err_rel << ","
              << best.orth_u_err << ","
              << best.orth_v_err << ","
              << (best.converged ? 1 : 0) << "\n";

    // 也可输出人类可读格式到stderr
    std::cerr << "\n=== " << mode << " " << size << "x" << size << " " << num_threads << "T ===\n";
    std::cerr << "  bidiag: " << best.bidiag_ms << " ms\n";
    std::cerr << "  gkh:    " << best.gkh_ms << " ms\n";
    std::cerr << "  total:  " << best.total_ms << " ms\n";
    std::cerr << "  recon:  " << best.recon_err_rel << "\n";
    std::cerr << "  orthU:  " << best.orth_u_err << "\n";
    std::cerr << "  orthV:  " << best.orth_v_err << "\n";
    std::cerr << "  conv:   " << (best.converged ? "yes" : "no") << "\n";

    return best.converged ? 0 : 1;
}
