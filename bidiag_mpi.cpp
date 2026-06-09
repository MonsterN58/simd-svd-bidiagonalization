// bidiag_mpi.cpp
// MPI 并行版上双对角化
// 各进程按行分配参与 Householder 变换，使用 MPI_Allreduce 归约。

#include "bidiag_mpi.h"
#include "matrix.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>
#include <mpi.h>

#if !defined(DISABLE_MANUAL_SIMD) && (defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__))
#include <immintrin.h>
#endif

// ---- 辅助函数 ----

static double vector_norm(const std::vector<double> &v)
{
    double sum = 0.0;
    for (double x : v)
        sum += x * x;
    return std::sqrt(sum);
}

static double dot_contiguous(const double *a, const double *b, int len)
{
    int i = 0;
    double sum = 0.0;

#if !defined(DISABLE_MANUAL_SIMD) && defined(__AVX__)
    __m256d acc = _mm256_setzero_pd();
    for (; i + 3 < len; i += 4)
    {
        const __m256d va = _mm256_loadu_pd(a + i);
        const __m256d vb = _mm256_loadu_pd(b + i);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(va, vb));
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc);
    sum += tmp[0] + tmp[1] + tmp[2] + tmp[3];
#elif !defined(DISABLE_MANUAL_SIMD) && defined(__SSE2__)
    __m128d acc = _mm_setzero_pd();
    for (; i + 1 < len; i += 2)
    {
        const __m128d va = _mm_loadu_pd(a + i);
        const __m128d vb = _mm_loadu_pd(b + i);
        acc = _mm_add_pd(acc, _mm_mul_pd(va, vb));
    }
    alignas(16) double tmp[2];
    _mm_store_pd(tmp, acc);
    sum += tmp[0] + tmp[1];
#endif

    for (; i < len; ++i)
        sum += a[i] * b[i];
    return sum;
}

static void add_scaled_contiguous(double *dst, const double *src, int len, double alpha)
{
    int i = 0;

#if !defined(DISABLE_MANUAL_SIMD) && defined(__AVX__)
    const __m256d valpha = _mm256_set1_pd(alpha);
    for (; i + 3 < len; i += 4)
    {
        const __m256d vd = _mm256_loadu_pd(dst + i);
        const __m256d vs = _mm256_loadu_pd(src + i);
        _mm256_storeu_pd(dst + i, _mm256_add_pd(vd, _mm256_mul_pd(valpha, vs)));
    }
#elif !defined(DISABLE_MANUAL_SIMD) && defined(__SSE2__)
    const __m128d valpha = _mm_set1_pd(alpha);
    for (; i + 1 < len; i += 2)
    {
        const __m128d vd = _mm_loadu_pd(dst + i);
        const __m128d vs = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, _mm_add_pd(vd, _mm_mul_pd(valpha, vs)));
    }
#endif

    for (; i < len; ++i)
        dst[i] += alpha * src[i];
}

// ---- MPI 并行版上二对角化 ----

Matrix to_bidiagonal_mpi(const Matrix &A, Matrix &U, Matrix &V)
{
    int rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int m, n;
    if (rank == 0)
    {
        if (A.rows() < A.cols())
            throw std::invalid_argument("to_bidiagonal_mpi: requires m >= n");
        m = A.rows();
        n = A.cols();
    }

    // 广播矩阵尺寸
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 所有进程需要完整的 B, U, V（因为 Householder 更新涉及全矩阵）
    Matrix B(m, n);
    U = Matrix(m, m, 0.0);
    V = Matrix(n, n, 0.0);

    if (rank == 0)
    {
        B = A;
        for (int i = 0; i < m; ++i)
            U.at(i, i) = 1.0;
        for (int i = 0; i < n; ++i)
            V.at(i, i) = 1.0;
    }

    // 广播初始 B
    // 由于 Matrix 使用 std::vector<double>，需要直接传元素
    {
        std::vector<double> B_data(m * n);
        if (rank == 0)
        {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    B_data[i * n + j] = B.at(i, j);
        }
        MPI_Bcast(B_data.data(), m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0)
        {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    B.at(i, j) = B_data[i * n + j];
        }
    }

    // 初始化 U = I_m, V = I_n（所有进程）
    if (rank != 0)
    {
        for (int i = 0; i < m; ++i)
            U.at(i, i) = 1.0;
        for (int i = 0; i < n; ++i)
            V.at(i, i) = 1.0;
    }

    // Profiling
    double time_comm = 0.0;

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤1: 左侧 Householder 变换
        // ================================================================
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
            x[i] = B.at(k + i, k);

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;
            std::vector<double> v(x);
            v[0] += sigma;
            double vTv = dot_contiguous(v.data(), v.data(), static_cast<int>(v.size()));

            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 计算 w = v^T * B[k:, k:]
                // 每个进程负责一部分行，然后 Allreduce
                // 按行静态划分
                int total_rows = m - k;
                int rows_per_proc = total_rows / mpi_size;
                int remainder = total_rows % mpi_size;
                int my_start = rank * rows_per_proc + std::min(rank, remainder);
                int my_count = rows_per_proc + (rank < remainder ? 1 : 0);

                std::vector<double> w_local(n - k, 0.0);
                for (int i = my_start; i < my_start + my_count; ++i)
                    add_scaled_contiguous(w_local.data(), &B.at(k + i, k), n - k, v[i]);

                std::vector<double> w(n - k, 0.0);
                double t_comm_start = MPI_Wtime();
                MPI_Allreduce(w_local.data(), w.data(), n - k, MPI_DOUBLE,
                              MPI_SUM, MPI_COMM_WORLD);
                time_comm += MPI_Wtime() - t_comm_start;

                // 更新 B 行（每个进程更新自己负责的行）
                for (int i = my_start; i < my_start + my_count; ++i)
                    add_scaled_contiguous(&B.at(k + i, k), w.data(), n - k, -beta * v[i]);

                // 同步 B（Allgather 每个进程更新的行）
                // 为简单起见，使用 Allreduce 将所有进程的更新合并
                // 先将 B 的更新行收集
                {
                    // 构建差量：每个进程只更新了 my_start..my_start+my_count-1 行
                    // 使用 Allgatherv 收集所有进程更新的行
                    std::vector<double> B_rows(total_rows * (n - k), 0.0);
                    for (int i = my_start; i < my_start + my_count; ++i)
                        for (int j = 0; j < n - k; ++j)
                            B_rows[i * (n - k) + j] = B.at(k + i, k + j);

                    std::vector<double> B_rows_all(total_rows * (n - k), 0.0);
                    t_comm_start = MPI_Wtime();
                    MPI_Allreduce(B_rows.data(), B_rows_all.data(),
                                  total_rows * (n - k), MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    time_comm += MPI_Wtime() - t_comm_start;

                    for (int i = 0; i < total_rows; ++i)
                        for (int j = 0; j < n - k; ++j)
                            B.at(k + i, k + j) = B_rows_all[i * (n - k) + j];
                }

                // 更新 U：每个进程负责 U 的一部分行
                int total_U_rows = m;
                int U_rows_per_proc = total_U_rows / mpi_size;
                int U_remainder = total_U_rows % mpi_size;
                int U_my_start = rank * U_rows_per_proc + std::min(rank, U_remainder);
                int U_my_count = U_rows_per_proc + (rank < U_remainder ? 1 : 0);

                for (int i = U_my_start; i < U_my_start + U_my_count; ++i)
                {
                    double wU_i = dot_contiguous(&U.at(i, k), v.data(), m - k);
                    add_scaled_contiguous(&U.at(i, k), v.data(), m - k, -beta * wU_i);
                }

                // 同步 U 的更新行
                {
                    std::vector<double> U_rows(total_U_rows * (m - k), 0.0);
                    for (int i = U_my_start; i < U_my_start + U_my_count; ++i)
                        for (int j = 0; j < m - k; ++j)
                            U_rows[i * (m - k) + j] = U.at(i, k + j);

                    std::vector<double> U_rows_all(total_U_rows * (m - k), 0.0);
                    t_comm_start = MPI_Wtime();
                    MPI_Allreduce(U_rows.data(), U_rows_all.data(),
                                  total_U_rows * (m - k), MPI_DOUBLE,
                                  MPI_SUM, MPI_COMM_WORLD);
                    time_comm += MPI_Wtime() - t_comm_start;

                    for (int i = 0; i < total_U_rows; ++i)
                        for (int j = 0; j < m - k; ++j)
                            U.at(i, k + j) = U_rows_all[i * (m - k) + j];
                }
            }
        }

        // 清除第k列对角线以下元素
        for (int i = k + 1; i < m; ++i)
            B.at(i, k) = 0.0;

        // ================================================================
        // 步骤2: 右侧 Householder 变换
        // ================================================================
        if (k < n - 2)
        {
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
                y[j] = B.at(k, k + 1 + j);

            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;
                std::vector<double> v(y);
                v[0] += sigma;
                double vTv = dot_contiguous(v.data(), v.data(), static_cast<int>(v.size()));

                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 更新 B：每个进程负责一部分行
                    int total_rows = m - k;
                    int rows_per_proc = total_rows / mpi_size;
                    int remainder = total_rows % mpi_size;
                    int my_start = rank * rows_per_proc + std::min(rank, remainder);
                    int my_count = rows_per_proc + (rank < remainder ? 1 : 0);

                    for (int i = my_start; i < my_start + my_count; ++i)
                    {
                        double w_i = dot_contiguous(&B.at(k + i, k + 1), v.data(), n - k - 1);
                        add_scaled_contiguous(&B.at(k + i, k + 1), v.data(), n - k - 1, -beta * w_i);
                    }

                    // 同步 B 更新
                    {
                        std::vector<double> B_rows(total_rows * (n - k - 1), 0.0);
                        for (int i = my_start; i < my_start + my_count; ++i)
                            for (int j = 0; j < n - k - 1; ++j)
                                B_rows[i * (n - k - 1) + j] = B.at(k + i, k + 1 + j);

                        std::vector<double> B_rows_all(total_rows * (n - k - 1), 0.0);
                        double t_comm_start = MPI_Wtime();
                        MPI_Allreduce(B_rows.data(), B_rows_all.data(),
                                      total_rows * (n - k - 1), MPI_DOUBLE,
                                      MPI_SUM, MPI_COMM_WORLD);
                        time_comm += MPI_Wtime() - t_comm_start;

                        for (int i = 0; i < total_rows; ++i)
                            for (int j = 0; j < n - k - 1; ++j)
                                B.at(k + i, k + 1 + j) = B_rows_all[i * (n - k - 1) + j];
                    }

                    // 更新 V：每个进程负责一部分行
                    int total_V_rows = n;
                    int V_rows_per_proc = total_V_rows / mpi_size;
                    int V_remainder = total_V_rows % mpi_size;
                    int V_my_start = rank * V_rows_per_proc + std::min(rank, V_remainder);
                    int V_my_count = V_rows_per_proc + (rank < V_remainder ? 1 : 0);

                    for (int i = V_my_start; i < V_my_start + V_my_count; ++i)
                    {
                        double wV_i = dot_contiguous(&V.at(i, k + 1), v.data(), n - k - 1);
                        add_scaled_contiguous(&V.at(i, k + 1), v.data(), n - k - 1, -beta * wV_i);
                    }

                    // 同步 V 更新
                    {
                        std::vector<double> V_rows(total_V_rows * (n - k - 1), 0.0);
                        for (int i = V_my_start; i < V_my_start + V_my_count; ++i)
                            for (int j = 0; j < n - k - 1; ++j)
                                V_rows[i * (n - k - 1) + j] = V.at(i, k + 1 + j);

                        std::vector<double> V_rows_all(total_V_rows * (n - k - 1), 0.0);
                        double t_comm_start = MPI_Wtime();
                        MPI_Allreduce(V_rows.data(), V_rows_all.data(),
                                      total_V_rows * (n - k - 1), MPI_DOUBLE,
                                      MPI_SUM, MPI_COMM_WORLD);
                        time_comm += MPI_Wtime() - t_comm_start;

                        for (int i = 0; i < total_V_rows; ++i)
                            for (int j = 0; j < n - k - 1; ++j)
                                V.at(i, k + 1 + j) = V_rows_all[i * (n - k - 1) + j];
                    }
                }
            }

            // 强制置零
            for (int j = k + 2; j < n; ++j)
                B.at(k, j) = 0.0;
        }
    }

    // Profiling 输出
    std::cerr << "[MPI-Bidiag rank " << rank << "] comm=" << std::fixed
              << std::setprecision(2) << time_comm * 1000.0 << "ms" << std::endl;

    return B;
}
