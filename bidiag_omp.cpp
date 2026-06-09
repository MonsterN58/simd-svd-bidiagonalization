#include "bidiag_omp.h"
#include "matrix.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <omp.h>

#if !defined(DISABLE_MANUAL_SIMD) && (defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__))
#include <immintrin.h>
#endif

// ---- 辅助函数（与串行版相同） ----

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

// ---- OpenMP并行版 ----

Matrix to_bidiagonal_omp(const Matrix &A, Matrix &U, Matrix &V, int num_threads)
{
    if (A.rows() < A.cols())
        throw std::invalid_argument("to_bidiagonal_omp: requires m >= n");

    if (num_threads <= 0)
        num_threads = 1;

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤1: 左侧Householder变换
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

                // 计算 w = v^T * B[k:, k:]  (规约，并行化需要手动合并)
                std::vector<double> w(n - k, 0.0);
                // 并行计算局部w，然后合并
                {
                    int nt = num_threads;
                    std::vector<std::vector<double>> w_locals(nt, std::vector<double>(n - k, 0.0));

#pragma omp parallel for schedule(static) num_threads(num_threads)
                    for (int i = 0; i < m - k; ++i)
                        add_scaled_contiguous(w_locals[omp_get_thread_num()].data(),
                                              &B.at(k + i, k), n - k, v[i]);

                    for (int t = 0; t < nt; ++t)
                        for (int j = 0; j < n - k; ++j)
                            w[j] += w_locals[t][j];
                }

                // 并行更新 B 行
#pragma omp parallel for schedule(static) num_threads(num_threads)
                for (int i = 0; i < m - k; ++i)
                    add_scaled_contiguous(&B.at(k + i, k), w.data(), n - k, -beta * v[i]);

                // 并行计算 wU 并更新 U（每行独立，合并为一个循环）
#pragma omp parallel for schedule(static) num_threads(num_threads)
                for (int i = 0; i < m; ++i)
                {
                    double wU_i = dot_contiguous(&U.at(i, k), v.data(), m - k);
                    add_scaled_contiguous(&U.at(i, k), v.data(), m - k, -beta * wU_i);
                }
            }
        }

        // 清除第k列对角线以下元素
        for (int i = k + 1; i < m; ++i)
            B.at(i, k) = 0.0;

        // ================================================================
        // 步骤2: 右侧Householder变换
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

                    // 并行计算 w[i] 并更新 B 行（每行独立）
#pragma omp parallel for schedule(static) num_threads(num_threads)
                    for (int i = 0; i < m - k; ++i)
                    {
                        double w_i = dot_contiguous(&B.at(k + i, k + 1), v.data(), n - k - 1);
                        add_scaled_contiguous(&B.at(k + i, k + 1), v.data(), n - k - 1, -beta * w_i);
                    }

                    // 并行计算 wV 并更新 V
#pragma omp parallel for schedule(static) num_threads(num_threads)
                    for (int i = 0; i < n; ++i)
                    {
                        double wV_i = dot_contiguous(&V.at(i, k + 1), v.data(), n - k - 1);
                        add_scaled_contiguous(&V.at(i, k + 1), v.data(), n - k - 1, -beta * wV_i);
                    }
                }
            }

            for (int j = k + 2; j < n; ++j)
                B.at(k, j) = 0.0;
        }
    }

    return B;
}
