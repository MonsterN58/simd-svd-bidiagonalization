// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
//
// 算法说明（你需要结合代码看）：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步（k = 0, 1, ..., n-1）：
//    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
//    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
//
// 例如，对一个 4x4 矩阵 A，第一步 k=0：
//   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
//   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
//
// 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
//
// 本组件输出：A = U * B * V^T
// 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵

#include "matrix.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#if defined(ENABLE_BIDIAG_PROFILE)
#include <chrono>
#include <iostream>
#endif

#if !defined(DISABLE_MANUAL_SIMD) && (defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__))
#include <immintrin.h>
#endif

#if defined(ENABLE_BIDIAG_PROFILE)
namespace
{
    struct BidiagProfileData
    {
        long long dot_calls = 0;
        long long add_scaled_calls = 0;
        double dot_ms = 0.0;
        double add_scaled_ms = 0.0;

        ~BidiagProfileData()
        {
            const double total = dot_ms + add_scaled_ms;
            std::cerr << "[bidiag-profile] dot_calls=" << dot_calls
                      << " dot_ms=" << dot_ms
                      << " add_scaled_calls=" << add_scaled_calls
                      << " add_scaled_ms=" << add_scaled_ms
                      << " measured_kernel_ms=" << total << '\n';
        }
    };

    static BidiagProfileData g_bidiag_profile;
}
#endif

// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
{
    double sum = 0.0;
    for (double x : v)
        sum += x * x;
    return std::sqrt(sum);
}

// 连续内存向量点积。Matrix 按行连续存储，因此 Householder 中按行访问的部分可直接向量化。
static double dot_contiguous(const double *a, const double *b, int len)
{
#if defined(ENABLE_BIDIAG_PROFILE)
    const auto profile_begin = std::chrono::steady_clock::now();
#endif
    int i = 0;
    double sum = 0.0;

#if !defined(DISABLE_MANUAL_SIMD) && defined(FORCE_AVX_UNROLL8) && defined(__AVX__)
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    for (; i + 7 < len; i += 8)
    {
        const __m256d va0 = _mm256_loadu_pd(a + i);
        const __m256d vb0 = _mm256_loadu_pd(b + i);
        const __m256d va1 = _mm256_loadu_pd(a + i + 4);
        const __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(va0, vb0));
        acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(va1, vb1));
    }

    alignas(32) double tmp0[4];
    alignas(32) double tmp1[4];
    _mm256_store_pd(tmp0, acc0);
    _mm256_store_pd(tmp1, acc1);
    sum += tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3];
#elif !defined(DISABLE_MANUAL_SIMD) && defined(__AVX512F__)
    __m512d acc = _mm512_setzero_pd();
    for (; i + 7 < len; i += 8)
    {
        const __m512d va = _mm512_loadu_pd(a + i);
        const __m512d vb = _mm512_loadu_pd(b + i);
        acc = _mm512_add_pd(acc, _mm512_mul_pd(va, vb));
    }

    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, acc);
    sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif !defined(DISABLE_MANUAL_SIMD) && defined(__AVX__)
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
    {
        sum += a[i] * b[i];
    }
#if defined(ENABLE_BIDIAG_PROFILE)
    const auto profile_end = std::chrono::steady_clock::now();
    g_bidiag_profile.dot_ms += std::chrono::duration<double, std::milli>(profile_end - profile_begin).count();
    ++g_bidiag_profile.dot_calls;
#endif
    return sum;
}

// dst += alpha * src，连续内存版本。
static void add_scaled_contiguous(double *dst, const double *src, int len, double alpha)
{
#if defined(ENABLE_BIDIAG_PROFILE)
    const auto profile_begin = std::chrono::steady_clock::now();
#endif
    int i = 0;

#if !defined(DISABLE_MANUAL_SIMD) && defined(FORCE_AVX_UNROLL8) && defined(__AVX__)
    const __m256d valpha = _mm256_set1_pd(alpha);
    for (; i + 7 < len; i += 8)
    {
        const __m256d vd0 = _mm256_loadu_pd(dst + i);
        const __m256d vs0 = _mm256_loadu_pd(src + i);
        const __m256d vd1 = _mm256_loadu_pd(dst + i + 4);
        const __m256d vs1 = _mm256_loadu_pd(src + i + 4);
        _mm256_storeu_pd(dst + i, _mm256_add_pd(vd0, _mm256_mul_pd(valpha, vs0)));
        _mm256_storeu_pd(dst + i + 4, _mm256_add_pd(vd1, _mm256_mul_pd(valpha, vs1)));
    }
#elif !defined(DISABLE_MANUAL_SIMD) && defined(__AVX512F__)
    const __m512d valpha = _mm512_set1_pd(alpha);
    for (; i + 7 < len; i += 8)
    {
        const __m512d vd = _mm512_loadu_pd(dst + i);
        const __m512d vs = _mm512_loadu_pd(src + i);
        _mm512_storeu_pd(dst + i, _mm512_add_pd(vd, _mm512_mul_pd(valpha, vs)));
    }
#elif !defined(DISABLE_MANUAL_SIMD) && defined(__AVX__)
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
    {
        dst[i] += alpha * src[i];
    }
#if defined(ENABLE_BIDIAG_PROFILE)
    const auto profile_end = std::chrono::steady_clock::now();
    g_bidiag_profile.add_scaled_ms += std::chrono::duration<double, std::milli>(profile_end - profile_begin).count();
    ++g_bidiag_profile.add_scaled_calls;
#endif
}

// 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
    if (A.rows() < A.cols())
    {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B = A;

    // U = I_m，V = I_n
    U = Matrix(m, m, 0.0);
    for (int i = 0; i < m; ++i)
        U.at(i, i) = 1.0;
    V = Matrix(n, n, 0.0);
    for (int i = 0; i < n; ++i)
        V.at(i, i) = 1.0;

    for (int k = 0; k < n; ++k)
    {
        // ================================================================
        // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
        // ================================================================

        // 提取第 k 列从第 k 行往下的子向量
        // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
        {
            x[i] = B.at(k + i, k);
        }

        double norm_x = vector_norm(x);

        if (norm_x > 1e-14 && k < m - 1)
        {
            // sign(x[0])：此处规定 x[0]==0 时取 +1
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

            // 实际上这里是+或者-都可以，手册里 Householder 一节是 -αe_1
            // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
            std::vector<double> v(x);
            v[0] += sigma; // v = x + sigma * e_1

            // 计算 v^T v
            double vTv = dot_contiguous(v.data(), v.data(), static_cast<int>(v.size()));

            // SIMD 加速 Householder：按行连续访问 B/U，向量化点积和 saxpy 更新。
            if (vTv > 1e-28)
            {
                const double beta = 2.0 / vTv;

                // 手册里的 Householder 矩阵定义为 H = I - beta * v * v^T，其中 beta = 2 / (v^T v)
                // 从左侧作用 H：B_new = H * B_old = B_old - beta * v * (v^T * B_old)
                std::vector<double> w(n - k, 0.0);
                for (int i = 0; i < m - k; ++i)
                    add_scaled_contiguous(w.data(), &B.at(k + i, k), n - k, v[i]);
                for (int i = 0; i < m - k; ++i)
                    add_scaled_contiguous(&B.at(k + i, k), w.data(), n - k, -beta * v[i]);

                // 累积 U：U_new = U_old * H_k
                // U[:, k:m] -= beta * (U[:, k:m] * v) * v^T
                std::vector<double> wU(m, 0.0);
                for (int i = 0; i < m; ++i)
                    wU[i] = dot_contiguous(&U.at(i, k), v.data(), m - k);
                for (int i = 0; i < m; ++i)
                    add_scaled_contiguous(&U.at(i, k), v.data(), m - k, -beta * wU[i]);
            }
        }

        // 清除第 k 列中对角线以下的元素
        // 理论上应为 0，但不能完全保证全是 0，这里强制置零
        for (int i = k + 1; i < m; ++i)
        {
            B.at(i, k) = 0.0;
        }

        // ================================================================
        // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
        //        （只在 k < n-2 时需要）
        // ================================================================

        if (k < n - 2)
        {
            // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
            {
                y[j] = B.at(k, k + 1 + j);
            }

            // 与之前类似，计算模长
            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

                // 构造 Householder 向量 v = y + sigma * e_1
                std::vector<double> v(y);
                v[0] += sigma;

                double vTv = dot_contiguous(v.data(), v.data(), static_cast<int>(v.size()));

                // SIMD 加速 Householder：按行连续访问 B/V，向量化点积和 saxpy 更新。
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;

                    // 注意：这里是从右侧作用 V_k
                    // B_new = B_old * V_k = B_old - beta * (B_old * v) * v^T
                    std::vector<double> w(m - k, 0.0);
                    for (int i = 0; i < m - k; ++i)
                        w[i] = dot_contiguous(&B.at(k + i, k + 1), v.data(), n - k - 1);
                    for (int i = 0; i < m - k; ++i)
                        add_scaled_contiguous(&B.at(k + i, k + 1), v.data(), n - k - 1, -beta * w[i]);

                    // 累积 V：V_new = V_old * V_k
                    // V[:, k+1:n] -= beta * (V[:, k+1:n] * v) * v^T
                    std::vector<double> wV(n, 0.0);
                    for (int i = 0; i < n; ++i)
                        wV[i] = dot_contiguous(&V.at(i, k + 1), v.data(), n - k - 1);
                    for (int i = 0; i < n; ++i)
                        add_scaled_contiguous(&V.at(i, k + 1), v.data(), n - k - 1, -beta * wV[i]);
                }
            }

            // 强制置零
            for (int j = k + 2; j < n; ++j)
            {
                B.at(k, j) = 0.0;
            }
        }
    }

    return B;
}
