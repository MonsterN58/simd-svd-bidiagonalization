#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include <immintrin.h>

static double dot_aligned(const double *a, const double *b, int n)
{
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 3 < n; i += 4)
    {
        acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_load_pd(a + i), _mm256_load_pd(b + i)));
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc);
    double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

static double dot_unaligned(const double *a, const double *b, int n)
{
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 3 < n; i += 4)
    {
        acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
    }
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, acc);
    double s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

int main()
{
    const std::vector<int> sizes = {1024, 4096, 16384, 65536};
    std::cout << "N,aligned_ms,unaligned_ms\n";
    for (int n : sizes)
    {
        std::vector<double> raw_a(n + 8), raw_b(n + 8);
        uintptr_t pa = reinterpret_cast<uintptr_t>(raw_a.data());
        uintptr_t pb = reinterpret_cast<uintptr_t>(raw_b.data());
        double *a = reinterpret_cast<double *>((pa + 31u) & ~uintptr_t(31u));
        double *b = reinterpret_cast<double *>((pb + 31u) & ~uintptr_t(31u));
        for (int i = 0; i < n + 4; ++i)
        {
            a[i] = 1.0 / (i + 1);
            b[i] = 0.5 + 1.0 / (i + 3);
        }

        volatile double guard = 0.0;
        const int reps = 20000000 / n;
        auto bench = [&](bool aligned) {
            std::vector<double> times;
            for (int r = 0; r < 7; ++r)
            {
                const auto beg = std::chrono::high_resolution_clock::now();
                for (int t = 0; t < reps; ++t)
                {
                    guard += aligned ? dot_aligned(a, b, n) : dot_unaligned(a + 1, b + 1, n);
                }
                const auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(end - beg).count());
            }
            std::sort(times.begin(), times.end());
            return times[times.size() / 2];
        };
        std::cout << n << ',' << bench(true) << ',' << bench(false) << '\n';
        if (guard == 0.123)
            std::cerr << guard << '\n';
    }
    return 0;
}
