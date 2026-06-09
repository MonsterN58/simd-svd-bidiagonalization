#include "gkh_omp.h"
#include "givens.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>

namespace
{

    struct Block
    {
        int l;
        int r;
    };

    // ---- 与串行版完全相同的辅助函数 ----

    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        for (int j = 0; j < M.cols(); ++j)
        {
            double a = M.at(r0, j);
            double b = M.at(r1, j);
            M.at(r0, j) = c * a + s * b;
            M.at(r1, j) = -s * a + c * b;
        }
    }

    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
        for (int i = 0; i < M.rows(); ++i)
        {
            double a = M.at(i, c0);
            double b = M.at(i, c1);
            M.at(i, c0) = a * c - b * s;
            M.at(i, c1) = a * s + b * c;
        }
    }

    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        apply_right_cols(U, r0, r1, c, -s);
    }

    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
            return B.at(l, l) * B.at(l, l);

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
            disc = 0.0;

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
            for (int j = 0; j < B.cols(); ++j)
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                    B.at(i, j) = 0.0;
    }

    // 串行版one_block_step（多block时直接调用）
    static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    {
        if (r <= l)
            return;

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0, s = 0.0, rr = 0.0;

        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols(B, l, l + 1, c, s);
        apply_right_cols(V, l, l + 1, c, s);

        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows(B, l, l + 1, c, s);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols(B, k, k + 1, c, s);
            apply_right_cols(V, k, k + 1, c, s);

            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows(B, k, k + 1, c, s);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    // ---- 单块优化：收集旋转参数，延迟V/U更新 ----

    struct RotP
    {
        double c, s;
        int c0, c1;
    };

    // 只更新B，收集V/U旋转参数
    static void one_block_step_collect(Matrix &B, int l, int r,
                                        std::vector<RotP> &right_params,
                                        std::vector<RotP> &left_params)
    {
        if (r <= l)
            return;

        const double mu = block_wilkinson_shift(B, l, r);

        double c, s, rr;

        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols(B, l, l + 1, c, s);
        right_params.push_back({c, s, l, l + 1});

        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows(B, l, l + 1, c, s);
        left_params.push_back({c, -s, l, l + 1});

        for (int k = l + 1; k <= r - 1; ++k)
        {
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols(B, k, k + 1, c, s);
            right_params.push_back({c, s, k, k + 1});

            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows(B, k, k + 1, c, s);
            left_params.push_back({c, -s, k, k + 1});
        }
    }

    static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
            return false;
        if (std::fabs(B.at(k, k + 1)) <= tol)
            return false;

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0, s = 0.0, rr = 0.0;
            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols(B, i, i + 1, c, s);
            apply_right_cols(V, i, i + 1, c, s);

            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows(B, i, i + 1, c, s);
                accumulate_left_into_U(U, i, i + 1, c, s);
            }
            changed = true;
        }
        cleanup_bidiagonal(B, tol);
        return changed;
    }

    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int n = B.cols();
        bool changed = false;
        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal(U, B, V, k, tol))
                    changed = true;
            }
        }
        return changed;
    }

    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
                B.at(k, k + 1) = 0.0;
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
                ++r;
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                    U.at(r, i) = -U.at(r, i);
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
            idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U, V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);
            for (int r = 0; r < U.rows(); ++r)
                U2.at(r, new_i) = U.at(r, old_i);
            for (int r = 0; r < V.rows(); ++r)
                V2.at(r, new_i) = V.at(r, old_i);
        }

        U = U2;
        V = V2;
        B = D;
    }

} // namespace

// ---- OpenMP并行版主函数 ----

bool gkh_svd_from_bidiagonal_omp(Matrix &U, Matrix &B, Matrix &V,
                                  int max_iter, double tol, int num_threads)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
        throw std::invalid_argument("gkh_svd_from_bidiagonal_omp: requires m >= n");
    if (U.rows() != m || U.cols() != m)
        throw std::invalid_argument("gkh_svd_from_bidiagonal_omp: U must be m x m");
    if (V.rows() != n || V.cols() != n)
        throw std::invalid_argument("gkh_svd_from_bidiagonal_omp: V must be n x n");

    if (num_threads <= 0)
        num_threads = 1;

    // 线程数=1时直接串行
    if (num_threads == 1)
    {
        bool converged = false;
        for (int iter = 0; iter < max_iter; ++iter)
        {
            cleanup_bidiagonal(B, tol);
            handle_diagonal_zeros(U, B, V, tol);
            auto blocks = split_active_blocks(B, n, tol);

            bool all_singletons = true;
            for (const auto &blk : blocks)
                if (blk.r > blk.l)
                {
                    all_singletons = false;
                    break;
                }
            if (all_singletons)
            {
                converged = true;
                break;
            }

            for (const auto &blk : blocks)
                if (blk.r > blk.l)
                    one_block_step(U, B, V, blk.l, blk.r);
        }

        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
            B.at(i, i + 1) = 0.0;
        make_nonnegative_and_sort(U, B, V);
        return converged;
    }

    // ---- 多线程版本 ----

    bool converged = false;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        std::vector<Block> blocks = split_active_blocks(B, n, tol);

        bool all_singletons = true;
        for (const auto &blk : blocks)
            if (blk.r > blk.l)
            {
                all_singletons = false;
                break;
            }

        if (all_singletons)
        {
            converged = true;
            break;
        }

        // 统计非平凡块数
        int num_active = 0;
        for (const auto &blk : blocks)
            if (blk.r > blk.l)
                ++num_active;

        if (num_active >= 2)
        {
            // 方案1：多个子矩阵间并行（子矩阵天然独立，无需锁）
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
            for (int i = 0; i < (int)blocks.size(); ++i)
            {
                if (blocks[i].r > blocks[i].l)
                {
                    one_block_step(U, B, V, blocks[i].l, blocks[i].r);
                }
            }
        }
        else
        {
            // 只有1个块：先对B做追赶（串行），收集旋转参数，
            // 然后行优先并行更新V和U
            std::vector<RotP> right_params, left_params;
            for (const auto &blk : blocks)
            {
                if (blk.r > blk.l)
                    one_block_step_collect(B, blk.l, blk.r, right_params, left_params);
            }

            // 行优先并行更新V
#pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int row = 0; row < V.rows(); ++row)
            {
                for (const auto &p : right_params)
                {
                    double a = V.at(row, p.c0);
                    double b = V.at(row, p.c1);
                    V.at(row, p.c0) = a * p.c - b * p.s;
                    V.at(row, p.c1) = a * p.s + b * p.c;
                }
            }

            // 行优先并行更新U
#pragma omp parallel for schedule(static) num_threads(num_threads)
            for (int row = 0; row < U.rows(); ++row)
            {
                for (const auto &p : left_params)
                {
                    double a = U.at(row, p.c0);
                    double b = U.at(row, p.c1);
                    U.at(row, p.c0) = a * p.c - b * p.s;
                    U.at(row, p.c1) = a * p.s + b * p.c;
                }
            }
        }
    }

    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i)
        B.at(i, i + 1) = 0.0;
    make_nonnegative_and_sort(U, B, V);

    return converged;
}
