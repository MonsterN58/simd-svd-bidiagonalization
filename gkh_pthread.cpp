// gkh_pthread.cpp
// Pthread并行版GKH迭代 - 方案1：子矩阵间并行 + 单块时行优先V/U并行 + 线程池

#include "gkh_pthread.h"
#include "givens.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <pthread.h>

#ifdef __linux__
#include <sched.h>
#endif

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

    struct RotP
    {
        double c, s;
        int c0, c1;
    };

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

// ---- 线程池 ----

struct alignas(64) PaddedDouble
{
    double value;
    char padding[64 - sizeof(double)];
};

enum WorkMode
{
    MODE_BLOCKS = 0, // 方案1：子矩阵间并行
    MODE_ROW_PARALLEL  // 单块：行优先V/U并行更新
};

struct SharedCtx
{
    Matrix *U, *B, *V;
    int num_threads;

    // 方案1：子矩阵间并行
    std::vector<Block> *blocks;
    std::atomic<int> next_block;

    // 单块：行优先V/U并行
    std::vector<RotP> *right_params;
    std::vector<RotP> *left_params;

    WorkMode mode;
    std::atomic<bool> stop;

    pthread_barrier_t start_barrier;
    pthread_barrier_t end_barrier;

    PaddedDouble *thread_work_ms;
};

struct ThreadArg
{
    int tid;
    SharedCtx *ctx;
};

static void set_cpu_affinity(int cpu_id)
{
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#else
    (void)cpu_id;
#endif
}

static void *gkh_worker(void *arg)
{
    ThreadArg *targ = (ThreadArg *)arg;
    SharedCtx *ctx = targ->ctx;
    const int tid = targ->tid;
    const int nthreads = ctx->num_threads;

    set_cpu_affinity(tid);

    // 预计算行范围（单块模式用）
    // 每次迭代前V/U的行数不变，但为安全起见在循环内计算
    int V_i0 = 0, V_i1 = 0, U_i0 = 0, U_i1 = 0;

    while (true)
    {
        // 等待主线程发信号
        pthread_barrier_wait(&ctx->start_barrier);

        if (ctx->stop.load(std::memory_order_acquire))
            break;

        using Clock = std::chrono::high_resolution_clock;
        auto work_start = Clock::now();

        if (ctx->mode == MODE_BLOCKS)
        {
            // 方案1：原子计数器领取block
            while (true)
            {
                int idx = ctx->next_block.fetch_add(1, std::memory_order_relaxed);
                if (idx >= (int)ctx->blocks->size())
                    break;
                const Block &blk = (*ctx->blocks)[idx];
                if (blk.r > blk.l)
                    one_block_step(*ctx->U, *ctx->B, *ctx->V, blk.l, blk.r);
            }
        }
        else
        {
            // 单块模式：行优先V/U并行更新
            const int V_rows = ctx->V->rows();
            const int V_chunk = (V_rows + nthreads - 1) / nthreads;
            V_i0 = std::min(tid * V_chunk, V_rows);
            V_i1 = std::min(V_i0 + V_chunk, V_rows);

            const int U_rows = ctx->U->rows();
            const int U_chunk = (U_rows + nthreads - 1) / nthreads;
            U_i0 = std::min(tid * U_chunk, U_rows);
            U_i1 = std::min(U_i0 + U_chunk, U_rows);

            // 更新V的行[V_i0, V_i1)
            for (int row = V_i0; row < V_i1; ++row)
            {
                for (const auto &p : *ctx->right_params)
                {
                    double a = ctx->V->at(row, p.c0);
                    double b = ctx->V->at(row, p.c1);
                    ctx->V->at(row, p.c0) = a * p.c - b * p.s;
                    ctx->V->at(row, p.c1) = a * p.s + b * p.c;
                }
            }

            // 更新U的行[U_i0, U_i1)
            for (int row = U_i0; row < U_i1; ++row)
            {
                for (const auto &p : *ctx->left_params)
                {
                    double a = ctx->U->at(row, p.c0);
                    double b = ctx->U->at(row, p.c1);
                    ctx->U->at(row, p.c0) = a * p.c - b * p.s;
                    ctx->U->at(row, p.c1) = a * p.s + b * p.c;
                }
            }
        }

        auto work_end = Clock::now();
        ctx->thread_work_ms[tid].value +=
            std::chrono::duration<double, std::milli>(work_end - work_start).count();

        // 通知主线程完成
        pthread_barrier_wait(&ctx->end_barrier);
    }

    return nullptr;
}

// ---- Pthread并行版主函数 ----

bool gkh_svd_from_bidiagonal_pthread(Matrix &U, Matrix &B, Matrix &V,
                                      int max_iter, double tol, int num_threads)
{
    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
        throw std::invalid_argument("gkh_svd_from_bidiagonal_pthread: requires m >= n");
    if (U.rows() != m || U.cols() != m)
        throw std::invalid_argument("gkh_svd_from_bidiagonal_pthread: U must be m x m");
    if (V.rows() != n || V.cols() != n)
        throw std::invalid_argument("gkh_svd_from_bidiagonal_pthread: V must be n x n");

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

    SharedCtx ctx;
    ctx.U = &U;
    ctx.B = &B;
    ctx.V = &V;
    ctx.num_threads = num_threads;
    ctx.stop.store(false, std::memory_order_relaxed);

    PaddedDouble *thread_work_ms = new PaddedDouble[num_threads]();
    ctx.thread_work_ms = thread_work_ms;

    pthread_barrier_init(&ctx.start_barrier, nullptr, num_threads);
    pthread_barrier_init(&ctx.end_barrier, nullptr, num_threads);

    // 创建工作线程
    std::vector<pthread_t> threads(num_threads - 1);
    std::vector<ThreadArg> args(num_threads - 1);
    for (int i = 0; i < num_threads - 1; ++i)
    {
        args[i].tid = i + 1;
        args[i].ctx = &ctx;
        pthread_create(&threads[i], nullptr, gkh_worker, &args[i]);
    }

    set_cpu_affinity(0);

    bool converged = false;
    std::vector<Block> blocks;
    std::vector<RotP> right_params, left_params;

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // 串行预处理
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        blocks = split_active_blocks(B, n, tol);

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
            // 方案1：子矩阵间并行
            ctx.mode = MODE_BLOCKS;
            ctx.blocks = &blocks;
            ctx.next_block.store(0, std::memory_order_relaxed);

            // 通知工作线程开始
            pthread_barrier_wait(&ctx.start_barrier);

            // 主线程也参与处理block
            using Clock = std::chrono::high_resolution_clock;
            auto main_work_start = Clock::now();

            while (true)
            {
                int idx = ctx.next_block.fetch_add(1, std::memory_order_relaxed);
                if (idx >= (int)blocks.size())
                    break;
                const Block &blk = blocks[idx];
                if (blk.r > blk.l)
                    one_block_step(U, B, V, blk.l, blk.r);
            }

            auto main_work_end = Clock::now();
            thread_work_ms[0].value +=
                std::chrono::duration<double, std::milli>(main_work_end - main_work_start).count();

            // 等待所有线程完成
            pthread_barrier_wait(&ctx.end_barrier);
        }
        else
        {
            // 单块：串行追赶B，收集旋转参数，行优先并行更新V/U
            right_params.clear();
            left_params.clear();
            for (const auto &blk : blocks)
            {
                if (blk.r > blk.l)
                    one_block_step_collect(B, blk.l, blk.r, right_params, left_params);
            }

            ctx.mode = MODE_ROW_PARALLEL;
            ctx.right_params = &right_params;
            ctx.left_params = &left_params;

            // 通知工作线程开始
            pthread_barrier_wait(&ctx.start_barrier);

            // 主线程也参与行优先V/U更新
            using Clock = std::chrono::high_resolution_clock;
            auto main_work_start = Clock::now();

            const int V_rows = V.rows();
            const int V_chunk = (V_rows + num_threads - 1) / num_threads;
            const int V_i0 = 0;
            const int V_i1 = std::min(V_chunk, V_rows);

            const int U_rows = U.rows();
            const int U_chunk = (U_rows + num_threads - 1) / num_threads;
            const int U_i0 = 0;
            const int U_i1 = std::min(U_chunk, U_rows);

            for (int row = V_i0; row < V_i1; ++row)
            {
                for (const auto &p : right_params)
                {
                    double a = V.at(row, p.c0);
                    double b = V.at(row, p.c1);
                    V.at(row, p.c0) = a * p.c - b * p.s;
                    V.at(row, p.c1) = a * p.s + b * p.c;
                }
            }

            for (int row = U_i0; row < U_i1; ++row)
            {
                for (const auto &p : left_params)
                {
                    double a = U.at(row, p.c0);
                    double b = U.at(row, p.c1);
                    U.at(row, p.c0) = a * p.c - b * p.s;
                    U.at(row, p.c1) = a * p.s + b * p.c;
                }
            }

            auto main_work_end = Clock::now();
            thread_work_ms[0].value +=
                std::chrono::duration<double, std::milli>(main_work_end - main_work_start).count();

            // 等待所有线程完成
            pthread_barrier_wait(&ctx.end_barrier);
        }
    }

    // 通知工作线程退出
    ctx.stop.store(true, std::memory_order_release);
    pthread_barrier_wait(&ctx.start_barrier);

    // 等待工作线程结束
    for (int i = 0; i < num_threads - 1; ++i)
        pthread_join(threads[i], nullptr);

    // 输出per-thread负载统计
    {
        double max_work = 0.0, min_work = 1e30;
        std::cerr << "[GKH-Pthread] per-thread work time (ms):";
        for (int i = 0; i < num_threads; ++i)
        {
            std::cerr << " T" << i << "=" << std::fixed << std::setprecision(2) << thread_work_ms[i].value;
            if (thread_work_ms[i].value > max_work)
                max_work = thread_work_ms[i].value;
            if (thread_work_ms[i].value < min_work)
                min_work = thread_work_ms[i].value;
        }
        double imbalance = (max_work > 0) ? (max_work - min_work) / max_work * 100.0 : 0.0;
        std::cerr << " | imbalance=" << imbalance << "%\n";
    }

    delete[] thread_work_ms;
    pthread_barrier_destroy(&ctx.start_barrier);
    pthread_barrier_destroy(&ctx.end_barrier);

    cleanup_bidiagonal(B, tol);
    for (int i = 0; i < n - 1; ++i)
        B.at(i, i + 1) = 0.0;
    make_nonnegative_and_sort(U, B, V);

    return converged;
}
