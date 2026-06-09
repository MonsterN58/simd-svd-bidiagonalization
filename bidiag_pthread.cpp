#include "bidiag_pthread.h"
#include "matrix.h"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <pthread.h>

#if !defined(DISABLE_MANUAL_SIMD) && (defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__))
#include <immintrin.h>
#endif

#ifdef __linux__
#include <sched.h>
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
        _mm_store_pd(dst + i, _mm_add_pd(vd, _mm_mul_pd(valpha, vs)));
    }
#endif

    for (; i < len; ++i)
        dst[i] += alpha * src[i];
}

// ---- CPU亲和性 ----

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

// ---- Pthread线程池 ----

enum BidiagPhase
{
    PHASE_EXIT = -1,
    PHASE_IDLE = 0,
    PHASE_LEFT_UPDATE_B = 1,
    PHASE_LEFT_UPDATE_U = 2,
    PHASE_RIGHT_UPDATE_BV = 3,
};

struct BidiagSharedCtx
{
    Matrix *B;
    Matrix *U;
    Matrix *V;

    int k;
    int m, n;
    double beta;
    const std::vector<double> *v;
    const std::vector<double> *w;

    std::vector<std::vector<double>> *w_locals;

    volatile BidiagPhase phase;

    pthread_barrier_t barrier;
    int num_threads;
};

struct BidiagThreadArg
{
    int tid;
    BidiagSharedCtx *ctx;
};

static void row_range(int tid, int num_threads, int total, int &start, int &end)
{
    start = tid * total / num_threads;
    end = (tid + 1) * total / num_threads;
}

static void *bidiag_worker(void *arg)
{
    BidiagThreadArg *targ = (BidiagThreadArg *)arg;
    const int tid = targ->tid;
    BidiagSharedCtx *ctx = targ->ctx;

    // 设置CPU亲和性
    set_cpu_affinity(tid);

    while (true)
    {
        pthread_barrier_wait(&ctx->barrier);

        if (ctx->phase == PHASE_EXIT)
            break;

        switch (ctx->phase)
        {
        case PHASE_LEFT_UPDATE_B:
        {
            const int total = ctx->m - ctx->k;
            int start, end;
            row_range(tid, ctx->num_threads, total, start, end);

            // 更新B行
            for (int i = start; i < end; ++i)
                add_scaled_contiguous(&ctx->B->at(ctx->k + i, ctx->k),
                                      ctx->w->data(), ctx->n - ctx->k,
                                      -ctx->beta * (*ctx->v)[i]);
            break;
        }

        case PHASE_LEFT_UPDATE_U:
        {
            const int total = ctx->m;
            int start, end;
            row_range(tid, ctx->num_threads, total, start, end);

            for (int i = start; i < end; ++i)
            {
                double wU_i = dot_contiguous(&ctx->U->at(i, ctx->k),
                                             ctx->v->data(), ctx->m - ctx->k);
                add_scaled_contiguous(&ctx->U->at(i, ctx->k),
                                      ctx->v->data(), ctx->m - ctx->k,
                                      -ctx->beta * wU_i);
            }
            break;
        }

        case PHASE_RIGHT_UPDATE_BV:
        {
            // B行部分
            {
                const int total = ctx->m - ctx->k;
                int start, end;
                row_range(tid, ctx->num_threads, total, start, end);

                for (int i = start; i < end; ++i)
                {
                    double w_i = dot_contiguous(&ctx->B->at(ctx->k + i, ctx->k + 1),
                                                ctx->v->data(), ctx->n - ctx->k - 1);
                    add_scaled_contiguous(&ctx->B->at(ctx->k + i, ctx->k + 1),
                                          ctx->v->data(), ctx->n - ctx->k - 1,
                                          -ctx->beta * w_i);
                }
            }
            // V行部分
            {
                const int total = ctx->n;
                int start, end;
                row_range(tid, ctx->num_threads, total, start, end);

                for (int i = start; i < end; ++i)
                {
                    double wV_i = dot_contiguous(&ctx->V->at(i, ctx->k + 1),
                                                 ctx->v->data(), ctx->n - ctx->k - 1);
                    add_scaled_contiguous(&ctx->V->at(i, ctx->k + 1),
                                          ctx->v->data(), ctx->n - ctx->k - 1,
                                          -ctx->beta * wV_i);
                }
            }
            break;
        }

        default:
            break;
        }

        pthread_barrier_wait(&ctx->barrier);
    }

    return nullptr;
}

// ---- Pthread并行版主函数 ----

Matrix to_bidiagonal_pthread(const Matrix &A, Matrix &U, Matrix &V, int num_threads)
{
    if (A.rows() < A.cols())
        throw std::invalid_argument("to_bidiagonal_pthread: requires m >= n");

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

    // 线程数=1时直接串行执行
    if (num_threads == 1)
    {
        for (int k = 0; k < n; ++k)
        {
            std::vector<double> x(m - k);
            for (int i = 0; i < m - k; ++i)
                x[i] = B.at(k + i, k);
            double norm_x = vector_norm(x);
            if (norm_x > 1e-14 && k < m - 1)
            {
                double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;
                std::vector<double> v(x);
                v[0] += sigma;
                double vTv = dot_contiguous(v.data(), v.data(), (int)v.size());
                if (vTv > 1e-28)
                {
                    const double beta = 2.0 / vTv;
                    std::vector<double> w(n - k, 0.0);
                    for (int i = 0; i < m - k; ++i)
                        add_scaled_contiguous(w.data(), &B.at(k + i, k), n - k, v[i]);
                    for (int i = 0; i < m - k; ++i)
                        add_scaled_contiguous(&B.at(k + i, k), w.data(), n - k, -beta * v[i]);
                    std::vector<double> wU(m, 0.0);
                    for (int i = 0; i < m; ++i)
                        wU[i] = dot_contiguous(&U.at(i, k), v.data(), m - k);
                    for (int i = 0; i < m; ++i)
                        add_scaled_contiguous(&U.at(i, k), v.data(), m - k, -beta * wU[i]);
                }
            }
            for (int i = k + 1; i < m; ++i)
                B.at(i, k) = 0.0;

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
                    double vTv = dot_contiguous(v.data(), v.data(), (int)v.size());
                    if (vTv > 1e-28)
                    {
                        const double beta = 2.0 / vTv;
                        std::vector<double> w(m - k, 0.0);
                        for (int i = 0; i < m - k; ++i)
                            w[i] = dot_contiguous(&B.at(k + i, k + 1), v.data(), n - k - 1);
                        for (int i = 0; i < m - k; ++i)
                            add_scaled_contiguous(&B.at(k + i, k + 1), v.data(), n - k - 1, -beta * w[i]);
                        std::vector<double> wV(n, 0.0);
                        for (int i = 0; i < n; ++i)
                            wV[i] = dot_contiguous(&V.at(i, k + 1), v.data(), n - k - 1);
                        for (int i = 0; i < n; ++i)
                            add_scaled_contiguous(&V.at(i, k + 1), v.data(), n - k - 1, -beta * wV[i]);
                    }
                }
                for (int j = k + 2; j < n; ++j)
                    B.at(k, j) = 0.0;
            }
        }
        return B;
    }

    // ---- 多线程版本 ----

    BidiagSharedCtx ctx;
    ctx.B = &B;
    ctx.U = &U;
    ctx.V = &V;
    ctx.m = m;
    ctx.n = n;
    ctx.num_threads = num_threads;
    ctx.phase = PHASE_IDLE;

    std::vector<std::vector<double>> w_locals(num_threads);
    ctx.w_locals = &w_locals;

    pthread_barrier_init(&ctx.barrier, nullptr, num_threads);

    std::vector<pthread_t> threads(num_threads - 1);
    std::vector<BidiagThreadArg> args(num_threads - 1);
    for (int i = 0; i < num_threads - 1; ++i)
    {
        args[i].tid = i + 1;
        args[i].ctx = &ctx;
        pthread_create(&threads[i], nullptr, bidiag_worker, &args[i]);
    }

    // 主线程绑定到核心0
    set_cpu_affinity(0);

    for (int k = 0; k < n; ++k)
    {
        ctx.k = k;

        // ---- 左侧Householder变换 ----
        std::vector<double> x(m - k);
        for (int i = 0; i < m - k; ++i)
            x[i] = B.at(k + i, k);

        double norm_x = vector_norm(x);

        bool do_left = false;
        std::vector<double> v_left, w_left;
        double beta_left = 0.0;

        if (norm_x > 1e-14 && k < m - 1)
        {
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;
            v_left = x;
            v_left[0] += sigma;
            double vTv = dot_contiguous(v_left.data(), v_left.data(), (int)v_left.size());

            if (vTv > 1e-28)
            {
                do_left = true;
                beta_left = 2.0 / vTv;
                ctx.beta = beta_left;
                ctx.v = &v_left;

                for (int t = 0; t < num_threads; ++t)
                    w_locals[t].assign(n - k, 0.0);

                w_left.assign(n - k, 0.0);
                for (int i = 0; i < m - k; ++i)
                    add_scaled_contiguous(w_left.data(), &B.at(k + i, k), n - k, v_left[i]);
                ctx.w = &w_left;
            }
        }

        // Phase 1: LEFT_UPDATE_B（即使不干活也要过barrier，否则worker死锁）
        ctx.phase = do_left ? PHASE_LEFT_UPDATE_B : PHASE_IDLE;
        pthread_barrier_wait(&ctx.barrier);

        if (do_left)
        {
            const int total = m - k;
            int start, end;
            row_range(0, num_threads, total, start, end);
            for (int i = start; i < end; ++i)
                add_scaled_contiguous(&B.at(k + i, k), w_left.data(), n - k, -beta_left * v_left[i]);
        }

        pthread_barrier_wait(&ctx.barrier);

        // Phase 2: LEFT_UPDATE_U
        ctx.phase = do_left ? PHASE_LEFT_UPDATE_U : PHASE_IDLE;
        pthread_barrier_wait(&ctx.barrier);

        if (do_left)
        {
            const int total = m;
            int start, end;
            row_range(0, num_threads, total, start, end);
            for (int i = start; i < end; ++i)
            {
                double wU_i = dot_contiguous(&U.at(i, k), v_left.data(), m - k);
                add_scaled_contiguous(&U.at(i, k), v_left.data(), m - k, -beta_left * wU_i);
            }
        }

        pthread_barrier_wait(&ctx.barrier);

        for (int i = k + 1; i < m; ++i)
            B.at(i, k) = 0.0;

        // ---- 右侧Householder变换 ----
        bool do_right = false;
        std::vector<double> v_right;
        double beta_right = 0.0;

        if (k < n - 2)
        {
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j)
                y[j] = B.at(k, k + 1 + j);

            double norm_y = vector_norm(y);

            if (norm_y > 1e-14)
            {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;
                v_right = y;
                v_right[0] += sigma;
                double vTv = dot_contiguous(v_right.data(), v_right.data(), (int)v_right.size());

                if (vTv > 1e-28)
                {
                    do_right = true;
                    beta_right = 2.0 / vTv;
                    ctx.beta = beta_right;
                    ctx.v = &v_right;
                }
            }
        }

        // Phase 3: RIGHT_UPDATE_BV
        ctx.phase = do_right ? PHASE_RIGHT_UPDATE_BV : PHASE_IDLE;
        pthread_barrier_wait(&ctx.barrier);

        if (do_right)
        {
            {
                const int total = m - k;
                int start, end;
                row_range(0, num_threads, total, start, end);
                for (int i = start; i < end; ++i)
                {
                    double w_i = dot_contiguous(&B.at(k + i, k + 1), v_right.data(), n - k - 1);
                    add_scaled_contiguous(&B.at(k + i, k + 1), v_right.data(), n - k - 1, -beta_right * w_i);
                }
            }
            {
                const int total = n;
                int start, end;
                row_range(0, num_threads, total, start, end);
                for (int i = start; i < end; ++i)
                {
                    double wV_i = dot_contiguous(&V.at(i, k + 1), v_right.data(), n - k - 1);
                    add_scaled_contiguous(&V.at(i, k + 1), v_right.data(), n - k - 1, -beta_right * wV_i);
                }
            }
        }

        pthread_barrier_wait(&ctx.barrier);

        if (k < n - 2)
        {
            for (int j = k + 2; j < n; ++j)
                B.at(k, j) = 0.0;
        }
    }

    ctx.phase = PHASE_EXIT;
    pthread_barrier_wait(&ctx.barrier);

    for (int i = 0; i < num_threads - 1; ++i)
        pthread_join(threads[i], nullptr);

    pthread_barrier_destroy(&ctx.barrier);

    return B;
}
