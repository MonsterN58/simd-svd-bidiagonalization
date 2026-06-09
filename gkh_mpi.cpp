// gkh_mpi.cpp
// MPI 并行版 GKH（Golub-Kahan）迭代实现
// 实现要求：
//   1. 主从式任务池架构（§7.1.1）
//   2. 改变迭代方式：从池中取非1×1子矩阵反复迭代直到可分割（§7.1.2）
//   3. Profiling 统计通信/计算时间（§7.1.3）

#include "gkh_mpi.h"
#include "givens.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <queue>
#include <iostream>
#include <iomanip>
#include <mpi.h>

// ===================== MPI 通信协议 =====================
// Tag 定义
static const int TAG_REQUEST   = 0;  // worker -> master: 请求任务
static const int TAG_TASK      = 1;  // master -> worker: 分配任务数据
static const int TAG_RESULT    = 2;  // worker -> master: 返回结果
static const int TAG_TERMINATE = 3;  // master -> worker: 终止信号

namespace
{
    // 活动块
    struct Block
    {
        int l;
        int r;
    };

    // ===================== 串行辅助函数（与 gkh.cpp 一致） =====================

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

    // 单块 bulge chase
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

    // 分割活动块
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

    // ===================== 串行版 GKH（单进程回退） =====================
    static bool gkh_serial(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
    {
        const int n = B.cols();
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

            for (int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i)
                if (blocks[i].r > blocks[i].l)
                    one_block_step(U, B, V, blocks[i].l, blocks[i].r);
        }

        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
            B.at(i, i + 1) = 0.0;
        make_nonnegative_and_sort(U, B, V);
        return converged;
    }

    // ===================== 子矩阵提取与写回 =====================

    // 从全局 B 中提取 [l,r] 范围的子矩阵数据（B 子块 + U 列 + V 列）
    struct SubMatrixData
    {
        int l, r;          // 全局范围
        int sub_size;      // r - l + 1
        int m, n;          // 全局 B 的尺寸
        std::vector<double> B_sub;  // sub_size x sub_size
        std::vector<double> U_cols; // m x sub_size
        std::vector<double> V_cols; // n x sub_size
    };

    static SubMatrixData extract_submatrix(const Matrix &U, const Matrix &B, const Matrix &V,
                                            int l, int r)
    {
        SubMatrixData data;
        data.l = l;
        data.r = r;
        data.sub_size = r - l + 1;
        data.m = B.rows();
        data.n = B.cols();

        // 提取 B[l..r, l..r]
        data.B_sub.resize(data.sub_size * data.sub_size, 0.0);
        for (int i = 0; i < data.sub_size; ++i)
            for (int j = 0; j < data.sub_size; ++j)
                data.B_sub[i * data.sub_size + j] = B.at(l + i, l + j);

        // 提取 U 的列 l..r
        data.U_cols.resize(data.m * data.sub_size);
        for (int col = 0; col < data.sub_size; ++col)
            for (int row = 0; row < data.m; ++row)
                data.U_cols[col * data.m + row] = U.at(row, l + col);

        // 提取 V 的列 l..r
        data.V_cols.resize(data.n * data.sub_size);
        for (int col = 0; col < data.sub_size; ++col)
            for (int row = 0; row < data.n; ++row)
                data.V_cols[col * data.n + row] = V.at(row, l + col);

        return data;
    }

    // 将子矩阵结果写回全局矩阵
    static void write_back_submatrix(Matrix &U, Matrix &B, Matrix &V,
                                      const SubMatrixData &data)
    {
        int l = data.l;
        int sub_size = data.sub_size;

        // 写回 B[l..r, l..r]
        for (int i = 0; i < sub_size; ++i)
            for (int j = 0; j < sub_size; ++j)
                B.at(l + i, l + j) = data.B_sub[i * sub_size + j];

        // 写回 U 列
        for (int col = 0; col < sub_size; ++col)
            for (int row = 0; row < data.m; ++row)
                U.at(row, l + col) = data.U_cols[col * data.m + row];

        // 写回 V 列
        for (int col = 0; col < sub_size; ++col)
            for (int row = 0; row < data.n; ++row)
                V.at(row, l + col) = data.V_cols[col * data.n + row];
    }

    // ===================== MPI 数据传输辅助 =====================

    // 发送子矩阵任务到 worker
    static void send_task(const SubMatrixData &data, int dest)
    {
        // 先发送元信息 [l, r, m, n]
        int meta[4] = {data.l, data.r, data.m, data.n};
        MPI_Send(meta, 4, MPI_INT, dest, TAG_TASK, MPI_COMM_WORLD);

        // 发送 B 子块
        MPI_Send(data.B_sub.data(), static_cast<int>(data.B_sub.size()),
                 MPI_DOUBLE, dest, TAG_TASK, MPI_COMM_WORLD);

        // 发送 U 列
        MPI_Send(data.U_cols.data(), static_cast<int>(data.U_cols.size()),
                 MPI_DOUBLE, dest, TAG_TASK, MPI_COMM_WORLD);

        // 发送 V 列
        MPI_Send(data.V_cols.data(), static_cast<int>(data.V_cols.size()),
                 MPI_DOUBLE, dest, TAG_TASK, MPI_COMM_WORLD);
    }

    // worker 接收子矩阵任务
    static SubMatrixData recv_task(int source)
    {
        SubMatrixData data;
        int meta[4];
        MPI_Recv(meta, 4, MPI_INT, source, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        data.l = meta[0];
        data.r = meta[1];
        data.m = meta[2];
        data.n = meta[3];
        data.sub_size = data.r - data.l + 1;

        data.B_sub.resize(data.sub_size * data.sub_size);
        MPI_Recv(data.B_sub.data(), static_cast<int>(data.B_sub.size()),
                 MPI_DOUBLE, source, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        data.U_cols.resize(data.m * data.sub_size);
        MPI_Recv(data.U_cols.data(), static_cast<int>(data.U_cols.size()),
                 MPI_DOUBLE, source, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        data.V_cols.resize(data.n * data.sub_size);
        MPI_Recv(data.V_cols.data(), static_cast<int>(data.V_cols.size()),
                 MPI_DOUBLE, source, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        return data;
    }

    // worker 发送结果回 master
    static void send_result(const SubMatrixData &data, int dest)
    {
        // 先发 meta
        int meta[4] = {data.l, data.r, data.m, data.n};
        MPI_Send(meta, 4, MPI_INT, dest, TAG_RESULT, MPI_COMM_WORLD);

        // 发新的子任务数目
        // (结果 B_sub 和新产生的子任务 blocks 一起发送)
        MPI_Send(data.B_sub.data(), static_cast<int>(data.B_sub.size()),
                 MPI_DOUBLE, dest, TAG_RESULT, MPI_COMM_WORLD);

        MPI_Send(data.U_cols.data(), static_cast<int>(data.U_cols.size()),
                 MPI_DOUBLE, dest, TAG_RESULT, MPI_COMM_WORLD);

        MPI_Send(data.V_cols.data(), static_cast<int>(data.V_cols.size()),
                 MPI_DOUBLE, dest, TAG_RESULT, MPI_COMM_WORLD);
    }

    // master 接收结果
    static SubMatrixData recv_result(int source)
    {
        SubMatrixData data;
        int meta[4];
        MPI_Recv(meta, 4, MPI_INT, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        data.l = meta[0];
        data.r = meta[1];
        data.m = meta[2];
        data.n = meta[3];
        data.sub_size = data.r - data.l + 1;

        data.B_sub.resize(data.sub_size * data.sub_size);
        MPI_Recv(data.B_sub.data(), static_cast<int>(data.B_sub.size()),
                 MPI_DOUBLE, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        data.U_cols.resize(data.m * data.sub_size);
        MPI_Recv(data.U_cols.data(), static_cast<int>(data.U_cols.size()),
                 MPI_DOUBLE, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        data.V_cols.resize(data.n * data.sub_size);
        MPI_Recv(data.V_cols.data(), static_cast<int>(data.V_cols.size()),
                 MPI_DOUBLE, source, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        return data;
    }

    // ===================== Worker：在局部子矩阵上反复迭代 =====================
    // 改变迭代方式(§7.1.2)：
    // 对子矩阵反复执行 bulge chase 直到该子矩阵可以再分割（或收敛为1×1）
    // 返回更新后的 SubMatrixData

    static void worker_iterate_submatrix(SubMatrixData &data, int max_iter, double tol)
    {
        int sub_size = data.sub_size;
        if (sub_size <= 1)
            return;

        // 构建局部矩阵：B_local 为 sub_size x sub_size
        // U_local 为 m x sub_size, V_local 为 n x sub_size
        Matrix B_local(sub_size, sub_size);
        for (int i = 0; i < sub_size; ++i)
            for (int j = 0; j < sub_size; ++j)
                B_local.at(i, j) = data.B_sub[i * sub_size + j];

        Matrix U_local(data.m, sub_size);
        for (int col = 0; col < sub_size; ++col)
            for (int row = 0; row < data.m; ++row)
                U_local.at(row, col) = data.U_cols[col * data.m + row];

        Matrix V_local(data.n, sub_size);
        for (int col = 0; col < sub_size; ++col)
            for (int row = 0; row < data.n; ++row)
                V_local.at(row, col) = data.V_cols[col * data.n + row];

        // 反复迭代直到可再分割
        // "可再分割"= 分割后产生多于1个块（即某个超对角元素收敛为零）
        for (int iter = 0; iter < max_iter; ++iter)
        {
            // 清理
            for (int i = 0; i < sub_size; ++i)
                for (int j = 0; j < sub_size; ++j)
                    if (j != i && j != i + 1 && std::fabs(B_local.at(i, j)) <= tol)
                        B_local.at(i, j) = 0.0;

            // 处理对角零
            {
                const double eps = std::numeric_limits<double>::epsilon();
                const double diag_tol = tol;
                const double super_tol = tol * (1.0 + 10.0 * eps);
                for (int k = 0; k < sub_size - 1; ++k)
                {
                    if (std::fabs(B_local.at(k, k)) <= diag_tol &&
                        std::fabs(B_local.at(k, k + 1)) > super_tol)
                    {
                        // chase zero diagonal locally
                        for (int i = k; i <= sub_size - 2; ++i)
                        {
                            double c, s, rr;
                            givens_rotation(B_local.at(i, i), B_local.at(i, i + 1), c, s, rr, false);
                            apply_right_cols(B_local, i, i + 1, c, s);
                            apply_right_cols(V_local, i, i + 1, c, s);

                            if (i + 1 < sub_size)
                            {
                                givens_rotation(B_local.at(i, i), B_local.at(i + 1, i), c, s, rr, true);
                                apply_left_rows(B_local, i, i + 1, c, s);
                                accumulate_left_into_U(U_local, i, i + 1, c, s);
                            }
                        }
                        // cleanup after chase
                        for (int ii = 0; ii < sub_size; ++ii)
                            for (int jj = 0; jj < sub_size; ++jj)
                                if (jj != ii && jj != ii + 1 && std::fabs(B_local.at(ii, jj)) <= tol)
                                    B_local.at(ii, jj) = 0.0;
                    }
                }
            }

            // 检查是否可以分割
            bool can_split = false;
            for (int k = 0; k < sub_size - 1; ++k)
            {
                const double a = std::fabs(B_local.at(k, k));
                const double d = std::fabs(B_local.at(k + 1, k + 1));
                const double crit = tol * (a + d + 1.0);
                if (std::fabs(B_local.at(k, k + 1)) <= crit)
                {
                    can_split = true;
                    break;
                }
            }

            // 检查是否全部收敛（所有超对角元素为0）
            bool all_converged = true;
            for (int k = 0; k < sub_size - 1; ++k)
            {
                const double a = std::fabs(B_local.at(k, k));
                const double d = std::fabs(B_local.at(k + 1, k + 1));
                const double crit = tol * (a + d + 1.0);
                if (std::fabs(B_local.at(k, k + 1)) > crit)
                {
                    all_converged = false;
                    break;
                }
            }

            if (can_split || all_converged)
                break;

            // 对整个子矩阵做一步 bulge chase
            // 先做分块（此时可能有内部多个块）
            std::vector<Block> local_blocks;
            {
                int ll = 0;
                while (ll < sub_size)
                {
                    int rr = ll;
                    while (rr < sub_size - 1 && std::fabs(B_local.at(rr, rr + 1)) > 0.0)
                        ++rr;
                    local_blocks.push_back({ll, rr});
                    ll = rr + 1;
                }
            }

            for (int i = static_cast<int>(local_blocks.size()) - 1; i >= 0; --i)
            {
                if (local_blocks[i].r > local_blocks[i].l)
                    one_block_step(U_local, B_local, V_local,
                                   local_blocks[i].l, local_blocks[i].r);
            }
        }

        // 将结果写回 data
        for (int i = 0; i < sub_size; ++i)
            for (int j = 0; j < sub_size; ++j)
                data.B_sub[i * sub_size + j] = B_local.at(i, j);

        for (int col = 0; col < sub_size; ++col)
            for (int row = 0; row < data.m; ++row)
                data.U_cols[col * data.m + row] = U_local.at(row, col);

        for (int col = 0; col < sub_size; ++col)
            for (int row = 0; row < data.n; ++row)
                data.V_cols[col * data.n + row] = V_local.at(row, col);
    }

} // namespace

// ===================== 主函数 =====================

bool gkh_svd_from_bidiagonal_mpi(Matrix &U, Matrix &B, Matrix &V,
                                  int max_iter, double tol)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int m = B.rows();
    const int n = B.cols();

    // 单进程或数据太小：退化为串行
    if (size <= 1)
    {
        return gkh_serial(U, B, V, max_iter, tol);
    }

    // ==================== Profiling 计时 ====================
    double time_comm = 0.0;     // 通信时间
    double time_compute = 0.0;  // 计算时间
    double time_wait = 0.0;     // 等待时间

    if (rank == 0)
    {
        // ==================== MASTER 进程 ====================
        if (m < n)
            throw std::invalid_argument("gkh_svd_from_bidiagonal_mpi: requires m >= n");

        double t_start = MPI_Wtime();

        // 第一步：初始清理
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);

        // 初始分割得到任务池
        std::queue<Block> task_pool;
        {
            auto blocks = split_active_blocks(B, n, tol);
            for (const auto &blk : blocks)
                if (blk.r > blk.l)
                    task_pool.push(blk);
        }

        // worker 状态追踪
        std::vector<bool> worker_busy(size, false);
        int busy_count = 0;
        int terminated_count = 0;  // 已终止的 worker 数

        // 主循环：分发任务、接收结果
        while (!task_pool.empty() || busy_count > 0)
        {
            // 使用 MPI_Probe 等待任意 worker 的消息
            MPI_Status status;
            double t_wait_start = MPI_Wtime();
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            time_wait += MPI_Wtime() - t_wait_start;

            int source = status.MPI_SOURCE;
            int tag = status.MPI_TAG;

            if (tag == TAG_REQUEST)
            {
                // Worker 请求任务
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, source, TAG_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (!task_pool.empty())
                {
                    // 分配任务
                    Block blk = task_pool.front();
                    task_pool.pop();

                    double t_comm_start = MPI_Wtime();
                    SubMatrixData sub_data = extract_submatrix(U, B, V, blk.l, blk.r);
                    send_task(sub_data, source);
                    time_comm += MPI_Wtime() - t_comm_start;

                    worker_busy[source] = true;
                    busy_count++;
                }
                else
                {
                    // 没有任务了，发送终止信号
                    int term = -1;
                    MPI_Send(&term, 1, MPI_INT, source, TAG_TERMINATE, MPI_COMM_WORLD);
                    terminated_count++;
                }
            }
            else if (tag == TAG_RESULT)
            {
                // Worker 返回结果
                double t_comm_start = MPI_Wtime();
                SubMatrixData result = recv_result(source);
                time_comm += MPI_Wtime() - t_comm_start;

                // 写回全局矩阵
                write_back_submatrix(U, B, V, result);

                worker_busy[source] = false;
                busy_count--;

                // 对返回的子矩阵范围再做分割，产生新的子任务
                int l = result.l;
                int r = result.r;

                // 在全局 B 上对 [l, r] 范围重新分割
                for (int k = l; k < r; ++k)
                {
                    const double a = std::fabs(B.at(k, k));
                    const double d = std::fabs(B.at(k + 1, k + 1));
                    const double crit = tol * (a + d + 1.0);
                    if (std::fabs(B.at(k, k + 1)) <= crit)
                        B.at(k, k + 1) = 0.0;
                }

                // 重新识别 [l,r] 范围内的活动块
                int ll = l;
                while (ll <= r)
                {
                    int rr = ll;
                    while (rr < r && std::fabs(B.at(rr, rr + 1)) > 0.0)
                        ++rr;
                    if (rr > ll) // 非1×1块，加入任务池
                        task_pool.push({ll, rr});
                    ll = rr + 1;
                }
            }
        }

        // 确保所有 worker 都收到终止信号
        // 阻塞等待，直到所有 size-1 个 worker 都已终止
        while (terminated_count < size - 1)
        {
            MPI_Status status;
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, TAG_REQUEST,
                     MPI_COMM_WORLD, &status);
            int term = -1;
            MPI_Send(&term, 1, MPI_INT, status.MPI_SOURCE, TAG_TERMINATE,
                     MPI_COMM_WORLD);
            terminated_count++;
        }

        double t_end = MPI_Wtime();

        // Profiling 输出
        std::cerr << "[MPI-GKH rank 0] total=" << std::fixed << std::setprecision(2)
                  << (t_end - t_start) * 1000.0 << "ms"
                  << " comm=" << time_comm * 1000.0 << "ms"
                  << " wait=" << time_wait * 1000.0 << "ms"
                  << std::endl;

        // 收尾
        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
            B.at(i, i + 1) = 0.0;
        make_nonnegative_and_sort(U, B, V);

        // 同步所有进程，确保 worker 全部退出后再进入下一个测试
        MPI_Barrier(MPI_COMM_WORLD);
        return true; // 主从模式下假设收敛（任务池空即代表全部收敛）
    }
    else
    {
        // ==================== WORKER 进程 ====================
        double t_start = MPI_Wtime();

        while (true)
        {
            // 向 master 请求任务
            int dummy = rank;
            double t_comm_start = MPI_Wtime();
            MPI_Send(&dummy, 1, MPI_INT, 0, TAG_REQUEST, MPI_COMM_WORLD);

            // 等待 master 回复（可能是任务或终止信号）
            MPI_Status status;
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            time_comm += MPI_Wtime() - t_comm_start;

            if (status.MPI_TAG == TAG_TERMINATE)
            {
                // 收到终止信号
                int term;
                MPI_Recv(&term, 1, MPI_INT, 0, TAG_TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                break;
            }

            // 接收任务数据
            t_comm_start = MPI_Wtime();
            SubMatrixData task_data = recv_task(0);
            time_comm += MPI_Wtime() - t_comm_start;

            // 在局部反复迭代子矩阵（改变迭代方式 §7.1.2）
            double t_comp_start = MPI_Wtime();
            worker_iterate_submatrix(task_data, max_iter, tol);
            time_compute += MPI_Wtime() - t_comp_start;

            // 发回结果
            t_comm_start = MPI_Wtime();
            send_result(task_data, 0);
            time_comm += MPI_Wtime() - t_comm_start;
        }

        double t_end = MPI_Wtime();

        // Profiling 输出
        std::cerr << "[MPI-GKH rank " << rank << "] total="
                  << std::fixed << std::setprecision(2)
                  << (t_end - t_start) * 1000.0 << "ms"
                  << " comm=" << time_comm * 1000.0 << "ms"
                  << " compute=" << time_compute * 1000.0 << "ms"
                  << std::endl;

        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        return true; // worker 返回值不重要
    }
}
