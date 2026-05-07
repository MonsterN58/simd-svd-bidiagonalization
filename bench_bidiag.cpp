#include "bidiagonalization.h"
#include "matrix.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static double bidiagonal_structure_error(const Matrix &B)
{
    double max_abs = 0.0;
    for (int i = 0; i < B.rows(); ++i)
    {
        for (int j = 0; j < B.cols(); ++j)
        {
            if (j != i && j != i + 1)
            {
                max_abs = std::max(max_abs, std::fabs(B.at(i, j)));
            }
        }
    }
    return max_abs;
}

static double time_once(int n, long long seed)
{
    Matrix A = Matrix::random(n, n, -1.0, 1.0, seed);
    Matrix U, V;
    const auto beg = std::chrono::high_resolution_clock::now();
    Matrix B = to_bidiagonal(A, U, V);
    const auto end = std::chrono::high_resolution_clock::now();
    const double err = bidiagonal_structure_error(B);
    if (err > 1e-9)
    {
        std::cerr << "bidiagonal structure error: " << err << '\n';
        std::exit(2);
    }
    return std::chrono::duration<double, std::milli>(end - beg).count();
}

int main(int argc, char **argv)
{
    const std::string label = argc >= 2 ? argv[1] : "unknown";
    const std::vector<int> sizes = argc >= 3 ? std::vector<int>{std::stoi(argv[2])}
                                             : std::vector<int>{100, 200, 400, 600, 800, 1000};
    const int reps_override = argc >= 4 ? std::stoi(argv[3]) : 0;

    std::cout << "variant,N,time_ms\n";
    for (int n : sizes)
    {
        std::vector<double> times;
        const int reps = reps_override > 0 ? reps_override : ((n <= 400) ? 5 : 3);
        for (int r = 0; r < reps; ++r)
        {
            times.push_back(time_once(n, 20260507LL + n * 17 + r));
        }
        std::sort(times.begin(), times.end());
        std::cout << label << ',' << n << ',' << times[times.size() / 2] << '\n';
    }
    return 0;
}
