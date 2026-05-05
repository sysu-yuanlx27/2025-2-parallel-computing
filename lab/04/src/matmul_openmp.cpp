#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace {

struct Config {
    int m = 128;
    int n = 128;
    int k = 128;
    int threads = 4;
    bool verify = false;
};

bool in_range(int x, int low, int high) {
    return x >= low && x <= high;
}

void fill_random(std::vector<double>& values, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& value : values) value = dist(gen);
}

Config parse_args(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error(
            "Usage: ./matmul_openmp <m> <n> <k> <threads(1-256)> [--verify]");
    }

    Config cfg;
    cfg.m = std::stoi(argv[1]);
    cfg.n = std::stoi(argv[2]);
    cfg.k = std::stoi(argv[3]);
    cfg.threads = std::stoi(argv[4]);

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify") cfg.verify = true;
        else throw std::runtime_error("Unknown argument: " + arg);
    }

    if (!in_range(cfg.m, 128, 8192) || !in_range(cfg.n, 128, 8192) || !in_range(cfg.k, 128, 8192)) {
        throw std::runtime_error("m, n, k must be in [128, 8192]");
    }
    if (!in_range(cfg.threads, 1, 256)) {
        throw std::runtime_error("threads must be in [1, 256]");
    }

    return cfg;
}

void serial_matmul(const std::vector<double>& a,
                   const std::vector<double>& b,
                   std::vector<double>& c,
                   int m,
                   int n,
                   int k) {
    std::fill(c.begin(), c.end(), 0.0);
    for (int i = 0; i < m; ++i) {
        const std::size_t a_base = static_cast<std::size_t>(i) * n;
        const std::size_t c_base = static_cast<std::size_t>(i) * k;
        for (int p = 0; p < n; ++p) {
            const double a_ip = a[a_base + p];
            const std::size_t b_base = static_cast<std::size_t>(p) * k;
            for (int j = 0; j < k; ++j) {
                c[c_base + j] += a_ip * b[b_base + j];
            }
        }
    }
}

void openmp_matmul(const std::vector<double>& a,
                   const std::vector<double>& b,
                   std::vector<double>& c,
                   int m,
                   int n,
                   int k,
                   int threads) {
    std::fill(c.begin(), c.end(), 0.0);
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const std::size_t a_base = static_cast<std::size_t>(i) * n;
        const std::size_t c_base = static_cast<std::size_t>(i) * k;
        for (int p = 0; p < n; ++p) {
            const double a_ip = a[a_base + p];
            const std::size_t b_base = static_cast<std::size_t>(p) * k;
            for (int j = 0; j < k; ++j) {
                c[c_base + j] += a_ip * b[b_base + j];
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);

        const std::size_t a_size = static_cast<std::size_t>(cfg.m) * cfg.n;
        const std::size_t b_size = static_cast<std::size_t>(cfg.n) * cfg.k;
        const std::size_t c_size = static_cast<std::size_t>(cfg.m) * cfg.k;

        std::vector<double> a(a_size);
        std::vector<double> b(b_size);
        std::vector<double> c(c_size, 0.0);

        fill_random(a, 42U);
        fill_random(b, 43U);

        const auto t0 = std::chrono::steady_clock::now();
        openmp_matmul(a, b, c, cfg.m, cfg.n, cfg.k, cfg.threads);
        const auto t1 = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        const double flops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k;
        const double gflops = flops / seconds / 1e9;

        std::cout << "Task: matrix_multiplication_openmp\n";
        std::cout << "Shape: A(" << cfg.m << "x" << cfg.n << "), B(" << cfg.n << "x" << cfg.k << ")\n";
        std::cout << "Threads: " << cfg.threads << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec): " << seconds << "\n";
        std::cout << std::setprecision(3) << "GFLOPS: " << gflops << "\n";

        if (cfg.verify) {
            std::vector<double> serial(c_size, 0.0);
            serial_matmul(a, b, serial, cfg.m, cfg.n, cfg.k);

            double max_abs_diff = 0.0;
            for (std::size_t i = 0; i < c_size; ++i) {
                max_abs_diff = std::max(max_abs_diff, std::abs(c[i] - serial[i]));
            }
            const bool ok = max_abs_diff < 1e-8;
            std::cout << "Verify: " << (ok ? "PASS" : "FAIL")
                      << ", max_abs_diff=" << std::setprecision(10) << max_abs_diff << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
