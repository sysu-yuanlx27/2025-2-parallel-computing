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
#include <thread>
#include <vector>

#include <pthread.h>

namespace {

struct Config {
    int m = 128;
    int n = 128;
    int k = 128;
    int threads = 4;
    bool verify = false;
};

struct WorkerArgs {
    const std::vector<double>* a = nullptr;
    const std::vector<double>* b = nullptr;
    std::vector<double>* c = nullptr;
    int n = 0;
    int k = 0;
    int row_begin = 0;
    int row_end = 0;
};

bool in_range(int x, int low, int high) {
    return x >= low && x <= high;
}

void fill_random(std::vector<double>& values, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& value : values) {
        value = dist(gen);
    }
}

void* worker_matmul(void* raw_ptr) {
    const WorkerArgs* args = static_cast<const WorkerArgs*>(raw_ptr);
    const std::vector<double>& a = *args->a;
    const std::vector<double>& b = *args->b;
    std::vector<double>& c = *args->c;

    for (int i = args->row_begin; i < args->row_end; ++i) {
        const std::size_t a_base = static_cast<std::size_t>(i) * args->n;
        const std::size_t c_base = static_cast<std::size_t>(i) * args->k;
        for (int p = 0; p < args->n; ++p) {
            const double a_ip = a[a_base + p];
            const std::size_t b_base = static_cast<std::size_t>(p) * args->k;
            for (int j = 0; j < args->k; ++j) {
                c[c_base + j] += a_ip * b[b_base + j];
            }
        }
    }

    return nullptr;
}

Config parse_args(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error(
            "Usage: ./matmul_pthreads <m> <n> <k> <threads(1-16)> [--verify]");
    }

    Config cfg;
    cfg.m = std::stoi(argv[1]);
    cfg.n = std::stoi(argv[2]);
    cfg.k = std::stoi(argv[3]);
    cfg.threads = std::stoi(argv[4]);

    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify") {
            cfg.verify = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (!in_range(cfg.m, 128, 2048) || !in_range(cfg.n, 128, 2048) || !in_range(cfg.k, 128, 2048)) {
        throw std::runtime_error("m, n, k must be in [128, 2048]");
    }
    if (!in_range(cfg.threads, 1, 16)) {
        throw std::runtime_error("threads must be in [1, 16]");
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

}  // namespace

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

        std::vector<pthread_t> workers(static_cast<std::size_t>(cfg.threads));
        std::vector<WorkerArgs> args(static_cast<std::size_t>(cfg.threads));

        const int rows_per_thread = cfg.m / cfg.threads;
        const int extra_rows = cfg.m % cfg.threads;

        int row_begin = 0;

        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < cfg.threads; ++t) {
            const int row_count = rows_per_thread + (t < extra_rows ? 1 : 0);
            const int row_end = row_begin + row_count;

            args[static_cast<std::size_t>(t)].a = &a;
            args[static_cast<std::size_t>(t)].b = &b;
            args[static_cast<std::size_t>(t)].c = &c;
            args[static_cast<std::size_t>(t)].n = cfg.n;
            args[static_cast<std::size_t>(t)].k = cfg.k;
            args[static_cast<std::size_t>(t)].row_begin = row_begin;
            args[static_cast<std::size_t>(t)].row_end = row_end;

            const int rc = pthread_create(&workers[static_cast<std::size_t>(t)],
                                          nullptr,
                                          worker_matmul,
                                          &args[static_cast<std::size_t>(t)]);
            if (rc != 0) {
                throw std::runtime_error("pthread_create failed");
            }
            row_begin = row_end;
        }

        for (int t = 0; t < cfg.threads; ++t) {
            pthread_join(workers[static_cast<std::size_t>(t)], nullptr);
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        const double flops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k;
        const double gflops = flops / seconds / 1e9;

        std::cout << "Task: matrix_multiplication_pthreads\n";
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
