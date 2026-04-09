#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_MKL
#include <mkl_cblas.h>
#endif

struct Config {
    int m = 512;
    int n = 512;
    int k = 512;
    std::string version = "naive";  // naive | reorder | unroll | mkl
    bool full_print = false;
    bool no_print = false;
};

static bool in_range(int x) {
    return x >= 512 && x <= 2048;
}

static void fill_random(std::vector<double>& v, uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& x : v) x = dist(gen);
}

static void print_matrix(const std::vector<double>& mat, int rows, int cols, const std::string& name, bool full_print) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    int r_lim = full_print ? rows : std::min(rows, 8);
    int c_lim = full_print ? cols : std::min(cols, 8);

    for (int i = 0; i < r_lim; ++i) {
        for (int j = 0; j < c_lim; ++j) {
            std::cout << std::fixed << std::setprecision(4) << mat[static_cast<size_t>(i) * cols + j] << " ";
        }
        if (!full_print && c_lim < cols) std::cout << "...";
        std::cout << "\n";
    }
    if (!full_print && r_lim < rows) {
        std::cout << "...\n";
    }
}

static void matmul_naive(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int m, int n, int k) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            double sum = 0.0;
            for (int p = 0; p < n; ++p) {
                sum += A[static_cast<size_t>(i) * n + p] * B[static_cast<size_t>(p) * k + j];
            }
            C[static_cast<size_t>(i) * k + j] = sum;
        }
    }
}

static void matmul_reorder(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int m, int n, int k) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < n; ++p) {
            const double a = A[static_cast<size_t>(i) * n + p];
            const size_t b_base = static_cast<size_t>(p) * k;
            const size_t c_base = static_cast<size_t>(i) * k;
            for (int j = 0; j < k; ++j) {
                C[c_base + j] += a * B[b_base + j];
            }
        }
    }
}

static void matmul_unroll4(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int m, int n, int k) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < m; ++i) {
        const size_t c_base = static_cast<size_t>(i) * k;
        for (int p = 0; p < n; ++p) {
            const double a = A[static_cast<size_t>(i) * n + p];
            const size_t b_base = static_cast<size_t>(p) * k;

            int j = 0;
            for (; j + 3 < k; j += 4) {
                C[c_base + j] += a * B[b_base + j];
                C[c_base + j + 1] += a * B[b_base + j + 1];
                C[c_base + j + 2] += a * B[b_base + j + 2];
                C[c_base + j + 3] += a * B[b_base + j + 3];
            }
            for (; j < k; ++j) {
                C[c_base + j] += a * B[b_base + j];
            }
        }
    }
}

static void matmul_mkl(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int m, int n, int k) {
#ifdef USE_MKL
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, A.data(), n, B.data(), k, 0.0, C.data(), k);
#else
    (void)A;
    (void)B;
    (void)C;
    (void)m;
    (void)n;
    (void)k;
    throw std::runtime_error("MKL support not enabled. Rebuild with -DUSE_MKL and MKL link flags.");
#endif
}

static Config parse_args(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error("Usage: ./matmul <m> <n> <k> <version: naive|reorder|unroll|mkl> [--full-print] [--no-print]");
    }

    Config cfg;
    cfg.m = std::stoi(argv[1]);
    cfg.n = std::stoi(argv[2]);
    cfg.k = std::stoi(argv[3]);
    cfg.version = argv[4];

    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full-print") == 0) {
            cfg.full_print = true;
        } else if (std::strcmp(argv[i], "--no-print") == 0) {
            cfg.no_print = true;
        }
    }

    if (!in_range(cfg.m) || !in_range(cfg.n) || !in_range(cfg.k)) {
        throw std::runtime_error("m, n, k must be in [512, 2048].");
    }

    if (cfg.version != "naive" && cfg.version != "reorder" && cfg.version != "unroll" && cfg.version != "mkl") {
        throw std::runtime_error("version must be one of: naive, reorder, unroll, mkl");
    }

    return cfg;
}

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);

        std::vector<double> A(static_cast<size_t>(cfg.m) * cfg.n);
        std::vector<double> B(static_cast<size_t>(cfg.n) * cfg.k);
        std::vector<double> C(static_cast<size_t>(cfg.m) * cfg.k);

        fill_random(A, 42);
        fill_random(B, 777);

        auto t0 = std::chrono::steady_clock::now();

        if (cfg.version == "naive") {
            matmul_naive(A, B, C, cfg.m, cfg.n, cfg.k);
        } else if (cfg.version == "reorder") {
            matmul_reorder(A, B, C, cfg.m, cfg.n, cfg.k);
        } else if (cfg.version == "unroll") {
            matmul_unroll4(A, B, C, cfg.m, cfg.n, cfg.k);
        } else {
            matmul_mkl(A, B, C, cfg.m, cfg.n, cfg.k);
        }

        auto t1 = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = t1 - t0;

        if (!cfg.no_print) {
            print_matrix(A, cfg.m, cfg.n, "A", cfg.full_print);
            print_matrix(B, cfg.n, cfg.k, "B", cfg.full_print);
            print_matrix(C, cfg.m, cfg.k, "C", cfg.full_print);
        }

        const double seconds = elapsed.count();
        const double flops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k;
        const double gflops = flops / seconds / 1e9;

        std::cout << "Version: " << cfg.version << "\n";
        std::cout << "Time (sec): " << std::fixed << std::setprecision(6) << seconds << "\n";
        std::cout << "GFLOPS: " << std::fixed << std::setprecision(3) << gflops << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
