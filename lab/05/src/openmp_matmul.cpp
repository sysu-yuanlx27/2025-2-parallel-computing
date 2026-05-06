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

enum class Schedule {
    Default,
    Static,
    Dynamic,
};

struct Config {
    int m = 128;
    int n = 128;
    int k = 128;
    int threads = 4;
    Schedule schedule = Schedule::Default;
    bool verify = false;
    bool print_matrices = false;
};

bool in_range(int x, int low, int high) {
    return x >= low && x <= high;
}

const char* schedule_name(Schedule schedule) {
    switch (schedule) {
        case Schedule::Default:
            return "default";
        case Schedule::Static:
            return "static";
        case Schedule::Dynamic:
            return "dynamic";
    }
    return "unknown";
}

Schedule parse_schedule(const std::string& text) {
    if (text == "default") return Schedule::Default;
    if (text == "static") return Schedule::Static;
    if (text == "dynamic") return Schedule::Dynamic;
    throw std::runtime_error("schedule must be one of: default, static, dynamic");
}

void fill_random(std::vector<double>& values, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& value : values) {
        value = dist(gen);
    }
}

void print_matrix(const std::string& name,
                  const std::vector<double>& matrix,
                  int rows,
                  int cols) {
    std::cout << name << " =\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << matrix[static_cast<std::size_t>(i) * cols + j];
        }
        std::cout << '\n';
    }
}

Config parse_args(int argc, char** argv) {
    if (argc < 5) {
        throw std::runtime_error(
            "Usage: ./openmp_matmul <m> <n> <k> <threads(1-16)> "
            "[default|static|dynamic] [--verify] [--print]");
    }

    Config cfg;
    cfg.m = std::stoi(argv[1]);
    cfg.n = std::stoi(argv[2]);
    cfg.k = std::stoi(argv[3]);
    cfg.threads = std::stoi(argv[4]);

    int arg_index = 5;
    if (arg_index < argc && std::string(argv[arg_index]).rfind("--", 0) != 0) {
        cfg.schedule = parse_schedule(argv[arg_index]);
        ++arg_index;
    }

    for (int i = arg_index; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--verify") {
            cfg.verify = true;
        } else if (arg == "--print") {
            cfg.print_matrices = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (!in_range(cfg.m, 128, 2048) ||
        !in_range(cfg.n, 128, 2048) ||
        !in_range(cfg.k, 128, 2048)) {
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

void openmp_matmul(const std::vector<double>& a,
                   const std::vector<double>& b,
                   std::vector<double>& c,
                   const Config& cfg) {
    std::fill(c.begin(), c.end(), 0.0);
    omp_set_num_threads(cfg.threads);

    if (cfg.schedule == Schedule::Static) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < cfg.m; ++i) {
            const std::size_t a_base = static_cast<std::size_t>(i) * cfg.n;
            const std::size_t c_base = static_cast<std::size_t>(i) * cfg.k;
            for (int p = 0; p < cfg.n; ++p) {
                const double a_ip = a[a_base + p];
                const std::size_t b_base = static_cast<std::size_t>(p) * cfg.k;
                for (int j = 0; j < cfg.k; ++j) {
                    c[c_base + j] += a_ip * b[b_base + j];
                }
            }
        }
    } else if (cfg.schedule == Schedule::Dynamic) {
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < cfg.m; ++i) {
            const std::size_t a_base = static_cast<std::size_t>(i) * cfg.n;
            const std::size_t c_base = static_cast<std::size_t>(i) * cfg.k;
            for (int p = 0; p < cfg.n; ++p) {
                const double a_ip = a[a_base + p];
                const std::size_t b_base = static_cast<std::size_t>(p) * cfg.k;
                for (int j = 0; j < cfg.k; ++j) {
                    c[c_base + j] += a_ip * b[b_base + j];
                }
            }
        }
    } else {
#pragma omp parallel for
        for (int i = 0; i < cfg.m; ++i) {
            const std::size_t a_base = static_cast<std::size_t>(i) * cfg.n;
            const std::size_t c_base = static_cast<std::size_t>(i) * cfg.k;
            for (int p = 0; p < cfg.n; ++p) {
                const double a_ip = a[a_base + p];
                const std::size_t b_base = static_cast<std::size_t>(p) * cfg.k;
                for (int j = 0; j < cfg.k; ++j) {
                    c[c_base + j] += a_ip * b[b_base + j];
                }
            }
        }
    }
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    double diff = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        diff = std::max(diff, std::abs(lhs[i] - rhs[i]));
    }
    return diff;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);

        std::vector<double> a(static_cast<std::size_t>(cfg.m) * cfg.n);
        std::vector<double> b(static_cast<std::size_t>(cfg.n) * cfg.k);
        std::vector<double> c(static_cast<std::size_t>(cfg.m) * cfg.k, 0.0);

        fill_random(a, 42U);
        fill_random(b, 43U);

        const auto t0 = std::chrono::steady_clock::now();
        openmp_matmul(a, b, c, cfg);
        const auto t1 = std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        const double flops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k;

        std::cout << "Task: OpenMP matrix multiplication\n";
        std::cout << "A: " << cfg.m << " x " << cfg.n << '\n';
        std::cout << "B: " << cfg.n << " x " << cfg.k << '\n';
        std::cout << "C: " << cfg.m << " x " << cfg.k << '\n';
        std::cout << "Threads: " << cfg.threads << '\n';
        std::cout << "Schedule: " << schedule_name(cfg.schedule) << '\n';
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec): " << seconds << '\n';
        std::cout << std::setprecision(3) << "GFLOPS: " << flops / seconds / 1e9 << '\n';

        if (cfg.verify) {
            std::vector<double> serial(c.size(), 0.0);
            serial_matmul(a, b, serial, cfg.m, cfg.n, cfg.k);
            const double diff = max_abs_diff(c, serial);
            std::cout << std::setprecision(10);
            std::cout << "Verify: " << (diff < 1e-8 ? "PASS" : "FAIL")
                      << ", max_abs_diff=" << diff << '\n';
        }

        if (cfg.print_matrices) {
            print_matrix("A", a, cfg.m, cfg.n);
            print_matrix("B", b, cfg.n, cfg.k);
            print_matrix("C", c, cfg.m, cfg.k);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
