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
    int size = 10000000;
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
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: ./array_sum_openmp <size> <threads(1-256)> [--verify]");
    }

    Config cfg;
    cfg.size = std::stoi(argv[1]);
    cfg.threads = std::stoi(argv[2]);

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify") cfg.verify = true;
        else throw std::runtime_error("Unknown argument: " + arg);
    }

    if (!in_range(cfg.size, 1000000, 1000000000)) {
        throw std::runtime_error("size must be in [1000000, 1000000000]");
    }
    if (!in_range(cfg.threads, 1, 256)) {
        throw std::runtime_error("threads must be in [1, 256]");
    }

    return cfg;
}

double serial_array_sum(const std::vector<double>& arr) {
    double sum = 0.0;
    for (const double val : arr) sum += val;
    return sum;
}

double openmp_array_sum(const std::vector<double>& arr, int threads) {
    double sum = 0.0;
    omp_set_num_threads(threads);
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (int i = 0; i < static_cast<int>(arr.size()); ++i) sum += arr[i];
    return sum;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);

        std::vector<double> arr(cfg.size);
        fill_random(arr, 42U);

        const auto t0 = std::chrono::steady_clock::now();
        double result = openmp_array_sum(arr, cfg.threads);
        const auto t1 = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        const double throughput = static_cast<double>(cfg.size) / seconds / 1e9;

        std::cout << "Task: array_sum_openmp\n";
        std::cout << "Size: " << cfg.size << "\n";
        std::cout << "Threads: " << cfg.threads << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec): " << seconds << "\n";
        std::cout << "Throughput (Gelem/sec): " << throughput << "\n";
        std::cout << "Result: " << result << "\n";

        if (cfg.verify) {
            double serial_result = serial_array_sum(arr);
            double diff = std::abs(result - serial_result);
            const bool ok = diff < 1e-6;
            std::cout << "Verify: " << (ok ? "PASS" : "FAIL")
                      << ", serial_result=" << std::setprecision(10) << serial_result
                      << ", diff=" << diff << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
