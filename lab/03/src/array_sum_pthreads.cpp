#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <pthread.h>

namespace {

struct Config {
    std::size_t n = 1ULL << 20;  // 1M
    int threads = 4;
    bool verify = false;
};

struct WorkerArgs {
    const std::vector<int>* data = nullptr;
    std::size_t begin = 0;
    std::size_t end = 0;
    long long partial = 0;
};

bool in_range(std::size_t x, std::size_t low, std::size_t high) {
    return x >= low && x <= high;
}

void fill_random(std::vector<int>& values, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (int& v : values) {
        v = dist(gen);
    }
}

void* worker_sum(void* raw_ptr) {
    WorkerArgs* args = static_cast<WorkerArgs*>(raw_ptr);
    const std::vector<int>& data = *args->data;

    long long local = 0;
    for (std::size_t i = args->begin; i < args->end; ++i) {
        local += data[i];
    }
    args->partial = local;
    return nullptr;
}

Config parse_args(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: ./array_sum_pthreads <n(1M-128M)> <threads(1-16)> [--verify]");
    }

    Config cfg;
    cfg.n = static_cast<std::size_t>(std::stoull(argv[1]));
    cfg.threads = std::stoi(argv[2]);

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify") {
            cfg.verify = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    const std::size_t one_million = 1ULL << 20;
    const std::size_t one_hundred_twenty_eight_million = 128ULL << 20;

    if (!in_range(cfg.n, one_million, one_hundred_twenty_eight_million)) {
        throw std::runtime_error("n must be in [1M, 128M], where M = 2^20");
    }
    if (cfg.threads < 1 || cfg.threads > 16) {
        throw std::runtime_error("threads must be in [1, 16]");
    }

    return cfg;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);

        std::vector<int> data(cfg.n);
        fill_random(data, 123U);

        std::vector<pthread_t> workers(static_cast<std::size_t>(cfg.threads));
        std::vector<WorkerArgs> args(static_cast<std::size_t>(cfg.threads));

        const std::size_t base = cfg.n / static_cast<std::size_t>(cfg.threads);
        const std::size_t rem = cfg.n % static_cast<std::size_t>(cfg.threads);

        std::size_t begin = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < cfg.threads; ++t) {
            const std::size_t len = base + (static_cast<std::size_t>(t) < rem ? 1 : 0);
            const std::size_t end = begin + len;

            args[static_cast<std::size_t>(t)].data = &data;
            args[static_cast<std::size_t>(t)].begin = begin;
            args[static_cast<std::size_t>(t)].end = end;
            args[static_cast<std::size_t>(t)].partial = 0;

            const int rc = pthread_create(&workers[static_cast<std::size_t>(t)],
                                          nullptr,
                                          worker_sum,
                                          &args[static_cast<std::size_t>(t)]);
            if (rc != 0) {
                throw std::runtime_error("pthread_create failed");
            }

            begin = end;
        }

        long long total = 0;
        for (int t = 0; t < cfg.threads; ++t) {
            pthread_join(workers[static_cast<std::size_t>(t)], nullptr);
            total += args[static_cast<std::size_t>(t)].partial;
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        const double throughput = static_cast<double>(cfg.n) / seconds / 1e6;

        std::cout << "Task: array_sum_pthreads\n";
        std::cout << "N: " << cfg.n << "\n";
        std::cout << "Threads: " << cfg.threads << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec): " << seconds << "\n";
        std::cout << std::setprecision(3) << "Throughput (Melems/s): " << throughput << "\n";
        std::cout << "Sum: " << total << "\n";

        if (cfg.verify) {
            const long long serial = std::accumulate(data.begin(), data.end(), 0LL);
            std::cout << "Verify: " << (serial == total ? "PASS" : "FAIL") << "\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
