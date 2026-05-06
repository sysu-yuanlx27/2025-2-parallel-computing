#include <algorithm>
#include <atomic>
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

#include <pthread.h>

namespace {

enum class Schedule {
    Static,
    Dynamic,
};

struct Config {
    int m = 128;
    int n = 128;
    int k = 128;
    int threads = 4;
    Schedule schedule = Schedule::Static;
    bool verify = false;
    bool print_matrices = false;
};

struct ParallelForTask {
    int start = 0;
    int end = 0;
    int inc = 1;
    void* (*functor)(int, void*) = nullptr;
    void* arg = nullptr;
    Schedule schedule = Schedule::Static;
    std::atomic<int> next_iteration{0};
};

struct ThreadArgs {
    ParallelForTask* task = nullptr;
    int iter_begin = 0;
    int iter_end = 0;
};

struct MatmulArgs {
    const std::vector<double>* a = nullptr;
    const std::vector<double>* b = nullptr;
    std::vector<double>* c = nullptr;
    int n = 0;
    int k = 0;
};

bool in_range(int x, int low, int high) {
    return x >= low && x <= high;
}

const char* schedule_name(Schedule schedule) {
    return schedule == Schedule::Static ? "static" : "dynamic";
}

Schedule parse_schedule(const std::string& text) {
    if (text == "static" || text == "default") return Schedule::Static;
    if (text == "dynamic") return Schedule::Dynamic;
    throw std::runtime_error("schedule must be one of: default, static, dynamic");
}

int iteration_count(int start, int end, int inc) {
    if (inc <= 0) {
        throw std::runtime_error("inc must be positive");
    }
    if (end <= start) {
        return 0;
    }
    return (end - start + inc - 1) / inc;
}

int iteration_index(const ParallelForTask& task, int logical_iteration) {
    return task.start + logical_iteration * task.inc;
}

void* parallel_for_worker(void* raw_ptr) {
    ThreadArgs* args = static_cast<ThreadArgs*>(raw_ptr);
    ParallelForTask& task = *args->task;

    if (task.schedule == Schedule::Dynamic) {
        while (true) {
            const int iter = task.next_iteration.fetch_add(1, std::memory_order_relaxed);
            if (iter >= args->iter_end) {
                break;
            }
            task.functor(iteration_index(task, iter), task.arg);
        }
        return nullptr;
    }

    for (int iter = args->iter_begin; iter < args->iter_end; ++iter) {
        task.functor(iteration_index(task, iter), task.arg);
    }
    return nullptr;
}

int parallel_for(int start,
                 int end,
                 int inc,
                 void* (*functor)(int, void*),
                 void* arg,
                 int num_threads,
                 Schedule schedule = Schedule::Static) {
    if (functor == nullptr) {
        throw std::runtime_error("functor must not be null");
    }
    if (!in_range(num_threads, 1, 16)) {
        throw std::runtime_error("num_threads must be in [1, 16]");
    }

    const int total_iters = iteration_count(start, end, inc);
    if (total_iters == 0) {
        return 0;
    }

    const int actual_threads = std::min(num_threads, total_iters);
    ParallelForTask task;
    task.start = start;
    task.end = end;
    task.inc = inc;
    task.functor = functor;
    task.arg = arg;
    task.schedule = schedule;

    std::vector<pthread_t> threads(static_cast<std::size_t>(actual_threads));
    std::vector<ThreadArgs> thread_args(static_cast<std::size_t>(actual_threads));

    for (int t = 0; t < actual_threads; ++t) {
        ThreadArgs& thread_arg = thread_args[static_cast<std::size_t>(t)];
        thread_arg.task = &task;

        if (schedule == Schedule::Dynamic) {
            thread_arg.iter_begin = 0;
            thread_arg.iter_end = total_iters;
        } else {
            const int base = total_iters / actual_threads;
            const int extra = total_iters % actual_threads;
            thread_arg.iter_begin = t * base + std::min(t, extra);
            thread_arg.iter_end = thread_arg.iter_begin + base + (t < extra ? 1 : 0);
        }

        const int rc = pthread_create(&threads[static_cast<std::size_t>(t)],
                                      nullptr,
                                      parallel_for_worker,
                                      &thread_arg);
        if (rc != 0) {
            throw std::runtime_error("pthread_create failed");
        }
    }

    for (pthread_t& thread : threads) {
        const int rc = pthread_join(thread, nullptr);
        if (rc != 0) {
            throw std::runtime_error("pthread_join failed");
        }
    }

    return actual_threads;
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
            "Usage: ./pthread_parallel_for_matmul <m> <n> <k> <threads(1-16)> "
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

void* matmul_row_functor(int row, void* raw_args) {
    MatmulArgs* args = static_cast<MatmulArgs*>(raw_args);
    const std::vector<double>& a = *args->a;
    const std::vector<double>& b = *args->b;
    std::vector<double>& c = *args->c;

    const std::size_t a_base = static_cast<std::size_t>(row) * args->n;
    const std::size_t c_base = static_cast<std::size_t>(row) * args->k;
    for (int p = 0; p < args->n; ++p) {
        const double a_ip = a[a_base + p];
        const std::size_t b_base = static_cast<std::size_t>(p) * args->k;
        for (int j = 0; j < args->k; ++j) {
            c[c_base + j] += a_ip * b[b_base + j];
        }
    }

    return nullptr;
}

void serial_matmul(const std::vector<double>& a,
                   const std::vector<double>& b,
                   std::vector<double>& c,
                   int m,
                   int n,
                   int k) {
    std::fill(c.begin(), c.end(), 0.0);
    for (int i = 0; i < m; ++i) {
        MatmulArgs args{&a, &b, &c, n, k};
        matmul_row_functor(i, &args);
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

        MatmulArgs matmul_args{&a, &b, &c, cfg.n, cfg.k};

        const auto t0 = std::chrono::steady_clock::now();
        const int actual_threads = parallel_for(0,
                                                cfg.m,
                                                1,
                                                matmul_row_functor,
                                                &matmul_args,
                                                cfg.threads,
                                                cfg.schedule);
        const auto t1 = std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        const double flops = 2.0 * static_cast<double>(cfg.m) * cfg.n * cfg.k;

        std::cout << "Task: Pthreads parallel_for matrix multiplication\n";
        std::cout << "A: " << cfg.m << " x " << cfg.n << '\n';
        std::cout << "B: " << cfg.n << " x " << cfg.k << '\n';
        std::cout << "C: " << cfg.m << " x " << cfg.k << '\n';
        std::cout << "Requested threads: " << cfg.threads << '\n';
        std::cout << "Actual threads: " << actual_threads << '\n';
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
