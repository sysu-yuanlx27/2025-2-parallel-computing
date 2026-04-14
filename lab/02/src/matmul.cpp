#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

struct Config {
    int m = 128;
    int n = 128;
    int k = 128;
    int print_mode = 0;  // 0: normal snippet, 1: full print, 2: no print
    int comm_mode = 1;   // 0: point-to-point, 1: collective
};

static bool in_range(int x) {
    return x >= 128 && x <= 2048;
}

static void fill_random(std::vector<double>& values, uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (double& value : values) {
        value = distribution(generator);
    }
}

static void print_matrix(const std::vector<double>& matrix, int rows, int cols, const std::string& name, bool full_print) {
    std::cout << name << " (" << rows << "x" << cols << "):" << '\n';
    const int row_limit = full_print ? rows : std::min(rows, 8);
    const int col_limit = full_print ? cols : std::min(cols, 8);

    for (int i = 0; i < row_limit; ++i) {
        for (int j = 0; j < col_limit; ++j) {
            std::cout << std::fixed << std::setprecision(4)
                      << matrix[static_cast<size_t>(i) * cols + j] << ' ';
        }
        if (!full_print && col_limit < cols) {
            std::cout << "...";
        }
        std::cout << '\n';
    }

    if (!full_print && row_limit < rows) {
        std::cout << "...\n";
    }
}

static void compute_partition(int rows, int ranks, std::vector<int>& row_counts, std::vector<int>& row_offsets) {
    row_counts.assign(ranks, 0);
    row_offsets.assign(ranks, 0);

    const int base_rows = rows / ranks;
    const int remainder = rows % ranks;

    int offset = 0;
    for (int rank = 0; rank < ranks; ++rank) {
        row_counts[rank] = base_rows + (rank < remainder ? 1 : 0);
        row_offsets[rank] = offset;
        offset += row_counts[rank];
    }
}

static void matmul_block(const std::vector<double>& local_a,
                         const std::vector<double>& b,
                         std::vector<double>& local_c,
                         int local_rows,
                         int n,
                         int k) {
    std::fill(local_c.begin(), local_c.end(), 0.0);

    for (int i = 0; i < local_rows; ++i) {
        const size_t a_base = static_cast<size_t>(i) * n;
        const size_t c_base = static_cast<size_t>(i) * k;
        for (int p = 0; p < n; ++p) {
            const double a_value = local_a[a_base + p];
            const size_t b_base = static_cast<size_t>(p) * k;
            for (int j = 0; j < k; ++j) {
                local_c[c_base + j] += a_value * b[b_base + j];
            }
        }
    }
}

static MPI_Datatype create_config_type() {
    MPI_Datatype config_type = MPI_DATATYPE_NULL;
    Config dummy;

    int block_lengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint displacements[5] = {0, 0, 0, 0, 0};
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    MPI_Aint base_addr = 0;
    MPI_Aint field_addr = 0;

    MPI_Get_address(&dummy, &base_addr);

    MPI_Get_address(&dummy.m, &field_addr);
    displacements[0] = field_addr - base_addr;

    MPI_Get_address(&dummy.n, &field_addr);
    displacements[1] = field_addr - base_addr;

    MPI_Get_address(&dummy.k, &field_addr);
    displacements[2] = field_addr - base_addr;

    MPI_Get_address(&dummy.print_mode, &field_addr);
    displacements[3] = field_addr - base_addr;

    MPI_Get_address(&dummy.comm_mode, &field_addr);
    displacements[4] = field_addr - base_addr;

    MPI_Type_create_struct(5, block_lengths, displacements, types, &config_type);
    MPI_Type_commit(&config_type);
    return config_type;
}

static Config parse_args(int argc, char** argv) {
    if (argc < 4) {
        throw std::runtime_error(
            "Usage: ./matmul_mpi_v2 <m> <n> <k> [--comm p2p|collective] [--full-print] [--no-print]");
    }

    Config config;
    config.m = std::stoi(argv[1]);
    config.n = std::stoi(argv[2]);
    config.k = std::stoi(argv[3]);

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--full-print") {
            config.print_mode = 1;
        } else if (arg == "--no-print") {
            config.print_mode = 2;
        } else if (arg == "--comm") {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value after --comm");
            }
            const std::string value = argv[++i];
            if (value == "p2p") {
                config.comm_mode = 0;
            } else if (value == "collective") {
                config.comm_mode = 1;
            } else {
                throw std::runtime_error("--comm must be one of: p2p, collective");
            }
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (!in_range(config.m) || !in_range(config.n) || !in_range(config.k)) {
        throw std::runtime_error("m, n, k must be in [128, 2048].");
    }

    return config;
}

static void distribute_and_collect_p2p(const Config& config,
                                       int world_rank,
                                       int world_size,
                                       const std::vector<int>& row_counts,
                                       const std::vector<int>& row_offsets,
                                       std::vector<double>& full_a,
                                       std::vector<double>& full_b,
                                       std::vector<double>& full_c,
                                       std::vector<double>& local_a,
                                       std::vector<double>& local_c) {
    const int local_rows = row_counts[world_rank];
    const size_t local_a_size = static_cast<size_t>(local_rows) * config.n;
    const size_t local_c_size = static_cast<size_t>(local_rows) * config.k;
    const size_t b_size = static_cast<size_t>(config.n) * config.k;

    if (world_rank == 0) {
        for (int rank = 1; rank < world_size; ++rank) {
            MPI_Send(full_b.data(), static_cast<int>(b_size), MPI_DOUBLE, rank, 10, MPI_COMM_WORLD);

            const int rows = row_counts[rank];
            const int offset = row_offsets[rank];
            MPI_Send(full_a.data() + static_cast<size_t>(offset) * config.n,
                     rows * config.n,
                     MPI_DOUBLE,
                     rank,
                     11,
                     MPI_COMM_WORLD);
        }

        if (local_a_size > 0) {
            std::copy(full_a.begin(), full_a.begin() + static_cast<std::ptrdiff_t>(local_a_size), local_a.begin());
        }
    } else {
        MPI_Recv(full_b.data(), static_cast<int>(b_size), MPI_DOUBLE, 0, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(local_a.data(), static_cast<int>(local_a_size), MPI_DOUBLE, 0, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    matmul_block(local_a, full_b, local_c, local_rows, config.n, config.k);

    if (world_rank == 0) {
        if (local_c_size > 0) {
            std::copy(local_c.begin(), local_c.end(), full_c.begin());
        }
        for (int rank = 1; rank < world_size; ++rank) {
            const int rows = row_counts[rank];
            const int offset = row_offsets[rank];
            MPI_Recv(full_c.data() + static_cast<size_t>(offset) * config.k,
                     rows * config.k,
                     MPI_DOUBLE,
                     rank,
                     12,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(local_c.data(), static_cast<int>(local_c_size), MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
    }
}

static void distribute_and_collect_collective(const Config& config,
                                              int world_rank,
                                              int world_size,
                                              const std::vector<int>& row_counts,
                                              const std::vector<int>& row_offsets,
                                              std::vector<double>& full_a,
                                              std::vector<double>& full_b,
                                              std::vector<double>& full_c,
                                              std::vector<double>& local_a,
                                              std::vector<double>& local_c) {
    (void)world_size;
    std::vector<int> a_counts(row_counts.size(), 0);
    std::vector<int> a_offsets(row_offsets.size(), 0);
    std::vector<int> c_counts(row_counts.size(), 0);
    std::vector<int> c_offsets(row_offsets.size(), 0);

    for (size_t rank = 0; rank < row_counts.size(); ++rank) {
        a_counts[rank] = row_counts[rank] * config.n;
        a_offsets[rank] = row_offsets[rank] * config.n;
        c_counts[rank] = row_counts[rank] * config.k;
        c_offsets[rank] = row_offsets[rank] * config.k;
    }

    const int local_rows = row_counts[world_rank];
    const size_t local_a_size = static_cast<size_t>(local_rows) * config.n;
    const size_t local_c_size = static_cast<size_t>(local_rows) * config.k;
    const size_t b_size = static_cast<size_t>(config.n) * config.k;

    MPI_Bcast(full_b.data(), static_cast<int>(b_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(world_rank == 0 ? full_a.data() : nullptr,
                 a_counts.data(),
                 a_offsets.data(),
                 MPI_DOUBLE,
                 local_a.empty() ? nullptr : local_a.data(),
                 static_cast<int>(local_a_size),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    matmul_block(local_a, full_b, local_c, local_rows, config.n, config.k);

    MPI_Gatherv(local_c.empty() ? nullptr : local_c.data(),
                static_cast<int>(local_c_size),
                MPI_DOUBLE,
                world_rank == 0 ? full_c.data() : nullptr,
                c_counts.data(),
                c_offsets.data(),
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Datatype config_type = create_config_type();

    Config config;

    try {
        if (world_rank == 0) {
            config = parse_args(argc, argv);
        }

        MPI_Bcast(&config, 1, config_type, 0, MPI_COMM_WORLD);

        std::vector<int> row_counts;
        std::vector<int> row_offsets;
        compute_partition(config.m, world_size, row_counts, row_offsets);

        const int local_rows = row_counts[world_rank];
        const size_t local_a_size = static_cast<size_t>(local_rows) * config.n;
        const size_t local_c_size = static_cast<size_t>(local_rows) * config.k;

        std::vector<double> full_a;
        std::vector<double> full_b(static_cast<size_t>(config.n) * config.k, 0.0);
        std::vector<double> full_c;

        if (world_rank == 0) {
            full_a.resize(static_cast<size_t>(config.m) * config.n);
            full_c.resize(static_cast<size_t>(config.m) * config.k);
            fill_random(full_a, 42);
            fill_random(full_b, 777);
        }

        std::vector<double> local_a(local_a_size, 0.0);
        std::vector<double> local_c(local_c_size, 0.0);

        MPI_Barrier(MPI_COMM_WORLD);
        const double start_time = MPI_Wtime();

        if (config.comm_mode == 0) {
            distribute_and_collect_p2p(config,
                                       world_rank,
                                       world_size,
                                       row_counts,
                                       row_offsets,
                                       full_a,
                                       full_b,
                                       full_c,
                                       local_a,
                                       local_c);
        } else {
            distribute_and_collect_collective(config,
                                              world_rank,
                                              world_size,
                                              row_counts,
                                              row_offsets,
                                              full_a,
                                              full_b,
                                              full_c,
                                              local_a,
                                              local_c);
        }

        const double local_elapsed = MPI_Wtime() - start_time;
        double elapsed = 0.0;
        MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (world_rank == 0) {
            const bool no_print = (config.print_mode == 2);
            const bool full_print = (config.print_mode == 1);

            if (!no_print) {
                print_matrix(full_a, config.m, config.n, "A", full_print);
                print_matrix(full_b, config.n, config.k, "B", full_print);
                print_matrix(full_c, config.m, config.k, "C", full_print);
            }

            const double flops = 2.0 * static_cast<double>(config.m) * config.n * config.k;
            const double gflops = flops / elapsed / 1e9;

            std::cout << "Version: MPI-v2-"
                      << (config.comm_mode == 0 ? "p2p" : "collective") << '\n';
            std::cout << "World size: " << world_size << '\n';
            std::cout << "Time (sec): " << std::fixed << std::setprecision(6) << elapsed << '\n';
            std::cout << "GFLOPS: " << std::fixed << std::setprecision(3) << gflops << '\n';
        }

        MPI_Type_free(&config_type);
        MPI_Finalize();
        return 0;
    } catch (const std::exception& error) {
        if (world_rank == 0) {
            std::cerr << "Error: " << error.what() << '\n';
        }
        MPI_Type_free(&config_type);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
