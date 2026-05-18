#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

enum class CommMode { Direct, Packed };

struct Options {
    int n = 1 << 16;
    int iterations = 10;
    CommMode mode = CommMode::Packed;
};

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

double lcg(double& seed) {
    constexpr double modulus = 0.2147483647e10;
    seed = std::fmod(16807.0 * seed, modulus);
    return (seed - 1.0) / (modulus - 1.0);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) {
            options.n = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            options.iterations = std::stoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            const std::string mode = argv[++i];
            if (mode == "direct") {
                options.mode = CommMode::Direct;
            } else if (mode == "packed") {
                options.mode = CommMode::Packed;
            } else {
                throw std::invalid_argument("--mode must be direct or packed");
            }
        } else if (arg == "--help") {
            std::cout << "Usage: fft_mpi [--n power_of_two] [--iters count] "
                         "[--mode direct|packed]\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown or incomplete option: " + arg);
        }
    }
    if (!is_power_of_two(options.n)) {
        throw std::invalid_argument("N must be a positive power of two");
    }
    if (options.iterations < 1) {
        throw std::invalid_argument("iterations must be positive");
    }
    return options;
}

void exchange_block_direct(const std::vector<double>& local,
                           std::vector<double>& peer,
                           int partner,
                           MPI_Comm comm) {
    // Baseline path: communicate one complex number at a time.  This keeps the
    // algorithm correct while intentionally exposing the cost of many messages.
    for (std::size_t i = 0; i < local.size(); i += 2) {
        MPI_Sendrecv(local.data() + i, 2, MPI_DOUBLE, partner, 0,
                     peer.data() + i, 2, MPI_DOUBLE, partner, 0,
                     comm, MPI_STATUS_IGNORE);
    }
}

void exchange_block_packed(const std::vector<double>& local,
                           std::vector<double>& peer,
                           int partner,
                           MPI_Comm comm) {
    int packed_capacity = 0;
    MPI_Pack_size(static_cast<int>(local.size()), MPI_DOUBLE, comm, &packed_capacity);

    std::vector<char> send_buffer(static_cast<std::size_t>(packed_capacity));
    std::vector<char> recv_buffer(static_cast<std::size_t>(packed_capacity));
    int send_position = 0;
    MPI_Pack(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
             send_buffer.data(), packed_capacity, &send_position, comm);

    MPI_Sendrecv(send_buffer.data(), send_position, MPI_PACKED, partner, 0,
                 recv_buffer.data(), packed_capacity, MPI_PACKED, partner, 0,
                 comm, MPI_STATUS_IGNORE);

    int recv_position = 0;
    MPI_Unpack(recv_buffer.data(), packed_capacity, &recv_position,
               peer.data(), static_cast<int>(peer.size()), MPI_DOUBLE, comm);
}

void local_stage_dit(std::vector<double>& values, int half, double sign) {
    const int span = 2 * half;
    for (int base = 0; base < static_cast<int>(values.size() / 2); base += span) {
        for (int j = 0; j < half; ++j) {
            const double angle = sign * 2.0 * kPi * static_cast<double>(j) / span;
            const double wr = std::cos(angle);
            const double wi = std::sin(angle);
            const int even = 2 * (base + j);
            const int odd = 2 * (base + j + half);
            const double ur = values[even];
            const double ui = values[even + 1];
            const double vr = values[odd];
            const double vi = values[odd + 1];
            const double tr = wr * vr - wi * vi;
            const double ti = wi * vr + wr * vi;
            values[even] = ur + tr;
            values[even + 1] = ui + ti;
            values[odd] = ur - tr;
            values[odd + 1] = ui - ti;
        }
    }
}

void local_stage_dif(std::vector<double>& values, int half, double sign) {
    const int span = 2 * half;
    for (int base = 0; base < static_cast<int>(values.size() / 2); base += span) {
        for (int j = 0; j < half; ++j) {
            const double angle = sign * 2.0 * kPi * static_cast<double>(j) / span;
            const double wr = std::cos(angle);
            const double wi = std::sin(angle);
            const int even = 2 * (base + j);
            const int odd = 2 * (base + j + half);
            const double ar = values[even];
            const double ai = values[even + 1];
            const double br = values[odd];
            const double bi = values[odd + 1];
            const double dr = ar - br;
            const double di = ai - bi;
            values[even] = ar + br;
            values[even + 1] = ai + bi;
            values[odd] = wr * dr - wi * di;
            values[odd + 1] = wi * dr + wr * di;
        }
    }
}

void distributed_stage_dit(std::vector<double>& local,
                           int local_n,
                           int half,
                           double sign,
                           CommMode mode,
                           int rank,
                           MPI_Comm comm) {
    const int rank_stride = half / local_n;
    const int partner = rank ^ rank_stride;
    std::vector<double> peer(local.size());

    if (mode == CommMode::Packed) {
        exchange_block_packed(local, peer, partner, comm);
    } else {
        exchange_block_direct(local, peer, partner, comm);
    }

    const int global_begin = rank * local_n;
    const int span = 2 * half;
    for (int i = 0; i < local_n; ++i) {
        const int global_index = global_begin + i;
        const int j = global_index % half;
        const double angle = sign * 2.0 * kPi * static_cast<double>(j) / span;
        const double wr = std::cos(angle);
        const double wi = std::sin(angle);
        const bool lower_half = (global_index % span) < half;

        const double self_r = local[2 * i];
        const double self_i = local[2 * i + 1];
        const double peer_r = peer[2 * i];
        const double peer_i = peer[2 * i + 1];

        if (lower_half) {
            const double tr = wr * peer_r - wi * peer_i;
            const double ti = wi * peer_r + wr * peer_i;
            local[2 * i] = self_r + tr;
            local[2 * i + 1] = self_i + ti;
        } else {
            const double tr = wr * self_r - wi * self_i;
            const double ti = wi * self_r + wr * self_i;
            local[2 * i] = peer_r - tr;
            local[2 * i + 1] = peer_i - ti;
        }
    }
}

void distributed_stage_dif(std::vector<double>& local,
                           int local_n,
                           int half,
                           double sign,
                           CommMode mode,
                           int rank,
                           MPI_Comm comm) {
    const int rank_stride = half / local_n;
    const int partner = rank ^ rank_stride;
    std::vector<double> peer(local.size());

    if (mode == CommMode::Packed) {
        exchange_block_packed(local, peer, partner, comm);
    } else {
        exchange_block_direct(local, peer, partner, comm);
    }

    const int global_begin = rank * local_n;
    const int span = 2 * half;
    for (int i = 0; i < local_n; ++i) {
        const int global_index = global_begin + i;
        const int j = global_index % half;
        const double angle = sign * 2.0 * kPi * static_cast<double>(j) / span;
        const double wr = std::cos(angle);
        const double wi = std::sin(angle);
        const bool lower_half = (global_index % span) < half;

        const double self_r = local[2 * i];
        const double self_i = local[2 * i + 1];
        const double peer_r = peer[2 * i];
        const double peer_i = peer[2 * i + 1];

        if (lower_half) {
            local[2 * i] = self_r + peer_r;
            local[2 * i + 1] = self_i + peer_i;
        } else {
            const double dr = peer_r - self_r;
            const double di = peer_i - self_i;
            local[2 * i] = wr * dr - wi * di;
            local[2 * i + 1] = wi * dr + wr * di;
        }
    }
}

void fft_dit(std::vector<double>& local,
             int global_n,
             int local_n,
             double sign,
             CommMode mode,
             int rank,
             MPI_Comm comm) {
    for (int half = 1; half < global_n; half *= 2) {
        if (half < local_n) {
            local_stage_dit(local, half, sign);
        } else {
            distributed_stage_dit(local, local_n, half, sign, mode, rank, comm);
        }
    }
}

void fft_dif(std::vector<double>& local,
             int global_n,
             int local_n,
             double sign,
             CommMode mode,
             int rank,
             MPI_Comm comm) {
    for (int half = global_n / 2; half >= 1; half /= 2) {
        if (half < local_n) {
            local_stage_dif(local, half, sign);
        } else {
            distributed_stage_dif(local, local_n, half, sign, mode, rank, comm);
        }
    }
}

std::vector<double> make_input(int n) {
    std::vector<double> values(2 * n);
    double seed = 331.0;
    for (int i = 0; i < n; ++i) {
        values[2 * i] = lcg(seed);
        values[2 * i + 1] = lcg(seed);
    }
    return values;
}


}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    try {
        const Options options = parse_options(argc, argv);
        if (!is_power_of_two(size) || options.n % size != 0) {
            throw std::invalid_argument("process count must be a power of two dividing N");
        }

        const int local_n = options.n / size;
        std::vector<double> global_input;
        if (rank == 0) {
            global_input = make_input(options.n);
        }

        std::vector<double> local(2 * local_n);
        MPI_Scatter(global_input.data(), 2 * local_n, MPI_DOUBLE,
                    local.data(), 2 * local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        const std::vector<double> initial_local = local;

        fft_dif(local, options.n, local_n, +1.0, options.mode, rank, MPI_COMM_WORLD);
        fft_dit(local, options.n, local_n, -1.0, options.mode, rank, MPI_COMM_WORLD);

        std::vector<double> roundtrip;
        if (rank == 0) {
            roundtrip.resize(2 * options.n);
        }
        MPI_Gather(local.data(), 2 * local_n, MPI_DOUBLE,
                   roundtrip.data(), 2 * local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double rms_error = 0.0;
        if (rank == 0) {
            const std::vector<double>& expected = global_input;
            const std::vector<double>& actual = roundtrip;
            for (int i = 0; i < 2 * options.n; ++i) {
                const double diff = expected[i] - actual[i] / options.n;
                rms_error += diff * diff;
            }
            rms_error = std::sqrt(rms_error / options.n);
        }

        local = initial_local;
        MPI_Barrier(MPI_COMM_WORLD);
        const double begin = MPI_Wtime();
        for (int iter = 0; iter < options.iterations; ++iter) {
            fft_dif(local, options.n, local_n, +1.0, options.mode, rank, MPI_COMM_WORLD);
            fft_dit(local, options.n, local_n, -1.0, options.mode, rank, MPI_COMM_WORLD);
            for (double& value : local) {
                value /= options.n;
            }
        }
        const double elapsed = MPI_Wtime() - begin;
        double max_elapsed = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const double transforms = 2.0 * options.iterations;
            const double flops = transforms * 5.0 * options.n * std::log2(options.n);
            std::cout << "MPI FFT\n"
                      << "  N             : " << options.n << '\n'
                      << "  processes     : " << size << '\n'
                      << "  mode          : "
                      << (options.mode == CommMode::Packed ? "packed" : "direct") << '\n'
                      << "  iterations    : " << options.iterations << '\n'
                      << "  RMS error     : " << std::scientific << rms_error << '\n'
                      << "  total time(s) : " << std::fixed << std::setprecision(6)
                      << max_elapsed << '\n'
                      << "  time/call(s)  : " << max_elapsed / transforms << '\n'
                      << "  MFLOPS        : " << flops / 1.0e6 / max_elapsed << '\n';
        }
    } catch (const std::exception& error) {
        if (rank == 0) {
            std::cerr << "error: " << error.what() << '\n';
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
