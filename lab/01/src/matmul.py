#!/usr/bin/env python3
import random
import sys
import time


def in_range(x: int) -> bool:
    return 512 <= x <= 2048


def rand_matrix(rows: int, cols: int, seed: int):
    random.seed(seed)
    return [[random.uniform(-1.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def matmul_naive(a, b, m: int, n: int, k: int):
    c = [[0.0 for _ in range(k)] for _ in range(m)]
    for i in range(m):
        for j in range(k):
            s = 0.0
            for p in range(n):
                s += a[i][p] * b[p][j]
            c[i][j] = s
    return c


def print_matrix(mat, rows: int, cols: int, name: str, full_print: bool):
    print(f"{name} ({rows}x{cols}):")
    r_lim = rows if full_print else min(rows, 8)
    c_lim = cols if full_print else min(cols, 8)

    for i in range(r_lim):
        row_vals = " ".join(f"{mat[i][j]:.4f}" for j in range(c_lim))
        suffix = " ..." if (not full_print and c_lim < cols) else ""
        print(row_vals + suffix)

    if not full_print and r_lim < rows:
        print("...")


def parse_args(argv):
    if len(argv) < 4:
        raise ValueError("Usage: python3 matmul.py <m> <n> <k> [--full-print] [--no-print]")

    m = int(argv[1])
    n = int(argv[2])
    k = int(argv[3])
    full_print = "--full-print" in argv[4:]
    no_print = "--no-print" in argv[4:]

    if not in_range(m) or not in_range(n) or not in_range(k):
        raise ValueError("m, n, k must be in [512, 2048].")

    return m, n, k, full_print, no_print


def main():
    try:
        m, n, k, full_print, no_print = parse_args(sys.argv)

        a = rand_matrix(m, n, 42)
        b = rand_matrix(n, k, 777)

        t0 = time.perf_counter()
        c = matmul_naive(a, b, m, n, k)
        t1 = time.perf_counter()

        if not no_print:
            print_matrix(a, m, n, "A", full_print)
            print_matrix(b, n, k, "B", full_print)
            print_matrix(c, m, k, "C", full_print)

        sec = t1 - t0
        flops = 2.0 * m * n * k
        gflops = flops / sec / 1e9

        print("Version: python")
        print(f"Time (sec): {sec:.6f}")
        print(f"GFLOPS: {gflops:.3f}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
