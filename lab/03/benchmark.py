#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

TIME_RE = re.compile(r"Time \(sec\):\s*([0-9]+(?:\.[0-9]+)?)")
GFLOPS_RE = re.compile(r"GFLOPS:\s*([0-9]+(?:\.[0-9]+)?)")
THROUGHPUT_RE = re.compile(r"Throughput \(Melems/s\):\s*([0-9]+(?:\.[0-9]+)?)")
SUM_RE = re.compile(r"Sum:\s*(-?[0-9]+)")


def run_cmd(cmd: Sequence[str], cwd: str) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def fmt(value: float, ndigits: int = 3) -> str:
    return f"{value:.{ndigits}f}"


def ensure_built(cwd: str, build_dir: str) -> None:
    abs_build = os.path.join(cwd, build_dir)
    cache = os.path.join(abs_build, "CMakeCache.txt")
    if not os.path.exists(cache):
        run_cmd(["cmake", "-S", ".", "-B", build_dir], cwd)
    run_cmd(["cmake", "--build", build_dir, "-j"], cwd)


def parse_matmul(output: str) -> Dict[str, float]:
    t = TIME_RE.search(output)
    g = GFLOPS_RE.search(output)
    if not t or not g:
        raise ValueError(f"Failed to parse matmul output:\n{output[:1200]}")
    return {"time": float(t.group(1)), "gflops": float(g.group(1))}


def parse_sum(output: str) -> Dict[str, float]:
    t = TIME_RE.search(output)
    thr = THROUGHPUT_RE.search(output)
    sm = SUM_RE.search(output)
    if not t or not thr or not sm:
        raise ValueError(f"Failed to parse array-sum output:\n{output[:1200]}")
    return {
        "time": float(t.group(1)),
        "throughput": float(thr.group(1)),
        "sum": float(sm.group(1)),
    }


def run_matmul(binary: str, size: int, threads: int, cwd: str) -> Dict[str, float]:
    out = run_cmd([binary, str(size), str(size), str(size), str(threads)], cwd)
    return parse_matmul(out)


def run_array_sum(binary: str, n: int, threads: int, cwd: str) -> Dict[str, float]:
    out = run_cmd([binary, str(n), str(threads)], cwd)
    return parse_sum(out)


def markdown_table(
    title: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    lines = [f"### {title}"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:" for _ in header[1:]]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def build_markdown(
    matmul_results: Dict[int, Dict[int, Dict[str, float]]],
    matmul_sizes: Sequence[int],
    sum_results: Dict[int, Dict[int, Dict[str, float]]],
    sum_sizes: Sequence[int],
    threads: Sequence[int],
) -> str:
    lines: List[str] = []
    lines.append("## Lab03 Benchmark Results")
    lines.append("")

    for size in matmul_sizes:
        rows: List[List[str]] = []
        for t in threads:
            m = matmul_results[size][t]
            rows.append([str(t), f"{fmt(m['time'], 6)}", fmt(m['gflops'])])
        lines.append(
            markdown_table(
                f"Pthreads Matrix Multiplication (m=n=k={size})",
                ["Threads", "Time (sec)", "GFLOPS"],
                rows,
            )
        )

    for n in sum_sizes:
        rows = []
        for t in threads:
            m = sum_results[n][t]
            rows.append([str(t), f"{fmt(m['time'], 6)}", fmt(m['throughput']), str(int(m['sum']))])
        lines.append(
            markdown_table(
                f"Pthreads Array Sum (n={n})",
                ["Threads", "Time (sec)", "Throughput (Melems/s)", "Sum"],
                rows,
            )
        )

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark lab03 pthreads programs.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory")
    parser.add_argument("--skip-build", action="store_true", help="Skip CMake build")

    parser.add_argument(
        "--matmul-sizes",
        default="128,256,512,1024,2048",
        help="Comma-separated sizes for m=n=k (range: 128..2048)",
    )
    parser.add_argument(
        "--sum-sizes",
        default="1048576,4194304,16777216,67108864,134217728",
        help="Comma-separated n values for array sum (range: 1M..128M)",
    )
    parser.add_argument(
        "--threads",
        default="1,2,4,8,16",
        help="Comma-separated thread counts (range: 1..16)",
    )
    parser.add_argument(
        "--save-md",
        default="",
        help="Optional path to save markdown benchmark table",
    )

    args = parser.parse_args()
    cwd = os.path.dirname(os.path.abspath(__file__))

    matmul_sizes = parse_int_list(args.matmul_sizes)
    sum_sizes = parse_int_list(args.sum_sizes)
    threads = parse_int_list(args.threads)

    for s in matmul_sizes:
        if s < 128 or s > 2048:
            print(f"Error: matmul size out of range [128,2048]: {s}", file=sys.stderr)
            return 1
    for n in sum_sizes:
        if n < (1 << 20) or n > (128 << 20):
            print(f"Error: array size out of range [1M,128M]: {n}", file=sys.stderr)
            return 1
    for t in threads:
        if t < 1 or t > 16:
            print(f"Error: thread count out of range [1,16]: {t}", file=sys.stderr)
            return 1

    if not args.skip_build:
        ensure_built(cwd, args.build_dir)

    matmul_bin = os.path.join(cwd, args.build_dir, "matmul_pthreads")
    sum_bin = os.path.join(cwd, args.build_dir, "array_sum_pthreads")

    if not os.path.exists(matmul_bin) or not os.path.exists(sum_bin):
        print("Error: binary not found, please build first.", file=sys.stderr)
        print(f"expected: {matmul_bin}", file=sys.stderr)
        print(f"expected: {sum_bin}", file=sys.stderr)
        return 1

    matmul_results: Dict[int, Dict[int, Dict[str, float]]] = {}
    for s in matmul_sizes:
        matmul_results[s] = {}
        for t in threads:
            metrics = run_matmul(matmul_bin, s, t, cwd)
            matmul_results[s][t] = metrics
            print(
                f"[matmul] size={s} threads={t}: "
                f"time={fmt(metrics['time'], 6)} sec, gflops={fmt(metrics['gflops'])}"
            )

    sum_results: Dict[int, Dict[int, Dict[str, float]]] = {}
    for n in sum_sizes:
        sum_results[n] = {}
        for t in threads:
            metrics = run_array_sum(sum_bin, n, t, cwd)
            sum_results[n][t] = metrics
            print(
                f"[sum] n={n} threads={t}: "
                f"time={fmt(metrics['time'], 6)} sec, throughput={fmt(metrics['throughput'])}, "
                f"sum={int(metrics['sum'])}"
            )

    md = build_markdown(matmul_results, matmul_sizes, sum_results, sum_sizes, threads)
    print("\n" + md)

    if args.save_md:
        save_path = args.save_md
        if not os.path.isabs(save_path):
            save_path = os.path.join(cwd, save_path)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved markdown: {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
