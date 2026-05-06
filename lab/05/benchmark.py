#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
import subprocess
import sys
from typing import Dict, List


TIME_RE = re.compile(r"Time \(sec\):\s*([0-9]+(?:\.[0-9]+)?)")
GFLOPS_RE = re.compile(r"GFLOPS:\s*([0-9]+(?:\.[0-9]+)?)")


def run_cmd(cmd: List[str], cwd: str) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def parse_metrics(output: str) -> Dict[str, float]:
    time_match = TIME_RE.search(output)
    gflops_match = GFLOPS_RE.search(output)
    if time_match is None or gflops_match is None:
        raise ValueError(f"Failed to parse metrics from output:\n{output}")
    return {
        "time": float(time_match.group(1)),
        "gflops": float(gflops_match.group(1)),
    }


def build_if_needed(root_dir: str, skip_build: bool) -> None:
    if skip_build:
        return
    run_cmd(
        ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
        root_dir,
    )
    run_cmd(["cmake", "--build", "build", "-j"], root_dir)


def executable_path(root_dir: str, program: str) -> str:
    path = os.path.join(root_dir, "build", program)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Executable not found: {path}")
    return path


def print_markdown(rows: List[Dict[str, object]]) -> None:
    print("| sample | program | m | n | k | threads | schedule | time (s) | GFLOPS |")
    print("|---:|---|---:|---:|---:|---:|---|---:|---:|")
    for row in rows:
        print(
            f"| {row['sample']} | {row['program']} | {row['m']} | {row['n']} | {row['k']} | "
            f"{row['threads']} | {row['schedule']} | {row['time']:.6f} | {row['gflops']:.3f} |"
        )


def save_csv(rows: List[Dict[str, object]], output_path: str) -> None:
    if not rows:
        return
    with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Lab 05 matrix multiplication with random m,n,k in [128, 2048]. "
            "Each executable randomly generates A(m x n), B(n x k), and computes C=A x B."
        )
    )
    parser.add_argument("--samples", type=int, default=5, help="Number of random matrix sizes")
    parser.add_argument("--seed", type=int, default=20260506, help="Random seed for m,n,k")
    parser.add_argument("--min-size", type=int, default=128, help="Lower bound for m,n,k")
    parser.add_argument("--max-size", type=int, default=2048, help="Upper bound for m,n,k")
    parser.add_argument("--threads", type=int, default=4, help="Thread count in [1, 16]")
    parser.add_argument(
        "--program",
        choices=["openmp", "pthread", "both"],
        default="both",
        help="Which implementation to benchmark",
    )
    parser.add_argument(
        "--schedules",
        type=str,
        default="default,static,dynamic",
        help="Comma-separated schedules: default,static,dynamic",
    )
    parser.add_argument("--verify", action="store_true", help="Run serial correctness check")
    parser.add_argument("--skip-build", action="store_true", help="Skip CMake configure/build")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path")
    args = parser.parse_args()

    if args.samples <= 0:
        print("Error: --samples must be positive", file=sys.stderr)
        return 1
    if args.min_size < 128 or args.max_size > 2048 or args.min_size > args.max_size:
        print("Error: size range must be within [128, 2048]", file=sys.stderr)
        return 1
    if args.threads < 1 or args.threads > 16:
        print("Error: --threads must be in [1, 16]", file=sys.stderr)
        return 1

    schedules = [item.strip() for item in args.schedules.split(",") if item.strip()]
    valid_schedules = {"default", "static", "dynamic"}
    if not schedules or any(schedule not in valid_schedules for schedule in schedules):
        print("Error: schedules must be chosen from default, static, dynamic", file=sys.stderr)
        return 1

    root_dir = os.path.dirname(os.path.abspath(__file__))
    build_if_needed(root_dir, args.skip_build)

    programs = []
    if args.program in ("openmp", "both"):
        programs.append(("openmp", executable_path(root_dir, "openmp_matmul")))
    if args.program in ("pthread", "both"):
        programs.append(("pthread", executable_path(root_dir, "pthread_parallel_for_matmul")))

    rng = random.Random(args.seed)
    rows: List[Dict[str, object]] = []

    for sample in range(1, args.samples + 1):
        m = rng.randint(args.min_size, args.max_size)
        n = rng.randint(args.min_size, args.max_size)
        k = rng.randint(args.min_size, args.max_size)

        for program_name, exe in programs:
            for schedule in schedules:
                cmd = [
                    exe,
                    str(m),
                    str(n),
                    str(k),
                    str(args.threads),
                    schedule,
                ]
                if args.verify:
                    cmd.append("--verify")
                metrics = parse_metrics(run_cmd(cmd, root_dir))
                rows.append(
                    {
                        "sample": sample,
                        "program": program_name,
                        "m": m,
                        "n": n,
                        "k": k,
                        "threads": args.threads,
                        "schedule": schedule,
                        "time": metrics["time"],
                        "gflops": metrics["gflops"],
                    }
                )

    print_markdown(rows)

    if args.csv:
        output_path = args.csv
        if not os.path.isabs(output_path):
            output_path = os.path.join(root_dir, output_path)
        save_csv(rows, output_path)
        print(f"\nSaved CSV: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
