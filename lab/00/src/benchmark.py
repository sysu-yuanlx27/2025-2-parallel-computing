#!/usr/bin/env python3
import argparse
import os
import re
import shutil
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
    t_match = TIME_RE.search(output)
    g_match = GFLOPS_RE.search(output)
    if not t_match or not g_match:
        raise ValueError(f"Failed to parse metrics from output:\n{output[:800]}")
    return {"time": float(t_match.group(1)), "gflops": float(g_match.group(1))}


def fmt(x: float, ndigits: int = 3) -> str:
    return f"{x:.{ndigits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lab0 matrix multiplication benchmarks")
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--skip-build", action="store_true", help="Skip make build step")
    parser.add_argument("--bin-o0", type=str, default="./matmul_O0", help="Path to O0 binary")
    parser.add_argument("--bin-o3", type=str, default="./matmul_O3", help="Path to O3 binary")
    parser.add_argument("--peak-gflops", type=float, default=0.0, help="Hardware peak GFLOPS for percentage column")
    parser.add_argument("--save-md", type=str, default="", help="Optional output markdown file path")
    parser.add_argument("--include-mkl", action="store_true", help="Attempt to build and run MKL version")
    args = parser.parse_args()

    for name, value in (("m", args.m), ("n", args.n), ("k", args.k)):
        if value < 512 or value > 2048:
            print(f"Error: {name} must be in [512, 2048]", file=sys.stderr)
            return 1

    cwd = os.path.dirname(os.path.abspath(__file__))

    if not args.skip_build:
        run_cmd(["make", "clean"], cwd)
        run_cmd(["make", "all"], cwd)

    bin_o0 = args.bin_o0
    bin_o3 = args.bin_o3
    if not os.path.isabs(bin_o0):
        bin_o0 = os.path.join(cwd, bin_o0)
    if not os.path.isabs(bin_o3):
        bin_o3 = os.path.join(cwd, bin_o3)

    results = []

    py_out = run_cmd(
        ["python3", "matmul.py", str(args.m), str(args.n), str(args.k), "--no-print"],
        cwd,
    )
    py_metrics = parse_metrics(py_out)
    results.append({"version": "1 Python", **py_metrics})

    v2_out = run_cmd(
        [bin_o0, str(args.m), str(args.n), str(args.k), "naive", "--no-print"],
        cwd,
    )
    v2_metrics = parse_metrics(v2_out)
    results.append({"version": "2 C/C++", **v2_metrics})

    v3_out = run_cmd(
        [bin_o0, str(args.m), str(args.n), str(args.k), "reorder", "--no-print"],
        cwd,
    )
    v3_metrics = parse_metrics(v3_out)
    results.append({"version": "3 Adjust Loop Order", **v3_metrics})

    v4_out = run_cmd(
        [bin_o3, str(args.m), str(args.n), str(args.k), "reorder", "--no-print"],
        cwd,
    )
    v4_metrics = parse_metrics(v4_out)
    results.append({"version": "4 Compiler Optimization", **v4_metrics})

    v5_out = run_cmd(
        [bin_o3, str(args.m), str(args.n), str(args.k), "unroll", "--no-print"],
        cwd,
    )
    v5_metrics = parse_metrics(v5_out)
    results.append({"version": "5 Loop Unrolling", **v5_metrics})

    if args.include_mkl:
        if shutil.which("make") is not None:
            try:
                run_cmd(["make", "matmul_mkl"], cwd)
                v6_out = run_cmd(
                    ["./matmul_mkl", str(args.m), str(args.n), str(args.k), "mkl", "--no-print"],
                    cwd,
                )
                v6_metrics = parse_metrics(v6_out)
                results.append({"version": "6 Intel MKL", **v6_metrics})
            except Exception as exc:
                print(f"Warning: MKL run skipped: {exc}", file=sys.stderr)

    base_time = results[0]["time"]

    lines = []
    lines.append(f"Size: m={args.m}, n={args.n}, k={args.k}")
    if args.peak_gflops > 0:
        lines.append(f"Peak GFLOPS: {fmt(args.peak_gflops, 3)}")
        lines.append("| Version | Time (sec) | Relative Speedup | Absolute Speedup | GFLOPS | Peak Performance (%) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
    else:
        lines.append("| Version | Time (sec) | Relative Speedup | Absolute Speedup | GFLOPS |")
        lines.append("|---|---:|---:|---:|---:|")

    prev_time = None
    for row in results:
        t = row["time"]
        rel = (prev_time / t) if prev_time else 1.0
        abs_speedup = base_time / t
        if args.peak_gflops > 0:
            peak_pct = (row["gflops"] / args.peak_gflops) * 100.0
            lines.append(
                f"| {row['version']} | {fmt(t, 6)} | {fmt(rel)} | {fmt(abs_speedup)} | {fmt(row['gflops'])} | {fmt(peak_pct)} |"
            )
        else:
            lines.append(
                f"| {row['version']} | {fmt(t, 6)} | {fmt(rel)} | {fmt(abs_speedup)} | {fmt(row['gflops'])} |"
            )
        prev_time = t

    out_text = "\n".join(lines)
    print(out_text)

    if args.save_md:
        md_path = args.save_md
        if not os.path.isabs(md_path):
            md_path = os.path.join(cwd, md_path)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(out_text)
            f.write("\n")
        print(f"Saved markdown table to: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
