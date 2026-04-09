#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

TIME_RE = re.compile(r"Time \(sec\):\s*([0-9]+(?:\.[0-9]+)?)")
GFLOPS_RE = re.compile(r"GFLOPS:\s*([0-9]+(?:\.[0-9]+)?)")


def run_cmd(cmd: Sequence[str], cwd: str) -> str:
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
    if not time_match or not gflops_match:
        raise ValueError(f"Failed to parse metrics from output:\n{output[:1000]}")
    return {
        "time": float(time_match.group(1)),
        "gflops": float(gflops_match.group(1)),
    }


def fmt(value: float, ndigits: int = 3) -> str:
    return f"{value:.{ndigits}f}"


def parse_int_list(text: str) -> List[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    return [int(item) for item in values]


def run_case(binary: str, size: int, processes: int, cwd: str) -> Dict[str, float]:
    command = [
        "mpirun",
        "--oversubscribe",
        "-np",
        str(processes),
        binary,
        str(size),
        str(size),
        str(size),
        "--no-print",
    ]
    output = run_cmd(command, cwd)
    return parse_metrics(output)


def build_markdown(results: Dict[int, Dict[int, Dict[str, float]]], sizes: Sequence[int], processes: Sequence[int]) -> str:
    lines = []
    lines.append("| Size | " + " | ".join(f"np={p}" for p in processes) + " |")
    lines.append("|---|" + "---:|" * len(processes))
    for size in sizes:
        cells = [str(size)]
        for process_count in processes:
            metrics = results[size][process_count]
            cells.append(f"{fmt(metrics['time'], 6)} s, {fmt(metrics['gflops'])} GFLOPS")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the MPI matrix multiplication implementation.")
    parser.add_argument("--binary", type=str, default="./matmul_mpi", help="Path to the MPI binary")
    parser.add_argument("--sizes", type=str, default="128,256,512,1024,2048", help="Comma-separated matrix sizes")
    parser.add_argument("--processes", type=str, default="1,2,4,8,16", help="Comma-separated MPI process counts")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step")
    parser.add_argument("--save-md", type=str, default="", help="Optional markdown output path")
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    binary = args.binary
    if not os.path.isabs(binary):
        binary = os.path.join(cwd, binary)

    sizes = parse_int_list(args.sizes)
    processes = parse_int_list(args.processes)

    if not args.skip_build:
        run_cmd(["make", "clean"], cwd)
        run_cmd(["make", "all"], cwd)

    if shutil.which("mpirun") is None:
        print("Error: mpirun not found", file=sys.stderr)
        return 1

    results: Dict[int, Dict[int, Dict[str, float]]] = {}
    for size in sizes:
        results[size] = {}
        for process_count in processes:
            metrics = run_case(binary, size, process_count, cwd)
            results[size][process_count] = metrics
            print(
                f"size={size} np={process_count}: time={fmt(metrics['time'], 6)} s, "
                f"gflops={fmt(metrics['gflops'])}"
            )

    markdown = build_markdown(results, sizes, processes)
    print()
    print(markdown)

    if args.save_md:
        md_path = args.save_md
        if not os.path.isabs(md_path):
            md_path = os.path.join(cwd, md_path)
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)
            handle.write("\n")
        print(f"Saved markdown table to: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
