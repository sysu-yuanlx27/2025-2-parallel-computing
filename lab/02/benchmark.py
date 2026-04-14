#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Sequence

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


def run_case(binary: str, size: int, processes: int, comm: str, cwd: str) -> Dict[str, float]:
    command = [
        "mpirun",
        "--oversubscribe",
        "-np",
        str(processes),
        binary,
        str(size),
        str(size),
        str(size),
        "--comm",
        comm,
        "--no-print",
    ]
    output = run_cmd(command, cwd)
    return parse_metrics(output)


def build_markdown(
    results: Dict[str, Dict[int, Dict[int, Dict[str, float]]]],
    sizes: Sequence[int],
    processes: Sequence[int],
    comm_modes: Sequence[str],
) -> str:
    lines = []
    for comm in comm_modes:
        lines.append(f"### Communication: {comm}")
        lines.append("| Size | " + " | ".join(f"np={p}" for p in processes) + " |")
        lines.append("|---|" + "---:|" * len(processes))
        for size in sizes:
            cells = [str(size)]
            for process_count in processes:
                metrics = results[comm][size][process_count]
                cells.append(f"{fmt(metrics['time'], 6)} s, {fmt(metrics['gflops'])} GFLOPS")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MPI matmul v2 with different communication modes.")
    parser.add_argument("--binary", type=str, default="./build/matmul_mpi_v2", help="Path to the MPI binary")
    parser.add_argument("--sizes", type=str, default="128,256,512,1024,2048", help="Comma-separated matrix sizes")
    parser.add_argument("--processes", type=str, default="1,2,4,8,16", help="Comma-separated MPI process counts")
    parser.add_argument("--comm", type=str, default="p2p,collective", help="Comma-separated communication modes")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step")
    parser.add_argument("--save-md", type=str, default="", help="Optional markdown output path")
    args = parser.parse_args()

    cwd = os.path.dirname(os.path.abspath(__file__))
    binary = args.binary
    if not os.path.isabs(binary):
        binary = os.path.join(cwd, binary)

    sizes = parse_int_list(args.sizes)
    processes = parse_int_list(args.processes)
    comm_modes = [item.strip() for item in args.comm.split(",") if item.strip()]

    for comm in comm_modes:
        if comm not in ("p2p", "collective"):
            print(f"Error: unsupported communication mode: {comm}", file=sys.stderr)
            return 1

    if not args.skip_build:
        run_cmd(["make", "clean"], cwd)
        run_cmd(["make", "all"], cwd)

    if shutil.which("mpirun") is None:
        print("Error: mpirun not found", file=sys.stderr)
        return 1

    results: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {comm: {} for comm in comm_modes}

    for comm in comm_modes:
        for size in sizes:
            results[comm][size] = {}
            for process_count in processes:
                metrics = run_case(binary, size, process_count, comm, cwd)
                results[comm][size][process_count] = metrics
                print(
                    f"comm={comm} size={size} np={process_count}: "
                    f"time={fmt(metrics['time'], 6)} s, gflops={fmt(metrics['gflops'])}"
                )

    markdown = build_markdown(results, sizes, processes, comm_modes)
    print("\n" + markdown)

    if args.save_md:
        md_path = args.save_md
        if not os.path.isabs(md_path):
            md_path = os.path.join(cwd, md_path)
        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print(f"Saved markdown table to: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
