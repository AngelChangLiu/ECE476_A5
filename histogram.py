from __future__ import annotations

import argparse
import dataclasses
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch

import student_kernel
from reference import check_implementation, generate_input, ref_kernel
from task import input_t, output_t
from utils import clear_l2_cache, set_seed


HERE = Path(__file__).resolve().parent
custom_kernel = student_kernel.custom_kernel


def _preload_custom_kernel() -> None:
    preload = getattr(student_kernel, "preload_custom_kernel", None)
    if preload is not None:
        preload()


@dataclasses.dataclass
class TestCase:
    args: dict[str, int]
    spec: str


@dataclasses.dataclass
class Stats:
    runs: int
    mean_ms: float
    std_ms: float
    best_ms: float
    worst_ms: float


def _parse_test_cases(path: Path) -> list[TestCase]:
    pattern = re.compile(r"\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*([+-]?[0-9]+)\s*")
    tests: list[TestCase] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        args: dict[str, int] = {}
        for part in line.split(";"):
            match = pattern.fullmatch(part)
            if match is None:
                raise ValueError(f"Invalid test case line: {line!r}")
            args[match.group(1)] = int(match.group(2))
        tests.append(TestCase(args=args, spec=line))
    return tests


def _clone_data(data):
    if isinstance(data, tuple):
        return tuple(_clone_data(x) for x in data)
    if isinstance(data, list):
        return [_clone_data(x) for x in data]
    if isinstance(data, dict):
        return {k: _clone_data(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.clone()
    return data


def _check_case(test: TestCase) -> tuple[bool, str]:
    print(f"Generating input for TestCase: {test.spec}")
    data = generate_input(**test.args)
    check_copy = _clone_data(data)
    print("Preloading custom kernel.")
    try:
        _preload_custom_kernel()
        torch.cuda.synchronize()
    except:
        print("Module load failed. Guess you won't need it.")
    
    print(f"Invoking custom kernel: {test.spec}")
    start = time.perf_counter()
    output = custom_kernel(data)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Invocation finished. Runtime: {end-start:.6f} seconds.")
    return check_implementation(check_copy, output)


def _stats(durations: list[float]) -> Stats:
    mean = sum(durations) / len(durations)
    if len(durations) == 1:
        std = 0.0
    else:
        variance = sum((duration - mean) ** 2 for duration in durations) / (len(durations) - 1)
        std = variance ** 0.5
    return Stats(
        runs=len(durations),
        mean_ms=mean,
        std_ms=std,
        best_ms=min(durations),
        worst_ms=max(durations),
    )


def _time_kernel(
    kernel: Callable[[input_t], output_t],
    data: input_t,
    max_runs: int,
    min_time_s: float,
) -> Stats:
    durations: list[float] = []
    started = time.perf_counter()
    for _ in range(max_runs):
        clear_l2_cache()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = kernel(data)
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end))
        if len(durations) >= 3 and time.perf_counter() - started >= min_time_s:
            break
    return _stats(durations)


def _benchmark_case(test: TestCase, max_runs: int, min_time_s: float) -> tuple[Stats, Stats]:
    data = generate_input(**test.args)
    _preload_custom_kernel()
    check_copy = _clone_data(data)
    output = custom_kernel(data)
    good, message = check_implementation(check_copy, output)
    if not good:
        raise RuntimeError(message)

    custom_stats = _time_kernel(custom_kernel, data, max_runs, min_time_s)
    reference_stats = _time_kernel(ref_kernel, data, max_runs, min_time_s)
    return custom_stats, reference_stats


def _run_profile(test: TestCase, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_prefix = output_dir / "histogram_profile"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{HERE}{os.pathsep}{env.get('PYTHONPATH', '')}"
    one_shot = (
        "import torch; "
        "from reference import generate_input; "
        "from histogram import custom_kernel, _preload_custom_kernel; "
        f"data=generate_input(**{test.args!r}); "
        "_preload_custom_kernel(); "
        "torch.cuda.synchronize(); "
        "custom_kernel(data); "
        "torch.cuda.synchronize(); "
    )
    cmd = [
        os.getenv("NCU_PATH", "ncu"),
        "--target-processes",
        "all",
        "--set",
        os.getenv("NCU_SET", "full"),
        "--clock-control",
        os.getenv("NCU_CLOCK_CONTROL", "none"),
        "--force-overwrite",
        "--export",
        str(export_prefix),
        sys.executable,
        "-c",
        one_shot,
    ]
    subprocess.run(cmd, check=True, env=env)
    print(f"wrote {export_prefix}.ncu-rep")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local histogram assignment.")
    parser.add_argument("mode", choices=["test", "benchmark", "profile"])
    parser.add_argument(
        "test_file",
        nargs="?",
        default=str(HERE / "test_cases" / "test.txt"),
        help="Path to a test case file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--min-time-s", type=float, default=1.0)
    parser.add_argument("--profile-output", default=str(HERE / "profile_data"))
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required for this assignment.")

    set_seed(args.seed)
    tests = _parse_test_cases(Path(args.test_file))

    if args.mode == "test":
        passed = True
        for idx, test in enumerate(tests):
            good, message = _check_case(test)
            status = "pass" if good else "fail"
            print(f"test {idx}: {status} ({test.spec})")
            if message:
                print(message)
            passed = passed and good
        return 0 if passed else 1

    if args.mode == "benchmark":
        for idx, test in enumerate(tests):
            stats, reference_stats = _benchmark_case(test, args.max_runs, args.min_time_s)
            speedup = reference_stats.mean_ms / stats.mean_ms
            print(
                f"benchmark {idx}: runs={stats.runs}, "
                f"mean={stats.mean_ms:.3f} ms, std={stats.std_ms:.3f} ms, "
                f"best={stats.best_ms:.3f} ms, worst={stats.worst_ms:.3f} ms "
                f"reference_mean={reference_stats.mean_ms:.3f} ms, "
                f"speedup={speedup:.2f}x "
                f"({test.spec})"
            )
        return 0

    _run_profile(tests[-1], Path(args.profile_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
