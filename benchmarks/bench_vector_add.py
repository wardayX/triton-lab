"""Compare PyTorch and Triton vector addition on a CUDA GPU."""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from triton_lab.vector_add import vector_add_out  # noqa: E402


def median_time_ms(fn, warmup=10, repeats=100, rounds=7):
    """Return median milliseconds per call, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()

        samples.append(start.elapsed_time(end) / repeats)

    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2**20, help="number of float32 elements")
    parser.add_argument("--block-size", type=int, default=256, help="elements per Triton program")
    parser.add_argument("--save", action="store_true", help="save this run to results/vector_add_benchmark.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. In Colab, select Runtime > Change runtime type > GPU.")
    if args.size < 1:
        raise SystemExit("--size must be at least 1")

    device = torch.device("cuda")
    x = torch.randn(args.size, device=device, dtype=torch.float32)
    y = torch.randn_like(x)
    out_torch = torch.empty_like(x)
    out_triton = torch.empty_like(x)

    def run_torch():
        torch.add(x, y, out=out_torch)

    def run_triton():
        vector_add_out(x, y, out_triton, block_size=args.block_size)

    # Compile and run both paths before timing; compilation is not part of steady-state timing.
    run_torch()
    run_triton()
    torch.cuda.synchronize()

    if not torch.allclose(out_torch, out_triton, rtol=1e-5, atol=1e-6):
        raise RuntimeError("The PyTorch and Triton outputs do not match")

    torch_ms = median_time_ms(run_torch)
    triton_ms = median_time_ms(run_triton)
    ratio = torch_ms / triton_ms
    estimated_gbps = (args.size * 12) / (triton_ms * 1e-3) / 1e9

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("Elements:", args.size)
    print("Triton block size:", args.block_size)
    print("Correctness: PASS")
    print("PyTorch median: %.6f ms/call" % torch_ms)
    print("Triton median:  %.6f ms/call" % triton_ms)
    print("PyTorch / Triton: %.3fx" % ratio)
    print("Estimated Triton bandwidth: %.2f GB/s (algorithmic estimate, not a profiler counter)" % estimated_gbps)

    if args.save:
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / "vector_add_benchmark.csv"
        row = {
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "elements": args.size,
            "block_size": args.block_size,
            "torch_median_ms": torch_ms,
            "triton_median_ms": triton_ms,
            "torch_over_triton": ratio,
            "correctness": "PASS",
        }
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        print("Saved results to:", output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
