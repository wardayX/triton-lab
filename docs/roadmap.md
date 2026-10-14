# Learning and project roadmap

The order matters. Each stage uses ideas from earlier ones. Do not start the next stage just because the code runs; first be able to explain the result and its limits.

## Stage 0 — Vector addition

**Goal:** understand the Triton execution model and basic GPU timing.

Learn: program IDs, offsets, block size, masks, load/store, correctness tests, CUDA events, warm-up, repeated measurements, medians, and why a custom kernel may not beat PyTorch on a simple operation.

Evidence: source code, a passing correctness test, and a benchmark run recorded with the GPU/software versions.

## Stage 1 — Matrix multiplication

**Goal:** see the difference between a straightforward implementation and a tiled implementation.

Learn: matrix shapes, 2D indexing, dot products, tiles, tile sizes, data reuse, and comparisons against `torch.matmul`. Test rectangular as well as square matrices. Verify numerical agreement before comparing speed.

Evidence: a Triton implementation, correctness tests, a size/configuration sweep, and a brief note explaining what improved or did not improve.

## Stage 2 — GPU memory and execution

**Goal:** explain where kernel time is going instead of changing settings blindly.

Learn: CPU vs GPU work, global memory and caches, registers, shared memory, warps, streaming multiprocessors, memory coalescing, occupancy, launch overhead, effective bandwidth, and the basics of a roofline model.

Evidence: change one parameter at a time and support observations with timing/profile data. Higher occupancy by itself is not proof of better performance.

## Stage 3 — Reductions and fused operations

**Goal:** understand why combining several operations can reduce memory traffic.

Implement and compare a stable softmax and then layer normalization. Compare a framework reference (which may use multiple GPU operations) with a custom fused Triton kernel. Test numerical error, edge cases, and several input shapes.

Evidence: reference comparison, tests, per-shape timing, and a short account of the trade-off.

## Stage 4 — Attention

**Goal:** connect kernel programming to Transformer inference.

Learn Q/K/V, attention scores, softmax, the causal mask, and why materializing a full attention matrix consumes memory. Study the online-softmax/tiled approach used by FlashAttention before attempting a small implementation. Compare with PyTorch's scaled-dot-product attention where compatible.

Evidence: correctness checks for causal and non-causal cases, memory/latency comparisons, and clear limits. Do not claim to reproduce FlashAttention unless the implementation and validation justify that claim.

## Stage 5 — CUDA C++

**Goal:** address the DevTech role's explicit interest in CUDA C/C++ kernels.

Implement a small kernel such as vector addition or reduction in CUDA C++, then compare it with the Triton and PyTorch versions under the same input and measurement conditions. Learn launch syntax, thread/block indexing, device memory, error checks, compilation, and synchronization as needed.

Evidence: source, build/run instructions, correctness tests, and benchmarks from a specified GPU. Colab may not expose a suitable CUDA compiler in every runtime; if unavailable, use an environment with a matching CUDA toolkit rather than pretending the code was tested.

## Stage 6 — Profiling

**Goal:** use evidence to choose the next optimization.

Learn `torch.profiler` first. Then use NVIDIA Nsight Compute or Nsight Systems when the runtime and permissions permit. Inspect kernel time, memory traffic, utilization, launch overhead, and the relevant limits for the workload.

Evidence: a saved profile or extracted metrics, a baseline comparison, and a written explanation of the dominant bottleneck.

## Stage 7 — Connected inference projects

Only after the kernel and benchmarking fundamentals are solid, begin the two follow-on projects:

- **FlexKV:** test KV-cache quantization and mixed precision. Start by reproducing a published baseline, then investigate whether an adaptive policy improves memory usage without unacceptable quality loss.
- **PagedKV:** build a small block allocator for KV-cache storage. Measure fragmentation and utilization first; only then add batching or sharing.

These are separate projects that reuse the skills from Triton-Lab. They should not be squeezed into the first milestone.

## Research habits for every stage

1. Keep the baseline unchanged and explain what it does.
2. Change one variable per experiment.
3. Check correctness before measuring performance.
4. Record device, software versions, input shape/dtype, configuration, units, repetitions, and results.
5. Report negative results and limitations as well as improvements.
6. Read the relevant upstream tutorial/paper and credit reused ideas.
7. Make commits as work progresses. Each commit should represent a change you can explain.
