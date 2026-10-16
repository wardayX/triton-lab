# Recorded experiment results

These values were supplied by the executed notebook outputs in the project's Google Colab session. Treat them as observations from that run, not universal performance claims. The notebook outputs contain additional details and correctness checks.

## Environment

The notebooks reported:

- GPU: NVIDIA Tesla T4
- PyTorch: `2.11.0+cu130`
- Triton: `3.6.0`
- CUDA C++ extension notebook: CUDA 13.0 toolchain, `nvcc` at `/usr/local/cuda/bin/nvcc`, Ninja 1.13.2, Python 3.13.15, and g++ 13.3

Colab environments are ephemeral and may change. Re-run experiments only when a new measurement is needed, and record the new environment alongside it.

## 1. Vector addition

- Correctness: passed for 1,000,003 FP32 elements, including a partial final block.
- Benchmark input: `N = 1 << 20`, FP32.
- PyTorch median: **0.055092 ms/call**.
- Triton median: **0.053108 ms/call**.
- Triton estimated effective bandwidth: **236.93 GB/s**.

The bandwidth is estimated from 12 bytes per FP32 element (two reads and one write), not measured from a DRAM hardware counter.

## 2. Tiled matrix multiplication

- Correctness shape: `(127, 95) @ (95, 141)`, producing `(127, 141)`; correctness passed.
- Timed benchmark shape: **`(512, 512) @ (512, 512)`**, FP16 inputs, FP32 accumulation in the Triton kernel.
- PyTorch: **0.037466 ms/call**.
- Triton: **0.230024 ms/call**.
- Triton estimated throughput: **1.167 TFLOP/s**.

PyTorch was faster for this particular benchmark. The irregular shape was used for the correctness/partial-tile check; it was not the shape used for the timing values above.

## 3. Block-size tuning

Input: `N = 1 << 22` FP32 elements. The estimated bandwidth assumes two input reads and one output write per element.

| Block size | Median time (ms) | Estimated bandwidth (GB/s) |
|---:|---:|---:|
| 128 | 0.197266 | 255.15 |
| 256 | 0.199421 | 252.39 |
| 512 | 0.207464 | 242.60 |
| 1024 | 0.210328 | 239.30 |
| 2048 | 0.210448 | 239.16 |

Block size 128 was best among the tested settings in this run. This does not establish it as universally optimal.

## 4. Softmax and LayerNorm

- Correctness: softmax passed; largest row-sum error was **1.1920928955078125e-07**.
- Correctness: LayerNorm passed against the PyTorch reference.
- Softmax timing shape: `(256, 1024)`, FP32.
- PyTorch softmax: **0.015846 ms/call**.
- Triton softmax: **0.020959 ms/call**.

PyTorch was faster in this softmax timing run. The notebook demonstrates softmax and LayerNorm as separate examples; LayerNorm was not applied to the softmax output.

## 5. Attention prototype

- Input shape: batch 1, 2 heads, context length 128, head dimension 64.
- Correctness: passed with maximum absolute error **0.00048828125** under the notebook's configured tolerance.
- PyTorch reference: **0.143600 ms/call**.
- Triton prototype: **0.066598 ms/call**.
- Reference/Triton time ratio: **2.156x** for this comparison.

This compares a tiled online-softmax learning prototype against the notebook's simple PyTorch reference. It is not a comparison against production FlashAttention or all optimized PyTorch SDPA paths. The prototype is non-causal and does not implement all production features.

## 6. CUDA C++ comparison

Vector addition timings from the same notebook run:

| Implementation | Median time (ms/call) |
|---|---:|
| PyTorch | 0.054950 |
| Triton | 0.052101 |
| CUDA C++ | 0.051766 |

The results are close. This single run does not establish a generally faster programming model. The CUDA extension compiled and loaded in the recorded Colab session after Ninja was installed and a fresh extension name was used.

## 7. Profiler experiment

The vector-add notebook profiled 10 calls to each implementation using `torch.profiler` with CPU and CUDA activities, shape recording, and memory profiling.

| Profiler entry | Calls | CUDA time reported |
|---|---:|---:|
| PyTorch `aten::add` CUDA total | 10 | 574.542 µs |
| Triton `vector_add_kernel` | 10 | 509.713 µs |
| Labeled `pytorch_vector_add` region CUDA total | 1 region | 599.855 µs |
| Labeled `triton_vector_add` region CUDA total | 1 region | 527.825 µs |

Profiler rows are hierarchical and overlap; do not sum region totals and kernel totals. Profiler instrumentation adds overhead, so these values are diagnostic and should not replace the clean benchmark timings above. The trace was exported in Colab to `/content/triton_lab_profile.json`; that path is temporary unless the file is separately downloaded or copied to persistent storage.

## 8. Automated tests

Recorded output from the repository root:

```text
test_empty_input (test_vector_add.VectorAddTests.test_empty_input) ... ok
test_non_multiple_of_block_size (test_vector_add.VectorAddTests.test_non_multiple_of_block_size) ... ok

Ran 2 tests in 1.848s

OK
```

The tests cover an empty tensor and a length not divisible by the block size. They do not constitute exhaustive testing of every dtype, shape, GPU, or kernel.
