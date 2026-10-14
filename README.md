# Triton-Lab

A small collection of GPU-kernel experiments focused on correctness, benchmarking, and understanding performance. The notebooks are meant to be run in order in Google Colab with a GPU runtime.

This repository does **not** assume a custom kernel is faster than a framework implementation. Each comparison should be verified on the stated hardware and software versions.

## Notebooks

| Notebook | Topic | Main learning goal |
|---|---|---|
| `00_vector_add.ipynb` | Vector addition | Triton programs, offsets, masks, correctness, basic GPU timing |
| `01_matrix_multiplication.ipynb` | Tiled matrix multiplication | 2D output tiles, accumulation, data reuse, partial tiles |
| `02_memory_and_block_tuning.ipynb` | Memory/performance analysis | Controlled block-size experiment and estimated effective bandwidth |
| `03_fused_softmax_layernorm.ipynb` | Fused operations | Reductions, numerical stability, fewer intermediate writes |
| `04_attention_flashattention_concepts.ipynb` | Attention | Online softmax and tiled attention inspired by FlashAttention |
| `05_cuda_cpp_comparison.ipynb` | CUDA C++ comparison | CUDA thread/block indexing, extension compilation, correctness and timing |

## Run in Google Colab

1. Upload the repository to GitHub or download and extract it.
2. Open a notebook in Colab.
3. Select **Runtime → Change runtime type → GPU**.
4. Run cells from top to bottom.
5. Record the GPU name, PyTorch/Triton versions, input sizes, correctness outcome, and timing results.

Colab images change over time. The first cell of each notebook prints the installed software versions. Avoid changing the PyTorch/CUDA toolchain unless a dependency is actually missing and you have checked compatibility. The CUDA C++ extension notebook also needs `nvcc`; it reports when the compiler is unavailable.

## Benchmarking notes

- Warm-up calls are used before timing.
- CUDA events measure work on the GPU execution timeline, and the ending event is synchronized before reading elapsed time.
- Several rounds are measured; the median is reported to reduce the influence of outliers.
- Outputs are checked against a PyTorch reference before performance is discussed.
- Very short kernels can be sensitive to launch overhead, queueing, clock changes, and other processes. Treat these introductory measurements as comparisons under one stated setup, not universal rankings.
- Effective bandwidth is estimated from the operation's expected input/output bytes. It is not a hardware-counter measurement of DRAM traffic.

## Project status

The six notebooks form a complete learning path, but the project should not be described as fully validated until the notebooks have been run on a GPU and their correctness/tests and measurements have been reviewed. No benchmark results are pre-filled in this repository.

## Suggested work order

1. Complete and understand vector addition.
2. Complete matrix multiplication and verify partial-tile handling.
3. Record a block-size tuning table and explain the trend.
4. Verify softmax and layer normalization against PyTorch.
5. Explain the online-softmax attention algorithm and its limitations.
6. Build the CUDA extension when the environment supports it and compare all three paths.

## References

- [Triton documentation and tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## Notes

Use this repository as an experiment log. Keep measured outputs, source changes, and conclusions traceable. Do not claim speedups without including the machine/runtime and benchmark method.
