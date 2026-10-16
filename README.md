# Triton-Lab

**A hands-on GPU-kernel learning project using PyTorch, Triton, and CUDA C++.**

Triton-Lab contains six executed Google Colab notebooks, a reusable vector-add implementation, a small test suite, a benchmark script, and recorded experiments. The goal is to learn how GPU kernels are written, checked, measured, and profiled—not to claim that every custom kernel beats a framework implementation.

## Project contents

| Notebook | Topic | What it demonstrates |
|---|---|---|
| [00 — Vector addition](notebooks/00_vector_add.ipynb) | Triton basics | Program IDs, offsets, masks, loads/stores, correctness, timing, and `torch.profiler` |
| [01 — Matrix multiplication](notebooks/01_matrix_multiplication.ipynb) | Tiled computation | 2D indexing, accumulation, tile sizes, and data reuse |
| [02 — Memory and block tuning](notebooks/02_memory_and_block_tuning.ipynb) | Performance tuning | Block-size sweep and estimated effective bandwidth |
| [03 — Softmax and LayerNorm](notebooks/03_fused_softmax_layernorm.ipynb) | Neural-network primitives | Reductions, numerical stability, normalization, and reference checks |
| [04 — Attention](notebooks/04_attention_flashattention_concepts.ipynb) | Online softmax | Tiled attention inspired by FlashAttention; a learning prototype, not a production replacement |
| [05 — CUDA C++ comparison](notebooks/05_cuda_cpp_comparison.ipynb) | Programming models | Compare PyTorch, Triton, and a compiled CUDA C++ vector-add extension |

Supporting files:

- `src/triton_lab/vector_add.py` — reusable vector-add implementation
- `tests/test_vector_add.py` — correctness tests for empty input and a partial final block
- `benchmarks/bench_vector_add.py` — standalone vector-add benchmark
- [`results/README.md`](results/README.md) — actual recorded measurements and caveats
- [`docs/roadmap.md`](docs/roadmap.md) — learning roadmap and possible follow-on projects

## Run in Google Colab

No local CUDA installation is required. Use a Colab GPU runtime.

1. Open a notebook above in GitHub and choose **Open in Colab**, or open Colab and upload the notebook.
2. In Colab, select **Runtime → Change runtime type → GPU**.
3. Run cells from top to bottom. The notebooks print the software versions and GPU detected by the runtime.
4. Keep the executed notebook if you want its outputs to be visible to reviewers. Colab's `/content` directory is temporary; save notebooks to Drive or download them before ending the runtime.

For the CUDA C++ comparison, the runtime must provide a compatible CUDA compiler (`nvcc`). The notebook records the compiler and extension build information. Colab runtimes and their installed toolchains can change over time.

## Automated tests

Run from the repository root in a Python environment with compatible PyTorch, Triton, and CUDA support:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

**Recorded test result:** 2 tests ran and passed in the Colab session used for this project. The tests cover empty input and a vector length that is not divisible by the block size. They are skipped when CUDA is unavailable, so check the test output rather than treating every successful process exit as proof that GPU tests ran.

## How to interpret the benchmarks

- Correctness is checked against PyTorch before timing is discussed.
- The notebooks use warm-up and repeated measurements; exact timing can vary with GPU model, runtime state, clocks, and software versions.
- Very small operations can be dominated by launch overhead.
- Estimated effective bandwidth is calculated from expected input/output bytes. It is not a hardware-counter measurement of DRAM traffic.
- The attention result compares the Triton prototype with the notebook's simple PyTorch reference, not with every optimized attention implementation.
- Profiler timings are diagnostic and include instrumentation overhead. Do not compare them as if they were clean benchmark timings.

## Project status

The core notebook path has been executed in Google Colab, the profiler example has produced a trace/table, and the two repository unit tests passed in the reported session. Recorded results are summarized in [`results/README.md`](results/README.md). These are learning experiments on a Tesla T4 with the software versions recorded in the notebooks, not a broad hardware evaluation or a production kernel library.

## References

- [Triton documentation and tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
