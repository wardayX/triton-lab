"""A small Triton vector-add kernel used for learning and benchmarking."""

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Each Triton program handles one block of elements.
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # The final block may extend past n, so those lanes must be masked.
    mask = offsets < n
    x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_values = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x_values + y_values, mask=mask)


def vector_add_out(x, y, out, block_size=256):
    """Add x and y into an existing output tensor on the GPU."""
    if not (x.is_cuda and y.is_cuda and out.is_cuda):
        raise ValueError("x, y, and out must be CUDA tensors")
    if x.ndim != 1 or y.ndim != 1 or out.ndim != 1:
        raise ValueError("x, y, and out must be one-dimensional")
    if x.shape != y.shape or x.shape != out.shape:
        raise ValueError("x, y, and out must have the same shape")
    if x.dtype != torch.float32 or y.dtype != torch.float32 or out.dtype != torch.float32:
        raise ValueError("this example currently supports float32 tensors only")
    if not (x.is_contiguous() and y.is_contiguous() and out.is_contiguous()):
        raise ValueError("x, y, and out must be contiguous")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n = x.numel()
    if n == 0:
        return out

    grid = (triton.cdiv(n, block_size),)
    _vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out


def vector_add(x, y, block_size=256):
    """Return a new tensor containing x + y using the Triton kernel."""
    out = torch.empty_like(x)
    return vector_add_out(x, y, out, block_size=block_size)
