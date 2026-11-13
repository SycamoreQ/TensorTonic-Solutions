import torch
import triton
import triton.language as tl


@triton.jit
def vectorized_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vectorized_add_kernel with a large BLOCK_SIZE for long arrays."""
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vectorized_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)