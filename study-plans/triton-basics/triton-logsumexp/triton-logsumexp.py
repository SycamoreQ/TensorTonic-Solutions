import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(axis=0) 
    j = tl.arange(0, BLOCK_SIZE)

    ptr = x_ptr + (i*x_row_stride) + j

    mask = j < n_cols

    row_data = tl.load(ptr, mask = mask, other=-float('inf'))

    m_i = tl.max(row_data, axis=0)
    sum = m_i + tl.log(tl.sum((tl.exp(row_data - m_i))))

    tl.store(out_ptr+i, sum)
    
def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )