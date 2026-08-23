#%%
import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
# Number of SMs on the active GPU. The persistent kernel below caps its
# launch grid at this value: launching more programs than SMs only adds
# scheduling overhead, since the persistent loop already lets each
# program walk through as many output tiles as it needs.
NUM_SMS = torch.cuda.get_device_properties(DEVICE).multi_processor_count
# %%
# calculate each element of the result matrix separately by loading each element of A and B once
@triton.jit
def matrix_multiplication_kernel_naive(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    # Accumulate in fp32 for both fp16 and fp32 inputs. tl.store casts the
    # result to c_ptr's element type, so the output dtype matches the input.
    acc = tl.zeros([], dtype=tl.float32)
    for k in range(K):
        a_val = tl.load(a_ptr + row * a_row_stride + k * a_col_stride)
        b_val = tl.load(b_ptr + k * b_row_stride + col * b_col_stride)
        acc += a_val.to(tl.float32) * b_val.to(tl.float32)
    c_m_n_ptr = c_ptr + row * c_row_stride + col * c_col_stride
    tl.store(c_m_n_ptr, acc)

def matrix_multiplication_naive(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_kernel_naive[(M, N)](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
    )
    return c

# %%

# calculate the each element of the result matrix separately by loading blocks of A and B
@triton.jit
def matrix_multiplication_kernel_naive_blocked(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    acc = tl.zeros([], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a_start_ptr = a_ptr + row * a_row_stride
        b_start_ptr = b_ptr + col * b_col_stride

        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = k_offsets * a_col_stride
        b_offsets = k_offsets * b_row_stride

        a_ptrs = a_start_ptr + a_offsets
        b_ptrs = b_start_ptr + b_offsets

        a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        acc += tl.sum(a_vals * b_vals)

    c_m_n_ptr = c_ptr + row * c_row_stride + col * c_col_stride
    tl.store(c_m_n_ptr, acc)

def matrix_multiplication_naive_blocked(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    BLOCK_SIZE_K = 128
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_kernel_naive_blocked[(M, N)](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_K,
    )
    return c

# %%
@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=['M', 'N', 'K'],
            x_vals=[128, 256, 512, 768, 1024],
            line_arg='provider',
            line_vals=['triton_naive', 'triton_naive_blocked', 'torch'],
            line_names=['Naive', 'K-blocked', 'PyTorch'],
            styles=[('blue', '-'), ('orange', '-'), ('green', '-')],
            xlabel='M=N=K',
            ylabel='TFLOPS',
            y_log=True,
            plot_name=f'matmul_naive_vs_naiveblocked_{dtype_name}',
            args={'dtype': dtype},
        )
        for dtype, dtype_name in [(torch.float16, 'fp16'), (torch.float32, 'fp32')]
    ])
def benchmark_naive(M, N, K, provider, dtype):
    a = torch.randn((M, K), device=DEVICE, dtype=dtype)
    b = torch.randn((K, N), device=DEVICE, dtype=dtype)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'triton_naive':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_naive(a, b))
    elif provider == 'triton_naive_blocked':
        ms = triton.testing.do_bench(lambda: matrix_multiplication_naive_blocked(a, b))
    elif provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)

benchmark_naive.run(show_plots=True, print_data=True)

# %%

"""
====================================================================
MEMORY COALESCING
====================================================================

Coalescing means: when the 32 lanes of a warp issue a load, the
addresses they touch fall in the same (or a small number of) 128-byte
cache lines, so the hardware can satisfy the warp with one wide
transaction instead of 32 separate ones. The unit of coalescing is
THE PROGRAM, not the grid -- different programs are different warps,
so spatial locality across `program_id` does not buy coalescing.
What does buy coalescing is: inside one program, the innermost axis
of the tile you build with `tl.arange` / `make_block_ptr` should have
stride 1 in memory. Triton maps the innermost tile axis to lane id,
so stride-1 along that axis => one LDG.128 per warp.

Why the two earlier kernels are NOT coalesced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

matrix_multiplication_kernel_naive (single-element):
    Loads are scalar (no `tl.arange`), one element per program. There
    is no within-program vector to coalesce in the first place; the
    warp issues 32 independent narrow loads. Maximally uncoalesced
    by construction.

matrix_multiplication_kernel_naive_blocked (1D K-block):
    A is fine, B is the problem.
    - a_ptrs = a_start_ptr + k_offsets
        Stride 1 along K (row-major A) -> 128 contiguous fp16 elements
        per warp = 256 B = 2 sectors. COALESCED.
    - b_ptrs = b_start_ptr + k_offsets * b_row_stride
        Walks DOWN a column of B with stride N. 128 lanes hit 128
        different cache lines. FULLY UNCOALESCED gather on B.
    This is the real reason the next kernel is faster, not "loading
    blocks instead of scalars" -- it's flipping B's access from a
    stride-N column gather to a stride-1 row-contiguous read.

Why every kernel from naive_row_major onward IS coalesced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting with matrix_multiplication_kernel_naive_row_major, every
load and store is built so that its INNERMOST tile axis is the
contiguous memory axis (stride 1) for row-major inputs:
- A's inner tile axis is K, with a_col_stride == 1.
- B's inner tile axis is N, with b_col_stride == 1.
- C's inner tile axis is N, with c_col_stride == 1.
Triton lays the innermost axis along lane id, so each warp emits one
wide LDG / STG per tile row. Coalesced.

The block-pointer kernel makes this intent explicit via order=(1, 0)
on every make_block_ptr -- "axis 1 is the fastest-varying dim in
memory" -- which is the canonical way to declare the coalescing
contract to the compiler.

Caveat: strides are runtime args, so the compiler can't prove
a_col_stride == 1 at compile time. A free hint where you know the
input is contiguous is `tl.assume(a_col_stride == 1)` (and similarly
for B, C); it lets Triton commit to the widest vectorized load
form rather than a more conservative fallback.
"""

# Compute one BLOCK_SIZE_N-wide section of an output row per program.
# Programs covering different column blocks of the same row reload that row of A.
@triton.jit
def matrix_multiplication_kernel_naive_row_major(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_SIZE_N
    a_start_ptr = a_ptr + row * a_row_stride
    b_start_ptr = b_ptr

    n_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    n_mask = n_offsets < N

    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = k_offsets * a_col_stride # 1d array with length BLOCK_SIZE_K
        b_offsets = k_offsets[:, None] * b_row_stride + n_offsets[None, :] * b_col_stride # 2d array with shape [BLOCK_SIZE_K, BLOCK_SIZE_N]

        a_ptrs = a_start_ptr + a_offsets
        b_ptrs = b_start_ptr + b_offsets

        a_mask = k_mask
        b_mask = k_mask[:, None] & n_mask[None, :]

        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32) # shape [BLOCK_SIZE_K]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32) # shape [BLOCK_SIZE_K, BLOCK_SIZE_N]

        acc += tl.sum(a_vals[:, None] * b_vals, axis=0) # shape [BLOCK_SIZE_N]

    c_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    c_mask = c_offsets < N
    c_m_n_ptr = c_ptr + row * c_row_stride + c_offsets * c_col_stride
    tl.store(c_m_n_ptr, acc, mask=c_mask)

def matrix_multiplication_naive_row_major(a: torch.Tensor, b: torch.Tensor):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64
    grid_size = (M, triton.cdiv(N, BLOCK_SIZE_N))
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_kernel_naive_row_major[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c

# %%
@triton.jit
def matrix_multiplication_tiled_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_SIZE_M
    col = tl.program_id(1) * BLOCK_SIZE_N

    m_offsets = tl.arange(0, BLOCK_SIZE_M) + row
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    n_mask = n_offsets < N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = m_offsets[:, None] * a_row_stride + k_offsets[None, :] * a_col_stride # shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
        b_offsets = k_offsets[:, None] * b_row_stride + n_offsets[None, :] * b_col_stride # shape [BLOCK_SIZE_K, BLOCK_SIZE_N]

        a_ptrs = a_ptr + a_offsets
        b_ptrs = b_ptr + b_offsets

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a_vals, b_vals, acc)

    c_offsets = m_offsets[:, None] * c_row_stride + n_offsets[None, :] * c_col_stride
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)

def matrix_multiplication_tiled(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 128,
    block_size_n: int = 128,
    block_size_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    BLOCK_SIZE_M = block_size_m
    BLOCK_SIZE_N = block_size_n
    BLOCK_SIZE_K = block_size_k
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_tiled_kernel[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[
            256, 512, 1024, 2048, 4096,
            4608, 5120, 6144, 7168,
            8192, 16384,
        ],
        line_arg='provider',
        line_vals=[
            'triton_tiled_64_64_64',
            'triton_tiled_128_128_64',
            'torch',
        ],
        line_names=[
            '64x64x64',
            '128x128x64', 'PyTorch',
        ],
        styles=[
            ('blue', '-'),
            ('orange', '-'), ('green', '-'),
        ],
        xlabel='M=N=K',
        ylabel='TFLOPS',
        plot_name='matmul-tiled-vs-torch-fp16',
        args={},
    ))
def benchmark_tiled(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'triton_tiled_64_64_64':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_tiled(a, b, 64, 64, 64)
        )
    elif provider == 'triton_tiled_128_128_64':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_tiled(a, b, 128, 128, 64)
        )
    elif provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)

benchmark_tiled.run(show_plots=True, print_data=True)

# %%


"""
In tile matmul, each program computes one output tile C[m_tile, n_tile] of size BLOCK_SIZE_M x BLOCK_SIZE_N.
To compute this, the program reads
- One row strip of A: rows m_tile * BLOCK_SIZE_M to m_tile * BLOCK_SIZE_M + BLOCK_SIZE_M - 1, all K columns. Call this A_m
- One column strip of B: columns n_tile * BLOCK_SIZE_N to n_tile * BLOCK_SIZE_N + BLOCK_SIZE_N - 1, all K rows. Call this B_n

Two programs that share the same m_tile reads the same A_m, and two programs that share the same n_tile reads the same B_n.
L2 reuse comes from arranging programs so that ones close in time (close in pid) share A_m or B_n.


====================================================================
CONCRETE EXAMPLE: 6 x 8 grid of output tiles, GROUP_SIZE_M = 2
====================================================================
num_pid_m = 6, num_pid_n = 8, GROUP_SIZE_M = 2
=> num_pid_in_group = 2 * 8 = 16
=> 48 programs total, 3 groups of 16

Row-major mapping (what a 2D grid effectively gives us):

         n=0  n=1  n=2  n=3  n=4  n=5  n=6  n=7
   m=0 |   0    1    2    3    4    5    6    7
   m=1 |   8    9   10   11   12   13   14   15
   m=2 |  16   17   18   19   20   21   22   23
   m=3 |  24   25   26   27   28   29   30   31
   m=4 |  32   33   34   35   36   37   38   39
   m=5 |  40   41   42   43   44   45   46   47

Supergrouped mapping (column-major within each height-2 strip):

         n=0  n=1  n=2  n=3  n=4  n=5  n=6  n=7
   m=0 |   0    2    4    6    8   10   12   14   <- group 0
   m=1 |   1    3    5    7    9   11   13   15
   m=2 |  16   18   20   22   24   26   28   30   <- group 1
   m=3 |  17   19   21   23   25   27   29   31
   m=4 |  32   34   36   38   40   42   44   46   <- group 2
   m=5 |  33   35   37   39   41   43   45   47

Both schedules compute the exact same 48 tiles and produce identical
results. Only the order changes -- and therefore which inputs the GPU is
hammering at any given moment.


====================================================================
L2 TRACE: 4 SMs concurrent, L2 holds 6 strips (toy numbers)
====================================================================

ROW-MAJOR:
  wave 1  pids 0..3   tiles (0,0..3)   load A0 + B0..B3            -> 5 HBM
  wave 2  pids 4..7   tiles (0,4..7)   A0 hit, load B4..B7         -> 4 HBM
                                       (B0..B3 evicted to make room)
  wave 3  pids 8..11  tiles (1,0..3)   A1 new, B0..B3 EVICTED      -> 5 HBM
  wave 4  pids 12..15 tiles (1,4..7)   A1 hit, B4..B7 EVICTED      -> 4 HBM
  ... pattern repeats for m=2,3,4,5 -> ~9 HBM loads per m-row
  TOTAL: ~54 HBM loads

SUPERGROUPED (GROUP_SIZE_M=2):
  wave 1  pids 0..3   tiles (0,0)(1,0)(0,1)(1,1)
                                       load A0,A1 + B0,B1          -> 4 HBM
  wave 2  pids 4..7   tiles (0,2)(1,2)(0,3)(1,3)
                                       A0,A1 hit, load B2,B3       -> 2 HBM
  wave 3  pids 8..11  tiles (0,4)(1,4)(0,5)(1,5)
                                       A0,A1 hit, load B4,B5       -> 2 HBM
                                       (B0,B1 evicted)
  wave 4  pids 12..15 tiles (0,6)(1,6)(0,7)(1,7)
                                       A0,A1 hit, load B6,B7       -> 2 HBM
  group 0 done -> 10 HBM loads. Groups 1 and 2 mirror this.
  TOTAL: ~30 HBM loads

Same 48 output tiles, ~45% less HBM (High-Bandwidth Memory, GPUs main DRAM) traffic.
Tensor cores spend less time stalled waiting for inputs -- that's where the TFLOPS gain comes from.


====================================================================
WHY COLUMN-MAJOR WITHIN A GROUP (and not row-major)?
====================================================================
The GPU dispatches programs in increasing pid order. So the first wave
of concurrent programs is always a contiguous prefix of pids. The
question is: which output tiles should that prefix cover?

With COLUMN-MAJOR within a group, the first GROUP_SIZE_M pids step
through one column of the group (different m, same n). They all share
the SAME B_n strip. Then the next GROUP_SIZE_M pids do the next column
and reuse the GROUP_SIZE_M A-strips already in L2. Both axes stay
narrow over the lifetime of the group -> tight working set.

With ROW-MAJOR within a group, the first num_pid_n pids would sweep
across the entire N axis (same m, different n). That immediately puts
num_pid_n distinct B-strips into the wave -- which is exactly the
wide-and-flat working set that the supergrouping trick was supposed to
fix. We'd be back to the row-major problem, just nested one level
deeper.

So column-major within a group is what makes both A and B stay reusable
inside the L2 working set. That is the entire mechanism by which
supergrouping cuts HBM traffic.
"""


@triton.jit
def matrix_multiplication_tiled_kernel_supergrouped(
    a_ptr, b_ptr, c_ptr,
    M, N, K, # a is MxK, b is KxN, c is MxN
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    # Let us see how many blocks we have in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Groups are horizontal strips of programs, so they cover entire column space of C
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group # which group this program is in
    first_pid_m = group_id * GROUP_SIZE_M # where does this group start
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) # if the num_pid_m is not a multiple of GROUP_SIZE_M, then the last group will have fewer than GROUP_SIZE_M programs

    local_pid = pid % num_pid_in_group # 0 ... num_pid_in_group - 1
    pid_m = first_pid_m + (local_pid % group_size_m)
    pid_n = local_pid // group_size_m

    """
    Within a group, local_pid ranges over group_size_m x num_pid_n positions.
    We traverse those positions in column-major order.
    - local_pid % group_size_m is which row inside the group
    - local_pid // group_size_m is which column inside the group
    """

    # From here on, the computation is identical to matrix_multiplication_tiled_kernel.
    # Only the (pid_m, pid_n) -> output-tile mapping changes. Same work, different order.
    row = pid_m * BLOCK_SIZE_M
    col = pid_n * BLOCK_SIZE_N

    m_offsets = tl.arange(0, BLOCK_SIZE_M) + row
    m_mask = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_SIZE_N) + col
    n_mask = n_offsets < N

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        k_offsets = tl.arange(0, BLOCK_SIZE_K) + k
        k_mask = k_offsets < K

        a_offsets = m_offsets[:, None] * a_row_stride + k_offsets[None, :] * a_col_stride
        b_offsets = k_offsets[:, None] * b_row_stride + n_offsets[None, :] * b_col_stride

        a_ptrs = a_ptr + a_offsets
        b_ptrs = b_ptr + b_offsets

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a_vals, b_vals, acc)

    c_offsets = m_offsets[:, None] * c_row_stride + n_offsets[None, :] * c_col_stride
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def matrix_multiplication_tiled_supergrouped(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 64,
    block_size_n: int = 64,
    block_size_k: int = 64,
    group_size_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    BLOCK_SIZE_M = block_size_m
    BLOCK_SIZE_N = block_size_n
    BLOCK_SIZE_K = block_size_k
    GROUP_SIZE_M = group_size_m
    # 1D launch grid is required for the supergrouped pid -> (pid_m, pid_n) mapping
    grid_size = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_tiled_kernel_supergrouped[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[
            256, 512, 1024, 2048, 4096,
            4608, 5120, 6144, 7168,
            8192, 16384,
        ],
        line_arg='provider',
        line_vals=[
            'triton_supergrouped_64_64_64',
            'triton_supergrouped_128_128_64',
            'torch',
        ],
        line_names=[
            'Grouped 64x64x64',
            'Grouped 128x128x64',
            'PyTorch',
        ],
        styles=[
            ('blue', '-'),
            ('orange', '-'),
            ('green', '-'),
        ],
        xlabel='M=N=K',
        ylabel='TFLOPS',
        plot_name='matmul-supergrouped-vs-torch-fp16',
        args={},
    ))
def benchmark_tiled_supergrouped(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'triton_supergrouped_64_64_64':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_tiled_supergrouped(a, b, 64, 64, 64)
        )
    elif provider == 'triton_supergrouped_128_128_64':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_tiled_supergrouped(a, b, 128, 128, 64)
        )
    elif provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)


benchmark_tiled_supergrouped.run(show_plots=True, print_data=True)
# %%
@triton.jit
def matrix_multiplication_block_pointers_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    local_pid = pid % num_pid_in_group
    pid_m = first_pid_m + (local_pid % group_size_m)
    pid_n = local_pid // group_size_m

    a_block_ptr = tl.make_block_ptr(
        a_ptr,
        shape=(M, K),
        strides=(a_row_stride, a_col_stride),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        b_ptr,
        shape=(K, N),
        strides=(b_row_stride, b_col_stride),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        c_ptr,
        shape=(M, N),
        strides=(c_row_stride, c_col_stride),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_vals = tl.load(
            a_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )
        b_vals = tl.load(
            b_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )

        acc = tl.dot(a_vals, b_vals, acc)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


def matrix_multiplication_block_pointers(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 64,
    block_size_n: int = 64,
    block_size_k: int = 64,
    group_size_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    grid_size = (
        triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),
    )
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_block_pointers_kernel[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        block_size_m, block_size_n, block_size_k, group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# %%
@triton.jit
def matrix_multiplication_persistent_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    pid_start = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_total = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Each launched program computes every NUM_PROGRAMS-th output tile.
    for pid in tl.range(
        pid_start, num_pid_total, NUM_PROGRAMS, flatten=True
    ):
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        local_pid = pid % num_pid_in_group
        pid_m = first_pid_m + (local_pid % group_size_m)
        pid_n = local_pid // group_size_m

        a_block_ptr = tl.make_block_ptr(
            a_ptr,
            shape=(M, K),
            strides=(a_row_stride, a_col_stride),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            b_ptr,
            shape=(K, N),
            strides=(b_row_stride, b_col_stride),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1, 0),
        )
        c_block_ptr = tl.make_block_ptr(
            c_ptr,
            shape=(M, N),
            strides=(c_row_stride, c_col_stride),
            offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(1, 0),
        )

        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a_vals = tl.load(
                a_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )
            b_vals = tl.load(
                b_block_ptr, boundary_check=(0, 1), padding_option="zero"
            )

            acc = tl.dot(a_vals, b_vals, acc)

            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        tl.store(c_block_ptr, acc, boundary_check=(0, 1))


def matrix_multiplication_persistent(
    a: torch.Tensor,
    b: torch.Tensor,
    block_size_m: int = 64,
    block_size_n: int = 64,
    block_size_k: int = 64,
    group_size_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 3,
):
    assert a.ndim == 2 and b.ndim == 2, "expected two 2D matrices"
    M, K = a.shape
    K_b, N = b.shape
    assert K == K_b, "incompatible matrix dimensions"
    assert a.device == b.device, "A and B must be on the same device"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    assert a.dtype in (torch.float16, torch.float32), "only fp16 and fp32 are supported"
    num_programs = min(
        NUM_SMS,
        triton.cdiv(M, block_size_m) * triton.cdiv(N, block_size_n),
    )
    grid_size = (num_programs,)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    matrix_multiplication_persistent_kernel[grid_size](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
        block_size_m, block_size_n, block_size_k, group_size_m,
        NUM_PROGRAMS=num_programs,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[
            256, 512, 1024, 2048, 4096,
            4608, 5120, 6144, 7168,
            8192, 16384,
        ],
        line_arg='provider',
        line_vals=[
            'triton_block_pointers_64_64_64',
            'triton_persistent_64_64_64',
            'triton_block_pointers_128_128_64',
            'triton_persistent_128_128_64',
            'torch',
        ],
        line_names=[
            'Block ptr 64x64x64',
            'Persistent 64x64x64',
            'Block ptr 128x128x64',
            'Persistent 128x128x64',
            'PyTorch',
        ],
        styles=[
            ('blue', '-'),
            ('blue', '--'),
            ('orange', '-'),
            ('orange', '--'),
            ('green', '-'),
        ],
        xlabel='M=N=K',
        ylabel='TFLOPS',
        plot_name='matmul-block-pointers-vs-persistent-vs-torch-fp16',
        args={},
    ))
def benchmark_block_pointers_vs_persistent(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'triton_block_pointers_64_64_64':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_block_pointers(a, b, 64, 64, 64)
        )
    elif provider == 'triton_persistent_64_64_64':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_persistent(a, b, 64, 64, 64)
        )
    elif provider == 'triton_block_pointers_128_128_64':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_block_pointers(a, b, 128, 128, 64)
        )
    elif provider == 'triton_persistent_128_128_64':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_persistent(a, b, 128, 128, 64)
        )
    elif provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)


benchmark_block_pointers_vs_persistent.run(show_plots=True, print_data=True)
# %%
def get_autotune_configs():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
    ]


# Two decorators, in this order: triton.autotune wraps triton.jit. The
# outer decorator is what the call site actually invokes; it picks a
# Config, then forwards into the JIT'd inner kernel with the constexprs
# from that Config injected.
@triton.autotune(
    configs=get_autotune_configs(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_multiplication_autotuned_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    a_row_stride, b_row_stride, c_row_stride,
    a_col_stride, b_col_stride, c_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    local_pid = pid % num_pid_in_group
    pid_m = first_pid_m + (local_pid % group_size_m)
    pid_n = local_pid // group_size_m

    # Free perf hints: the integer-analysis pass uses these to drop sign
    # handling on address arithmetic.
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(a_row_stride > 0)
    tl.assume(a_col_stride > 0)
    tl.assume(b_row_stride > 0)
    tl.assume(b_col_stride > 0)
    tl.assume(c_row_stride > 0)
    tl.assume(c_col_stride > 0)

    a_block_ptr = tl.make_block_ptr(
        a_ptr,
        shape=(M, K),
        strides=(a_row_stride, a_col_stride),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        b_ptr,
        shape=(K, N),
        strides=(b_row_stride, b_col_stride),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        c_ptr,
        shape=(M, N),
        strides=(c_row_stride, c_col_stride),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_vals = tl.load(
            a_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )
        b_vals = tl.load(
            b_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )

        acc = tl.dot(a_vals, b_vals, acc)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


def matrix_multiplication_autotuned(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device=DEVICE, dtype=a.dtype)
    # The grid depends on BLOCK_SIZE_M and BLOCK_SIZE_N, which the autotuner
    # picks. So the grid is a callable that receives the chosen Config's
    # constexpr dict as `META` and returns the actual launch shape.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    # Note: BLOCK_SIZE_*, GROUP_SIZE_M are NOT passed here -- the autotuner
    # injects them from the winning Config.
    matrix_multiplication_autotuned_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), b.stride(0), c.stride(0),
        a.stride(1), b.stride(1), c.stride(1),
    )
    return c


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[
            256, 512, 1024, 2048, 4096,
            4608, 5120, 6144, 7168,
            8192, 16384,
        ],
        line_arg='provider',
        line_vals=['triton_autotuned', 'torch'],
        line_names=['Autotuned', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        xlabel='M=N=K',
        ylabel='TFLOPS',
        plot_name='matmul-autotuned-block-pointers-vs-torch-fp16',
        args={},
    ))
def benchmark_autotuned(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'triton_autotuned':
        ms = triton.testing.do_bench(
            lambda: matrix_multiplication_autotuned(a, b)
        )
    elif provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(a, b))

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms)


benchmark_autotuned.run(show_plots=True, print_data=True)

# %%
