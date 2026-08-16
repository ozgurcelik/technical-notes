# %%
import torch

import triton
import triton.language as tl

DEVICE = 'cuda'
# %%
@triton.jit
def add_kernel(
    x_ptr, # pointer to the input vector x
    y_ptr, # pointer to the input vector y
    output_ptr, # pointer to the output vector
    n_elements, # so that we will know where to stop
    BLOCK_SIZE: tl.constexpr, # the size of the block
):
    # identify which program we are running
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(output_ptr + offsets, out, mask=mask)
# %%
def add(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int = 32,
    num_warps: int = 4,
):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    num_blocks = triton.cdiv(n_elements, block_size)
    add_kernel[(num_blocks,)](
        x, y, output, n_elements, block_size, num_warps=num_warps
    )
    return output

# %%
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
# %%
BLOCK_SIZES = [16, 64, 256, 1024, 4096]
PROVIDERS = ['torch'] + [f'triton-{block_size}' for block_size in BLOCK_SIZES]
LINE_NAMES = ['Torch'] + [f'Triton (BLOCK_SIZE={block_size})' for block_size in BLOCK_SIZES]
COLORS = ['black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
STYLES = [(color, '-') for color in COLORS]

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 30, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=PROVIDERS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=STYLES,  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={'metric': 'gbps'},  # Values for function arguments not in `x_names` and `y_name`.
    ),
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 30, 1)],
        x_log=True,
        y_log=True,
        line_arg='provider',
        line_vals=PROVIDERS,
        line_names=LINE_NAMES,
        styles=STYLES,
        ylabel='Latency (us)',
        plot_name='vector-add-latency',
        args={'metric': 'latency'},
    ),
])
def benchmark(size, provider, metric):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    else:
        block_size = int(provider.rsplit('-', maxsplit=1)[1])
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y, block_size), quantiles=quantiles
        )

    if metric == 'gbps':
        gbps = lambda elapsed_ms: 3 * x.numel() * x.element_size() * 1e-9 / (elapsed_ms * 1e-3)
        # Throughput is inversely proportional to latency, so the error-bound order is reversed.
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    latency_us = lambda elapsed_ms: elapsed_ms * 1e3
    return latency_us(ms), latency_us(min_ms), latency_us(max_ms)
# %%
benchmark.run(print_data=True, show_plots=True)
# %%
WARP_STYLES = [('tab:blue', '-'), ('tab:orange', '-'), ('tab:green', '-')]

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 30, 1)],
        x_log=True,
        line_arg='num_warps',
        line_vals=[1, 2, 4],
        line_names=['1 warp/program', '2 warps/program', '4 warps/program'],
        styles=WARP_STYLES,
        ylabel='GB/s',
        plot_name='vector-add-block-64-warp-comparison',
        args={'block_size': 64},
    ),
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 30, 1)],
        x_log=True,
        line_arg='num_warps',
        line_vals=[1, 2, 4],
        line_names=['1 warp/program', '2 warps/program', '4 warps/program'],
        styles=WARP_STYLES,
        ylabel='GB/s',
        plot_name='vector-add-block-16-warp-comparison',
        args={'block_size': 16},
    ),
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 30, 1)],
        x_log=True,
        line_arg='num_warps',
        line_vals=[2, 4, 8],
        line_names=['2 warps/program', '4 warps/program', '8 warps/program'],
        styles=WARP_STYLES,
        ylabel='GB/s',
        plot_name='vector-add-block-1024-warp-comparison',
        args={'block_size': 1024},
    ),
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 30, 1)],
        x_log=True,
        line_arg='num_warps',
        line_vals=[4, 8],
        line_names=['4 warps/program', '8 warps/program'],
        styles=WARP_STYLES[:2],
        ylabel='GB/s',
        plot_name='vector-add-block-4096-warp-comparison',
        args={'block_size': 4096},
    ),
])
def benchmark_num_warps(size, num_warps, block_size):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: add(x, y, block_size, num_warps), quantiles=quantiles
    )
    gbps = lambda elapsed_ms: 3 * x.numel() * x.element_size() * 1e-9 / (elapsed_ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# %%
benchmark_num_warps.run(print_data=True, show_plots=True)
# %%
