import argparse
import math
import pickle
import time
import gc
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from collections import defaultdict, OrderedDict
import numpy as np
import os

from utils.test_tensors import TestTensors
from utils.attn_impl_wrappers import ATTN_IMPL_FACTORIES, RunConfig

from flash_attn.utils.benchmark import benchmark_forward
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn_interface import _flash_attn_forward, _flash_attn_varlen_forward


def terse_type_str(dtype: torch.dtype) -> str:
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float8_e4m3fn: "fp8",
    }.get(dtype, str(dtype))


def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def time_fwd(func: Callable, *args, **kwargs) -> float:
    time.sleep(1)
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean


def flops(
    seqlens_q: torch.Tensor, 
    headdim: int, 
    nheads_q: int, 
    causal: bool, 
    mode="fwd"
) -> float:
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * (seqlens_q ** 2).sum().item() * nheads_q * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop: float, time: float) -> float:
    return (flop / time / 1e12) if not math.isnan(time) else 0.0


@dataclass(frozen=True)
class ConfigKey:
    dim: int
    dtype: torch.dtype
    causal: bool
    headdim: int
    batch_size: int
    seqlen: int
    page_size: int

    def to_tuple(self):
        return (self.dim, self.dtype, self.causal, self.headdim, self.batch_size, self.seqlen, self.page_size)

    @classmethod
    def from_tuple(cls, tup):
        return cls(*tup)


def save_results(time_f: Dict, speed_f: Dict, output_path: str):
    """Save benchmark results to a pickle file."""
    results = {
        'time_forward': time_f,
        'speed_forward': speed_f,
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)


def benchmark_attention_methods(
    causal_vals: List[bool],
    head_dims: List[int],
    bs_seqlen_pairs: List[Tuple[int, int]],
    dtypes: List[torch.dtype],
    repeats: int,
    device: str,
    dropout_p: float,
    dims: List[int],
    methods: List[str],
    page_sizes: List[int],
    validate: bool = False,
    profile: bool = False,
    output_path: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    time_f = {}
    speed_f = {}

    method_factories = ATTN_IMPL_FACTORIES
    # +15 for possible page_size info
    max_method_width = max(len(m) for m in methods) + 16

    val_factory, val_supported_dtypes = ATTN_IMPL_FACTORIES["Pytorch"][:2]

    if profile:
        from torch.cuda import nvtx

    try:
        for dim, causal, headdim, (batch_size, seqlen) in \
            product(dims, causal_vals, head_dims, bs_seqlen_pairs):
            # Skip invalid combinations where headdim doesn't divide dim evenly
            if dim % headdim != 0:
                continue

            torch.cuda.empty_cache()
            nheads = dim // headdim

            config = RunConfig(
                causal=causal,
                dropout_p=dropout_p,
            )

            print(
                f"### causal={causal}, dim={dim}, headdim={headdim}, "
                f"batch_size={batch_size}, seqlen={seqlen} ###"
            )

            for dtype in dtypes:
                # Generate reference output once if validation is needed and not profiling
                if validate and not profile:
                    reference_tensors = TestTensors.generate(
                        dtype=dtype,
                        batch_size=batch_size,
                        max_seqlen_q=seqlen,
                        max_seqlen_kv=seqlen,
                        nheads_q=nheads,
                        nheads_kv=nheads,
                        headdim=headdim,
                        device=device,
                        page_size=page_sizes[0],  
                        randomize_page_order=True,
                    )
                    ref_output = val_factory(reference_tensors, config)()
                    del reference_tensors

                # For each page size, generate tensors once
                for ps in page_sizes:
                    tensors = TestTensors.generate(
                        dtype=dtype,
                        batch_size=batch_size,
                        max_seqlen_q=seqlen,
                        max_seqlen_kv=seqlen,
                        nheads_q=nheads,
                        nheads_kv=nheads,
                        headdim=headdim,
                        device=device,
                        page_size=ps,
                        randomize_page_order=True,
                    )

                    for method in methods:
                        factory_tuple = method_factories.get(method, None)
                        if factory_tuple is None:
                            print(f"Method {method} is not implemented.")
                            continue

                        if len(factory_tuple) == 2:
                            factory, suppoted_dtypes = factory_tuple
                            supports_page_size = None
                        else:
                            factory, suppoted_dtypes, supports_page_size = factory_tuple

                        if dtype not in suppoted_dtypes:
                            print(f"Method {method} does not support dtype {dtype}.")
                            continue

                        # Construct method name with page size if supported
                        method_name = method
                        if supports_page_size is not None:
                            method_name = f"{method} (page_size={ps})"

                        # Check paging support
                        # If no paging support and ps != page_sizes[0], skip
                        if supports_page_size is None and ps != page_sizes[0]:
                            continue
                        # If paging support is available, but doesn't support this ps, skip
                        if supports_page_size is not None and not supports_page_size(ps):
                            print(f"Skipping {method_name}, unsupported page_size")
                            continue



                        try:
                            fn = factory(tensors, config)
                        except Exception as e:
                            print(f"Cannot create {method_name} due to {e}")
                            continue

                        # Test run
                        try:
                            fn()
                        except Exception as e:
                            print(f"Cannot run {method_name} due to {e}")
                            del fn
                            continue

                        config_key = ConfigKey(dim, dtype, causal, headdim, batch_size, seqlen, ps)

                        if profile:
                            torch.cuda.synchronize()
                            nvtx_name = f"ATTN-{method}/"
                            id = nvtx.range_start(nvtx_name)
                            fn()
                            nvtx.range_end(id)
                            torch.cuda.synchronize()
                            time_f[(config_key, method_name)] = 0.0
                            speed_f[(config_key, method_name)] = 0.0
                            print(
                                f"{method_name.ljust(max_method_width)} "
                                f"({terse_type_str(dtype):<4}) "
                                f"ran once for profiling",
                            )
                        else:
                            # Actual benchmarking
                            forward_time = time_fwd(
                                fn, repeats=repeats, verbose=False)
                            time_f[(config_key, method_name)] = forward_time
                            forward_speed = efficiency(
                                flops(
                                    torch.tensor([seqlen] * batch_size),
                                    headdim,
                                    nheads,
                                    causal,
                                    mode="fwd",
                                ),
                                forward_time,
                            )
                            speed_f[(config_key, method_name)] = forward_speed
                            print(
                                f"{method_name.ljust(max_method_width)} "
                                f"({terse_type_str(dtype):<4}) "
                                f"fwd: {speed_f[(config_key, method_name)]:>6.2f} "
                                f"TFLOPs/s, {time_f[(config_key, method_name)] * 1e3:6.2f} ms",
                            )

                            if validate:
                                output = fn()
                                # For fp8 cudnn skip validation
                                if not (method == "cuDNN" and dtype == torch.float8_e4m3fn):
                                    tols = {
                                        torch.float16: 0.05,
                                        torch.bfloat16: 0.05,
                                        torch.float8_e4m3fn: 0.1,
                                    }
                                    torch.testing.assert_close(
                                        output, ref_output.to(output.dtype),
                                        atol=tols[dtype], rtol=tols[dtype]
                                    )

                        del fn
                    del tensors
                    gc.collect()
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if output_path:
            save_results(time_f, speed_f, output_path)
            print(f"Partial results saved to {output_path}")
        raise

    if output_path:
        save_results(time_f, speed_f, output_path)

    return time_f, speed_f


def parse_bs_seqlen_pairs(args, default: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Parse batch size and sequence length combinations from either:
    1. A list of 'batch_size,seqlen' strings
    2. Separate lists of batch sizes and sequence lengths
    """
    if getattr(args, 'bs_seqlen_pairs', None) is not None:
        if getattr(args, 'bss', None) is not None or getattr(args, 'seqlens', None) is not None:
            raise ValueError("Cannot specify both --bs-seqlen-pairs and (--bss, --seqlens)")
        return [tuple(map(int, s.split(','))) for s in args.bs_seqlen_pairs]  # type: ignore
    else:
        if getattr(args, 'bss', None) is None and getattr(args, 'seqlens', None) is None:
            return default
        elif getattr(args, 'bss', None) is None or getattr(args, 'seqlens', None) is None:
            raise ValueError("When using --bss and --seqlens, must specify both")
        return list(product(args.bss, args.seqlens))


def run_benchmark(args):
    methods = args.methods
    repeats = args.repeats
    device = args.device
    dtypes = [getattr(torch, dt) for dt in args.dtypes]
    causal_vals = [c == 'True' for c in args.causal]
    head_dims = args.head_dims
    dims = args.dims
    page_sizes = args.page_sizes

    bs_seqlen_pairs = parse_bs_seqlen_pairs(args, default=[
        (32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)])

    dropout_p = args.dropout_p

    torch.manual_seed(0)

    benchmark_attention_methods(
        causal_vals,
        head_dims,
        bs_seqlen_pairs,
        dtypes,
        repeats,
        device,
        dropout_p,
        dims,
        methods,
        page_sizes=page_sizes,
        validate=args.validate,
        profile=args.profile,
        output_path=args.output_path,
    )


def load_results(pickle_path):
    """Load results from pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def dtype_to_str(dtype):
    """Convert torch.dtype to string representation."""
    return str(dtype).split('.')[-1]


@dataclass
class BarPlotData:
    data: Dict[str, float] = field(default_factory=lambda: dict())
    color: Optional[str] = None
    hatch: Optional[str] = None


@dataclass
class SubplotSpec:
    data: Dict[str, BarPlotData] = field(default_factory=lambda: OrderedDict())
    title: str = None
    x_ticks: List[str] = field(default_factory=lambda: list())
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    
    def get_data_array(self, name):
        return [self.data[name].data.get(x, np.nan) for x in self.x_ticks]
    
    def unique_bars(self):
        return self.data.keys()


def create_plot_specs(
    results, by_total_tokens=False, merge_dtypes=False, group_by_seqlen=False
) -> List[SubplotSpec]:
    import matplotlib.pyplot as plt
    
    speed_results = results['speed_forward']

    hatch_patterns = {
        'float32': '\\\\',
        'bfloat16': '//',
        'float16': '',
        'float8_e4m3fn': 'x',
        None: ''
    }

    methods = []
    for ((_, m), _) in speed_results.items():
        if m not in methods:
            methods.append(m)

    default_colors = plt.get_cmap('Set3').colors
    color_scheme = {}
    for idx, method in enumerate(methods):
        color_scheme[method] = default_colors[idx % len(default_colors)]

    def get_subplot_key(c: ConfigKey):
        key = [("dim", c.dim), ("headdim", c.headdim), ("causal", c.causal)]
        if not merge_dtypes:
            key.append(("dtype", dtype_to_str(c.dtype)))
        if by_total_tokens:
            key.append(("total_tokens", c.batch_size * c.seqlen))
        elif group_by_seqlen:
            key.append(("seq_len", c.seqlen))
        else:
            key.append(("batch_size", c.batch_size))
        return tuple(key)

    num_to_str = lambda x: str(x) if x < 1000 else f"{x//1000}k"

    def get_x_tick(c: ConfigKey):
        if by_total_tokens:
            return f"b{num_to_str(c.batch_size)},s{num_to_str(c.seqlen)}"
        elif group_by_seqlen:
            return num_to_str(c.batch_size)
        else:
            return num_to_str(c.seqlen)
    
    if by_total_tokens:
        x_label = "Batch size, Sequence length"
    elif group_by_seqlen:
        x_label = "Batch size"
    else:
        x_label = "Sequence length"
    y_label = "Speed (TFLOPs/s)"

    def get_label_color_hatch(c: ConfigKey, method: str):
        color = color_scheme.get(method, '#808080')
        if merge_dtypes:
            hatch = hatch_patterns.get(dtype_to_str(c.dtype), '')
            label = f'{method} ({dtype_to_str(c.dtype)})'
        else:
            label = method
            hatch = ''
        return label, color, hatch

    subplot_specs = OrderedDict()

    for ((config_key, m), speed) in speed_results.items():
        subplot_key = get_subplot_key(config_key)
        if subplot_key not in subplot_specs:
            subplot_specs[subplot_key] = SubplotSpec(
                title=", ".join([f"{k}={v}" for k, v in subplot_key]),
                x_label=x_label,
                y_label=y_label)
        
        x_tick = get_x_tick(config_key)
        label, color, hatch = get_label_color_hatch(config_key, m)
        
        if x_tick not in subplot_specs[subplot_key].x_ticks:
            subplot_specs[subplot_key].x_ticks.append(x_tick)
        if label not in subplot_specs[subplot_key].data:
            subplot_specs[subplot_key].data[label] = BarPlotData()
        
        bar_plot_data = subplot_specs[subplot_key].data[label]
        bar_plot_data.data[x_tick] = speed
        bar_plot_data.color = color
        bar_plot_data.hatch = hatch
    
    return list(subplot_specs.values())


def add_value_labels(ax, bars, rotation=90):
    """Add value labels on top of bars with rotation."""
    max_height = max(bar.get_height() for bar in bars if not np.isnan(bar.get_height()))
    ax.set_ylim(top=max_height * 1.1)
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                    f'{int(round(height))}',
                    ha='center', va='bottom',
                    rotation=rotation,
                    fontsize=8)


def plot_spec(subplot_specs: List[SubplotSpec], output_path: str, n_cols: Optional[int]):
    import matplotlib.pyplot as plt
    
    n_plots = len(subplot_specs)

    if n_cols is None:
        n_cols = min(2 if n_plots <= 4 else 3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_plots == 1:
        axes = [axes]

    axes = axes.flatten() if n_plots > 1 else axes

    plt.rcParams['patch.linewidth'] = 1.0

    for i, subplot_spec in enumerate(subplot_specs):
        ax = axes[i]
        all_bars = []
        n_bars = len(subplot_spec.unique_bars())
        bar_width =  0.8 / n_bars
        
        for bar_id, name in enumerate(subplot_spec.unique_bars()):
            x_positions = [i - (n_bars-1)*bar_width/2 + bar_id*bar_width 
                for i in range(len(subplot_spec.x_ticks))]
            
            bars = ax.bar(x_positions, subplot_spec.get_data_array(name),
                          bar_width,
                          color=subplot_spec.data[name].color,
                          hatch=subplot_spec.data[name].hatch,
                          label=name,
                          edgecolor='#404040', linewidth=1.0)
            all_bars.extend(bars)

        ax.set_xticks(range(len(subplot_spec.x_ticks)))
        ax.set_xticklabels(subplot_spec.x_ticks, rotation=45, ha='right')

        add_value_labels(ax, all_bars)
        ax.set_ylabel(subplot_spec.y_label)
        ax.set_xlabel(subplot_spec.x_label)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_title(subplot_spec.title)

    # Handle legend
    handles, labels = axes[0].get_legend_handles_labels() if n_plots > 0 else ([], [])
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saving plot to: {os.path.abspath(output_path)}")
    plt.close()


def plot_results(args):
    plot_suffix = 'total_tokens' if args.by_total_tokens else \
        ('seqlen_grouped' if args.group_by_seqlen else 'batchsize_grouped')
    if args.merge_dtypes:
        plot_suffix += '_merged_dtypes'
    output_path = f'{args.output_prefix}_{plot_suffix}.png'
    
    results = load_results(args.pickle_file)
    subplot_specs = create_plot_specs(
        results, args.by_total_tokens, args.merge_dtypes, args.group_by_seqlen)
    plot_spec(subplot_specs, output_path, args.ncols)


def main():
    parser = argparse.ArgumentParser(description='Benchmark and plot attention implementations')
    subparsers = parser.add_subparsers(dest='subcommand')

    parser_run = subparsers.add_parser('run', help='Run benchmarks')
    parser_run.add_argument('--methods', nargs='+', default=['Pytorch', 'cuDNN', 'Flash3'],
                            choices=ATTN_IMPL_FACTORIES.keys(), help='Methods to benchmark')
    parser_run.add_argument('--repeats', type=int, default=20, help='Number of repeats for timing')
    parser_run.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser_run.add_argument(
        '--dtypes',
        nargs='+',
        default=['float16'],
        help='Data types to benchmark',
        choices=['float16', 'bfloat16', 'float8_e4m3fn'],
    )
    parser_run.add_argument(
        '--causal',
        nargs='+',
        type=str,
        default=['False', 'True'],
        help='Causal values',
    )
    parser_run.add_argument(
        '--head-dims',
        nargs='+',
        type=int,
        default=[64, 128, 256],
        help='Head dimensions to test',
    )
    parser_run.add_argument(
        '--dims',
        nargs='+',
        type=int,
        default=[2048],
        help='Total dimensions to test',
    )
    parser_run.add_argument(
        '--bs-seqlen-pairs',
        nargs='+',
        type=str,
        help='Batch size and sequence pairs to test (format: batch_size,seqlen)',
    )
    parser_run.add_argument(
        '--bss',
        nargs='+',
        type=int,
        help='Batch sizes to test (must be specified with --seqlens)',
    )
    parser_run.add_argument(
        '--seqlens',
        nargs='+',
        type=int,
        help='Sequence lengths to test (must be specified with --bss)',
    )
    parser_run.add_argument('--dropout_p', type=float, default=0.0, help='Dropout probability')
    parser_run.add_argument(
        '--output-path',
        type=str,
        help='Path to save benchmark results as pickle file',
    )
    parser_run.add_argument(
        '--validate',
        action='store_true',
        help='Whether to validate the outputs against the reference implementation',
    )
    parser_run.add_argument(
        '--profile',
        action='store_true',
        help='Run each kernel once with NVTX annotations for profiling',
    )
    parser_run.add_argument(
        '--page-sizes',
        nargs='+',
        type=int,
        default=[32],
        help='Page sizes to sweep over (for kernels that support paging).'
    )

    parser_plot = subparsers.add_parser('plot', help='Plot benchmark results')
    parser_plot.add_argument('pickle_file', type=str, help='Path to pickle file with benchmark results')
    parser_plot.add_argument('--output-prefix', type=str, default='attention_benchmark',
                          help='Prefix for output plot files')
    parser_plot.add_argument('--by-total-tokens', action='store_true',
                          help='Plot results grouped by total number of tokens (batch_size × seq_len)')
    parser_plot.add_argument('--merge-dtypes', action='store_true',
                          help='Merge different dtypes into the same plot using different patterns')
    parser_plot.add_argument('--group-by-seqlen', action='store_true',
                          help='When not using total tokens mode, group plots by sequence length instead of batch size')
    parser_plot.add_argument('--ncols', type=int,
                          help='Set ncols for the plot grid')

    args = parser.parse_args()

    if args.subcommand == 'run':
        run_benchmark(args)
    elif args.subcommand == 'plot':
        plot_results(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
