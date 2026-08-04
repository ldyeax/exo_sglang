"""SM86 microbenchmark for the isolated DSV4 INT4 C4 indexer PoC.

This compares one B=1 decode scorer invocation against the architecture-neutral
packed-FP8 scorer.  It does not enable either implementation in the server.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch
import triton.testing
from sglang.kernels.ops.attention.dsv4.fp8_storage_indexer import (
    DSV4_INDEXER_PAGE_BYTES,
    fp8_storage_paged_mqa_logits_triton,
)
from sglang.kernels.ops.attention.dsv4.int4_c4_indexer_poc import (
    INT4_C4_POC_HEAD_DIM,
    INT4_C4_POC_NUM_HEADS,
    INT4_C4_POC_PAGE_BYTES,
    INT4_C4_POC_PAGE_SIZE,
    INT4_C4_POC_VALUE_BYTES,
    int4_c4_paged_mqa_logits_triton,
)

_FP8_VALUE_BYTES = INT4_C4_POC_PAGE_SIZE * INT4_C4_POC_HEAD_DIM
_DEFAULT_LENGTHS = (128 * 1024, 524288)


def _median_cuda_ms(
    function: Callable[[], object],
    *,
    warmup: int,
    repetitions: int,
) -> float:
    # Triton's arguments are durations in milliseconds, not call counts.  Its
    # time-based warmup is important for short indexer kernels because a few
    # eager calls do not raise an idle RTX 3090 to its steady application clock.
    return float(
        triton.testing.do_bench(
            function,
            warmup=warmup,
            rep=repetitions,
            return_mode="median",
        )
    )


def _make_case(
    capacity: int,
    live_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if capacity % INT4_C4_POC_PAGE_SIZE:
        raise ValueError("benchmark lengths must be multiples of the 64-token page")
    if not 0 < live_length <= capacity:
        raise ValueError("live length must be positive and no larger than capacity")
    num_pages = capacity // INT4_C4_POC_PAGE_SIZE
    generator = torch.Generator(device=device).manual_seed(117)

    query_bf16 = torch.randn(
        (batch_size, 1, INT4_C4_POC_NUM_HEADS, INT4_C4_POC_HEAD_DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_fp8 = query_bf16.to(torch.float8_e4m3fn)
    weights = (
        torch.randn(
            (batch_size, INT4_C4_POC_NUM_HEADS),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        / INT4_C4_POC_NUM_HEADS
    )
    lengths = torch.full(
        (batch_size,), live_length, dtype=torch.int32, device=device
    )
    page_table = (
        torch.arange(num_pages, dtype=torch.int32, device=device)[None, :]
        .expand(batch_size, -1)
        .contiguous()
    )

    int4_cache = torch.empty(
        (num_pages, INT4_C4_POC_PAGE_BYTES), dtype=torch.uint8, device=device
    )
    int4_cache[:, :INT4_C4_POC_VALUE_BYTES].random_(0, 256, generator=generator)
    int4_cache[:, INT4_C4_POC_VALUE_BYTES:].view(torch.bfloat16).fill_(0.125)

    fp8_cache = torch.empty(
        (num_pages, DSV4_INDEXER_PAGE_BYTES), dtype=torch.uint8, device=device
    )
    # E4M3FN byte values 0..126 avoid the two NaN encodings while exercising
    # the current byte-storage decode path.
    fp8_cache[:, :_FP8_VALUE_BYTES].random_(0, 127, generator=generator)
    fp8_cache[:, _FP8_VALUE_BYTES:].view(torch.float32).fill_(0.125)
    bf16_cache = torch.randn(
        (
            num_pages,
            INT4_C4_POC_PAGE_SIZE,
            INT4_C4_POC_HEAD_DIM,
        ),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )

    return {
        "query_bf16": query_bf16,
        "query_fp8": query_fp8,
        "weights": weights,
        "lengths": lengths,
        "page_table": page_table,
        "int4_cache": int4_cache,
        "fp8_cache": fp8_cache,
        "bf16_cache": bf16_cache,
        "int4_output": torch.empty(
            (batch_size, capacity), dtype=torch.float32, device=device
        ),
        "fp8_output": torch.empty(
            (batch_size, capacity), dtype=torch.float32, device=device
        ),
    }


def benchmark_length(
    capacity: int,
    *,
    batch_size: int,
    live_length: int,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> dict[str, float | int]:
    from sglang.srt.layers.attention.dsv4.indexer import (
        bf16_direct_paged_mqa_logits_tilelang,
    )

    case = _make_case(capacity, live_length, batch_size, device)

    def run_int4() -> torch.Tensor:
        return int4_c4_paged_mqa_logits_triton(
            case["query_bf16"],
            case["int4_cache"],
            case["weights"],
            case["lengths"],
            case["page_table"],
            None,
            capacity,
            clean_logits=False,
            out=case["int4_output"],
        )

    def run_fp8() -> torch.Tensor:
        return fp8_storage_paged_mqa_logits_triton(
            case["query_fp8"],
            case["fp8_cache"],
            case["weights"],
            case["lengths"],
            case["page_table"],
            None,
            capacity,
            clean_logits=False,
            out=case["fp8_output"],
        )

    def run_bf16() -> torch.Tensor:
        return bf16_direct_paged_mqa_logits_tilelang(
            case["query_bf16"],
            case["bf16_cache"],
            case["weights"],
            case["lengths"],
            case["page_table"],
            None,
            capacity,
            clean_logits=False,
        )

    # Prime both Triton specializations before timing either provider.
    run_int4()
    run_fp8()
    run_bf16()
    torch.cuda.synchronize()
    int4_ms = _median_cuda_ms(run_int4, warmup=warmup, repetitions=repetitions)
    fp8_ms = _median_cuda_ms(run_fp8, warmup=warmup, repetitions=repetitions)
    bf16_ms = _median_cuda_ms(run_bf16, warmup=warmup, repetitions=repetitions)
    scanned_tokens = batch_size * live_length
    return {
        "batch_size": batch_size,
        "capacity": capacity,
        "live_length": live_length,
        "int4_page_bytes": INT4_C4_POC_PAGE_BYTES,
        "fp8_page_bytes": DSV4_INDEXER_PAGE_BYTES,
        "bf16_page_bytes": (
            INT4_C4_POC_PAGE_SIZE
            * INT4_C4_POC_HEAD_DIM
            * torch.bfloat16.itemsize
        ),
        "int4_ms": int4_ms,
        "fp8_ms": fp8_ms,
        "bf16_ms": bf16_ms,
        "int4_vs_fp8": int4_ms / fp8_ms,
        "int4_vs_bf16": int4_ms / bf16_ms,
        "int4_scan_gtokens_per_second": scanned_tokens / int4_ms / 1.0e6,
        "fp8_scan_gtokens_per_second": scanned_tokens / fp8_ms / 1.0e6,
        "bf16_scan_gtokens_per_second": scanned_tokens / bf16_ms / 1.0e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=list(_DEFAULT_LENGTHS)
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--live-length",
        type=int,
        help="Runtime sequence length; defaults to each captured capacity",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup duration in ms")
    parser.add_argument(
        "--repetitions", type=int, default=200, help="Measurement duration in ms"
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        raise SystemExit("BF16 tensor cores (SM80+) are required")

    results = [
        benchmark_length(
            length,
            batch_size=args.batch_size,
            live_length=args.live_length or length,
            device=device,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        for length in args.lengths
    ]
    report = {
        "device": torch.cuda.get_device_name(device),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
