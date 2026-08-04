from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _BytesPerTokenPool:
    def __init__(self, value: int):
        self.value = value

    def get_bytes_per_token(self) -> int:
        return self.value


def test_selective_c128_exact_524k_physical_storage_delta() -> None:
    context_tokens = 524_288
    c128_page_size = 2
    c128_tokens = context_tokens // 128
    num_pages_with_allocator_padding = (
        c128_tokens + c128_page_size + 1
    ) // c128_page_size
    # The shipped Flash checkpoint has 20 logical C128 compressor layers.
    # There is one physical C128 pool allocation per such layer; the paired
    # C4 compressor layers use their own pool and must not be counted here.
    c128_layers = 20

    fp8_page_bytes = 1_728
    bf16_page_bytes = 2_048
    assert num_pages_with_allocator_padding == 2_049
    assert (
        num_pages_with_allocator_padding
        * (bf16_page_bytes - fp8_page_bytes)
        * c128_layers
        == 13_113_600
    )
    # The capacity planner deliberately excludes each pool's one padded page.
    assert context_tokens * ((1_024 - 864) / 128) * c128_layers == 13_107_200


def test_scheduler_init_info_reports_selective_c128_physical_contract() -> None:
    kv_pool = SimpleNamespace(
        use_int4_storage=False,
        use_int4_indexer_storage=False,
        use_selective_c128_bf16_storage=True,
        swa_kv_pool=_BytesPerTokenPool(584),
        c4_indexer_kv_pool=_BytesPerTokenPool(132),
        c128_kv_pool=_BytesPerTokenPool(1_024),
        kv_storage_mode="fp8_e4m3+sparse_c128_bfloat16",
    )
    scheduler = object.__new__(Scheduler)
    scheduler.max_total_num_tokens = 524_288
    scheduler.max_req_input_len = 524_287
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(token_to_kv_pool=kv_pool)
    )

    info = scheduler.get_init_info()
    assert info["dsv4_sm86_c128_bf16_storage"] is True
    assert info["dsv4_latent_kv_bytes_per_token"] == 584
    assert info["dsv4_c4_kv_bytes_per_token"] == 1_024
    assert info["dsv4_c4_indexer_bytes_per_token"] == 132
    assert info["dsv4_c128_kv_bytes_per_token"] == 1_024
    assert info["dsv4_kv_storage_mode"] == "fp8_e4m3+sparse_c128_bfloat16"
