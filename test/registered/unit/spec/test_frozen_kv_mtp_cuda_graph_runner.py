from sglang.srt.model_executor.runner import kt_capture_buffers
from sglang.srt.speculative import frozen_kv_mtp_cuda_graph_runner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_frozen_kv_mtp_registers_expanded_kt_capture_sizes(monkeypatch):
    recorded_sizes = []

    monkeypatch.setattr(
        frozen_kv_mtp_cuda_graph_runner,
        "register_kt_capture_batch_sizes",
        lambda capture_batch_sizes: recorded_sizes.append(list(capture_batch_sizes)),
    )

    frozen_kv_mtp_cuda_graph_runner._register_kt_capture_batch_sizes(
        [1, 4], captured_request_width=5
    )

    assert recorded_sizes == [[5, 20]]


def test_kt_capture_batch_size_registration_accumulates(monkeypatch):
    recorded_sizes = []
    temp_buffer = (object(),)

    class FakeKTMoEWrapper:
        @staticmethod
        def get_capture_batch_sizes():
            return [6, 256]

        @staticmethod
        def set_capture_batch_sizes(capture_batch_sizes):
            recorded_sizes.append(capture_batch_sizes)

    class FakeKExpertsCPUBuffer:
        temp_bs = 5
        capture_buffers = {}

    FakeKExpertsCPUBuffer.temp_buffer = temp_buffer

    monkeypatch.setattr(kt_capture_buffers, "KTRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        kt_capture_buffers, "KTMoEWrapper", FakeKTMoEWrapper, raising=False
    )
    monkeypatch.setattr(
        kt_capture_buffers,
        "KExpertsCPUBuffer",
        FakeKExpertsCPUBuffer,
        raising=False,
    )

    kt_capture_buffers.register_kt_capture_batch_sizes([5, 256, 2048])

    assert recorded_sizes == [[5, 6, 256, 2048]]
    assert FakeKExpertsCPUBuffer.capture_buffers == {5: temp_buffer}
