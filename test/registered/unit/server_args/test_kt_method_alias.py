import argparse

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_rawfp8_method_alias_normalizes_before_dummy_short_circuit(caplog) -> None:
    server_args = ServerArgs(model_path="dummy", kt_method="RAWFP8")

    assert server_args.kt_method == "FP8"
    assert "--kt-method RAWFP8 is deprecated; using FP8" in caplog.text


def test_rawfp8_cli_alias_is_admitted() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(
        ["--model-path", "dummy", "--kt-method", "RAWFP8"]
    )

    assert ServerArgs.from_cli_args(namespace).kt_method == "FP8"
