import email
import os
import runpy
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"


def _project_metadata() -> dict[str, object]:
    with (PYTHON_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]


def _setup_call(monkeypatch, version: str | None) -> dict[str, object]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("setuptools.setup", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("SGLANG_BUILD_RUST_EXTS", "none")
    if version is None:
        monkeypatch.delenv("SGLANG_KT_VERSION", raising=False)
    else:
        monkeypatch.setenv("SGLANG_KT_VERSION", version)

    runpy.run_path(str(PYTHON_ROOT / "setup.py"), run_name="__main__")

    assert len(calls) == 1
    return calls[0]


def test_primary_package_metadata_matches_the_kt_runtime_contract() -> None:
    project = _project_metadata()
    assert project["name"] == "sglang-kt"

    requirements = [Requirement(value) for value in project["dependencies"]]
    by_name = {canonicalize_name(requirement.name): requirement for requirement in requirements}
    assert "transformers" not in by_name
    assert str(by_name["transformers-kt"].specifier) == "==5.6.0.post1"
    assert str(by_name["torch"].specifier) == "==2.9.1"
    assert str(by_name["torchaudio"].specifier) == "==2.9.1"
    assert str(by_name["torchvision"].specifier) == "==0.24.1"
    assert str(by_name["torchcodec"].specifier) == "==0.8.0"
    assert str(by_name["torchao"].specifier) == "==0.9.0"
    assert str(by_name["flashinfer-python"].specifier) == "==0.6.9"
    assert str(by_name["flashinfer-cubin"].specifier) == "==0.6.9"
    assert str(by_name["sglang-kernel"].specifier) == "==0.3.21"

    extras = project["optional-dependencies"]
    self_references = [
        Requirement(value)
        for values in extras.values()
        for value in values
        if canonicalize_name(Requirement(value).name) in {"sglang", "sglang-kt"}
    ]
    assert self_references
    assert {
        canonicalize_name(requirement.name) for requirement in self_references
    } == {"sglang-kt"}


def test_setup_uses_kt_release_version_only_when_supplied(monkeypatch) -> None:
    assert _setup_call(monkeypatch, "0.6.3.post1")["version"] == "0.6.3.post1"
    assert "version" not in _setup_call(monkeypatch, None)


def test_built_wheel_exposes_kt_name_version_and_dependencies(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(PYTHON_ROOT / "pyproject.toml", source / "pyproject.toml")
    shutil.copy2(PYTHON_ROOT / "setup.py", source / "setup.py")
    shutil.copy2(REPO_ROOT / "README.md", source / "README.md")
    shutil.copy2(REPO_ROOT / "LICENSE", source / "LICENSE")
    package = source / "sglang"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "SGLANG_BUILD_RUST_EXTS": "none",
            "SGLANG_KT_VERSION": "0.6.3.post1",
        }
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    wheels = tuple(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    assert wheels[0].name.startswith("sglang_kt-0.6.3.post1-")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_paths = [name for name in wheel.namelist() if name.endswith("/METADATA")]
        assert len(metadata_paths) == 1
        metadata = email.message_from_bytes(wheel.read(metadata_paths[0]))

    assert metadata["Name"] == "sglang-kt"
    assert metadata["Version"] == "0.6.3.post1"
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    requirement_names = {canonicalize_name(requirement.name) for requirement in requirements}
    assert "transformers-kt" in requirement_names
    assert "transformers" not in requirement_names
    assert "sglang" not in requirement_names
