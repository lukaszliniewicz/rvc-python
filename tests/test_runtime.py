from pathlib import Path
from unittest.mock import patch

import pytest

import run
from rvc_python.configs.config import Config


def test_resolve_backend_honors_explicit_choice():
    assert run.resolve_backend("cpu") == "cpu"
    assert run.resolve_backend("cuda") == "cuda"


def test_resolve_backend_auto_uses_detected_hardware():
    with patch.object(run, "_nvidia_gpu_is_available", return_value=False):
        assert run.resolve_backend("auto") == "cpu"
    with patch.object(run, "_nvidia_gpu_is_available", return_value=True):
        assert run.resolve_backend("auto") == "cuda"


def test_fairseq_wheels_are_platform_specific_and_hash_pinned():
    windows = run.fairseq_wheel_for_platform("Windows", "AMD64")
    linux = run.fairseq_wheel_for_platform("Linux", "x86_64")

    assert "win_amd64.whl" in windows
    assert "linux_x86_64.whl" in linux
    assert "#sha256=" in windows
    assert "#sha256=" in linux


def test_fairseq_wheel_rejects_unsupported_platform():
    with pytest.raises(RuntimeError, match="Supported platforms"):
        run.fairseq_wheel_for_platform("Linux", "aarch64")


def test_prepare_runtime_installs_verified_platform_wheel_when_fairseq_is_missing():
    def version(package):
        return {"pip": "24.0", "numpy": "1.23.5"}[package]

    with patch.object(run.importlib.metadata, "version", side_effect=version), patch.object(
        run,
        "_fairseq_runtime_is_ready",
        return_value=False,
    ), patch.object(run, "_torch_runtime_is_ready", return_value=True), patch.object(
        run,
        "fairseq_wheel_for_platform",
        return_value="https://example.test/fairseq.whl#sha256=abc",
    ), patch.object(run, "_pip_install") as install, patch.object(run.subprocess, "run"):
        run.prepare_runtime("cpu")

    install.assert_called_once_with("--no-deps", "https://example.test/fairseq.whl#sha256=abc")


def test_cpu_device_is_normalized_and_uses_fp32():
    lib_dir = Path(__file__).parents[1] / "rvc_python"
    config = Config(str(lib_dir), "cpu:0")

    assert config.device == "cpu"
    assert config.is_half is False


def test_cuda_request_fails_when_cuda_is_unavailable():
    lib_dir = Path(__file__).parents[1] / "rvc_python"
    with patch("rvc_python.configs.config.torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            Config(str(lib_dir), "cuda:0")
