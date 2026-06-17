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
