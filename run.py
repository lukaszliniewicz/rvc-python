#!/usr/bin/env python3
"""Prepare the RVC runtime and start its local FastAPI service."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rvc-run")

FAIRSEQ_WHEELS = {
    ("windows", "x86_64"): (
        "https://huggingface.co/Jmica/rvc/resolve/"
        "01b388e059df1218a5a7b48b91305b2e06fed030/"
        "fairseq-0.12.4-cp311-cp311-win_amd64.whl?download=true"
        "#sha256=ddb6d64db6d37b40d535b9ad7774395732937cbf564ed3adcacd486f9edd50a8"
    ),
    ("linux", "x86_64"): (
        "https://huggingface.co/JackismyShephard/ultimate-rvc/resolve/"
        "8ca15ee3b546bd7dd4725e88bb9a997181e4c298/"
        "fairseq-0.12.2-cp311-cp311-linux_x86_64.whl"
        "#sha256=81b5af664d23ea941175de10e1176da5113177a8668979829705eff27d4db54d"
    ),
}
TORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu121",
}
TORCH_PACKAGES = ("torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1")


def _normalized_machine(value: str | None = None) -> str:
    machine = str(value or platform.machine() or "").strip().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def fairseq_wheel_for_platform(system: str | None = None, machine: str | None = None) -> str:
    key = (str(system or platform.system()).strip().lower(), _normalized_machine(machine))
    try:
        return FAIRSEQ_WHEELS[key]
    except KeyError as exc:
        raise RuntimeError(
            f"RVC does not provide a verified Fairseq wheel for {key[0]} {key[1] or 'unknown'}. "
            "Supported platforms are Windows x86_64 and Linux x86_64."
        ) from exc


def _pip_install(*args: str) -> None:
    command = [sys.executable, "-m", "pip", "install", *args]
    log.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def _nvidia_gpu_is_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    except OSError:
        return False


def resolve_backend(requested_backend: str) -> str:
    backend = str(requested_backend or "auto").strip().lower()
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported RVC backend: {requested_backend}")
    if backend != "auto":
        return backend
    return "cuda" if _nvidia_gpu_is_available() else "cpu"


def _torch_runtime_is_ready(backend: str) -> bool:
    try:
        import torch
        import torchaudio
        import torchvision

        versions_ready = (
            torch.__version__.startswith("2.3.1")
            and torchaudio.__version__.startswith("2.3.1")
            and torchvision.__version__.startswith("0.18.1")
        )
        if not versions_ready:
            return False
        if backend == "cpu":
            return torch.version.cuda is None
        return torch.version.cuda == "12.1"
    except Exception:
        return False


def _fairseq_runtime_is_ready() -> bool:
    try:
        import fairseq
        from fairseq import checkpoint_utils

        return bool(fairseq and checkpoint_utils)
    except Exception:
        return False


def prepare_runtime(backend: str) -> None:
    if not importlib.metadata.version("pip").startswith("24."):
        _pip_install("pip==24.0")

    if not _fairseq_runtime_is_ready():
        _pip_install("--no-deps", fairseq_wheel_for_platform())

    if not _torch_runtime_is_ready(backend):
        _pip_install(
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            *TORCH_PACKAGES,
            "--index-url",
            TORCH_INDEXES[backend],
        )

    try:
        numpy_ready = importlib.metadata.version("numpy") == "1.23.5"
    except importlib.metadata.PackageNotFoundError:
        numpy_ready = False
    if not numpy_ready:
        _pip_install("--upgrade", "--force-reinstall", "numpy==1.23.5")

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import fairseq, numpy, rvc_python, torch, torchvision, torchaudio; "
                "assert numpy.__version__ == '1.23.5'"
            ),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="RVC service bootstrapper")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--models-dir", default=str(Path(__file__).parent / "rvc_models"))
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    backend = resolve_backend(args.backend)
    device = args.device or ("cpu" if backend == "cpu" else "cuda:0")
    if backend == "cpu" and not device.startswith("cpu"):
        raise ValueError("The CPU backend requires a CPU device.")
    if backend == "cuda" and not device.startswith("cuda"):
        raise ValueError("The CUDA backend requires a CUDA device.")

    log.info("Using RVC %s backend (%s).", backend.upper(), device)
    prepare_runtime(backend)
    if args.prepare_only:
        return 0

    return subprocess.call(
        [
            sys.executable,
            "-m",
            "rvc_python",
            "api",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--models-dir",
            args.models_dir,
            "--device",
            device,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
