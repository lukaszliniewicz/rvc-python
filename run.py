#!/usr/bin/env python3
"""Prepare the RVC runtime and start its local FastAPI service."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rvc-run")

FAIRSEQ_WHEEL = (
    "https://huggingface.co/Jmica/rvc/resolve/main/"
    "fairseq-0.12.4-cp311-cp311-win_amd64.whl?download=true"
)
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
TORCH_PACKAGES = ("torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1")


def _pip_install(*args: str) -> None:
    command = [sys.executable, "-m", "pip", "install", *args]
    log.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def _torch_runtime_is_ready() -> bool:
    try:
        import torch
        import torchaudio
        import torchvision

        return (
            torch.__version__.startswith("2.3.1")
            and torchaudio.__version__.startswith("2.3.1")
            and torchvision.__version__.startswith("0.18.1")
            and torch.version.cuda == "12.1"
        )
    except Exception:
        return False


def prepare_runtime() -> None:
    if not importlib.metadata.version("pip").startswith("24."):
        _pip_install("pip==24.0")

    if importlib.util.find_spec("fairseq") is None:
        _pip_install(FAIRSEQ_WHEEL)

    if not _torch_runtime_is_ready():
        _pip_install(
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            *TORCH_PACKAGES,
            "--index-url",
            TORCH_INDEX,
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
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    prepare_runtime()
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
            args.device,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
