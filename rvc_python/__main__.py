"""Command line entry points for RVC inference and the Pandrator RVC service."""

from __future__ import annotations

import sys
from argparse import ArgumentParser

import uvicorn

from rvc_python.api import create_app
from rvc_python.infer import RVCInference


def _resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _add_inference_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--models-dir", default="rvc_models", help="Directory containing RVC models")
    parser.add_argument("--device", default="auto", help="Device to use, or 'auto'")
    parser.add_argument("--method", default="rmvpe", choices=["harvest", "crepe", "rmvpe", "pm"])
    parser.add_argument("--index-rate", type=float, default=0.6)
    parser.add_argument("--filter-radius", type=int, default=3)
    parser.add_argument("--resample-sr", type=int, default=0)
    parser.add_argument("--rms-mix-rate", type=float, default=0.25)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--pitch", default=0, type=int)


def main() -> None:
    parser = ArgumentParser(description="RVC inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cli_parser = subparsers.add_parser("cli", help="Run one CLI conversion")
    cli_parser.add_argument("--input", required=True, help="Input audio path")
    cli_parser.add_argument("--output", default="out.wav", help="Output WAV path")
    cli_parser.add_argument("--model", required=True, help="RVC model name")
    _add_inference_arguments(cli_parser)

    api_parser = subparsers.add_parser("api", help="Start the Pandrator RVC service")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8050)
    _add_inference_arguments(api_parser)

    args = parser.parse_args()
    device = _resolve_device(args.device)
    rvc = RVCInference(models_dir=args.models_dir, device=device)
    rvc.set_params(
        f0method=args.method,
        f0up_key=args.pitch,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        resample_sr=args.resample_sr,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
    )

    if args.command == "cli":
        rvc.load_model(args.model)
        rvc.infer_file(args.input, args.output)
        print(f"Processed file saved to: {args.output}")
        return

    app = create_app(rvc, device=device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
