# Pandrator RVC Service

This fork provides RVC inference as a local FastAPI service for Pandrator. It is
based on [daswer123/rvc-python](https://github.com/daswer123/rvc-python) and the
Windows Python 3.11 work from
[JarodMica/rvc-python](https://github.com/JarodMica/rvc-python).

## Run on Windows

`run.bat` creates and maintains dedicated Pixi environments with Python 3.11,
Fairseq, NumPy 1.23.5, and either CPU-only or CUDA 12.1 PyTorch 2.3.1.

```bat
run.bat --models-dir "C:\Pandrator\models\rvc"
run.bat --backend cpu --models-dir "C:\Pandrator\models\rvc"
run.bat --backend cuda --models-dir "C:\Pandrator\models\rvc"
```

The default `auto` backend uses CUDA when an NVIDIA GPU is available and CPU
otherwise. The service binds to `127.0.0.1:8050` by default. To prepare an
environment without starting the service:

```bat
run.bat --backend cpu --prepare-only
run.bat --backend cuda --prepare-only
```

## Run on Linux

Linux x86_64 uses the same isolated CPU or CUDA 12.1 environments through
`run.sh`. The launcher accepts an existing Pixi binary or downloads a pinned
Pixi release into the parent installation directory.

```bash
./run.sh --backend cpu --models-dir "$HOME/.local/share/pandrator/models/rvc"
./run.sh --backend cuda --models-dir "$HOME/.local/share/pandrator/models/rvc"
./run.sh --backend cpu --prepare-only
./run.sh --pixi-path /opt/pandrator/bin/pixi --backend cpu --prepare-only
```

`auto` selects CUDA only when `nvidia-smi` reports a working NVIDIA device;
otherwise it selects CPU. AMD GPUs currently use the CPU backend. Linux
aarch64 is rejected explicitly because the verified Python 3.11 Fairseq wheel
is only available for x86_64.

## Tests

```bash
pixi run --environment test test
```

## API

- `GET /health`
- `GET /v1/models`
- `POST /v1/models/refresh`
- `POST /v1/unload`
- `POST /v1/convert`

`POST /v1/convert` accepts multipart form data:

- `audio`: input audio file
- `model`: model directory name
- `pitch`
- `f0_method`
- `index_rate`
- `filter_radius`
- `volume_envelope`
- `protect`
- `resample_sr`

The response is an `audio/wav` body. Model loading, parameter changes, and
inference are serialized because the underlying RVC runtime is stateful.

Models use this layout:

```text
rvc_models/
  voice-name/
    voice-name.pth
    voice-name.index
```

## Python Usage

The underlying inference class remains available:

```python
from rvc_python.infer import RVCInference

rvc = RVCInference(models_dir="rvc_models", device="cpu")
rvc.load_model("voice-name")
rvc.infer_file("input.wav", "output.wav")
```
