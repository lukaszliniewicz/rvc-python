"""FastAPI service contract used by Pandrator."""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from loguru import logger


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _refresh_models(rvc: Any) -> list[str]:
    return sorted(rvc.refresh_models(), key=str.casefold)


def _remove_file(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def create_app(rvc: Any, *, device: str | None = None) -> FastAPI:
    """Create the RVC service around one configured inference instance."""
    app = FastAPI(title="Pandrator RVC Service", version="1.0.0")
    app.state.rvc = rvc
    app.state.device = device or str(getattr(rvc, "device", "unknown"))
    app.state.lock = threading.RLock()
    app.state.active_params = None

    @app.get("/health")
    def health():
        with app.state.lock:
            return {
                "status": "ok",
                "ready": True,
                "device": app.state.device,
                "active_model": app.state.rvc.current_model,
                "models": len(app.state.rvc.list_models()),
            }

    @app.get("/v1/models")
    def list_models():
        with app.state.lock:
            models = _refresh_models(app.state.rvc)
            return {"models": models, "active_model": app.state.rvc.current_model}

    @app.post("/v1/models/refresh")
    def refresh_models():
        with app.state.lock:
            models = _refresh_models(app.state.rvc)
            return {"models": models, "active_model": app.state.rvc.current_model}

    @app.post("/v1/unload")
    def unload_model():
        with app.state.lock:
            app.state.rvc.unload_model()
            app.state.active_params = None
            return {"status": "ok"}

    @app.post("/v1/convert")
    def convert_audio(
        audio: UploadFile = File(...),
        model: str = Form(...),
        pitch: int = Form(0),
        f0_method: str = Form("rmvpe"),
        index_rate: float = Form(0.3),
        filter_radius: int = Form(3),
        volume_envelope: float = Form(1.0),
        protect: float = Form(0.3),
        resample_sr: int = Form(40000),
    ):
        model = model.strip()
        if not model:
            raise _error(422, "model_required", "A model name is required.")
        if f0_method not in {"rmvpe", "crepe", "harvest", "pm"}:
            raise _error(422, "invalid_f0_method", f"Unsupported f0 method: {f0_method}")
        if not 0.0 <= index_rate <= 1.0:
            raise _error(422, "invalid_index_rate", "index_rate must be between 0 and 1.")
        if not 0.0 <= volume_envelope <= 1.0:
            raise _error(422, "invalid_volume_envelope", "volume_envelope must be between 0 and 1.")
        if not 0.0 <= protect <= 0.5:
            raise _error(422, "invalid_protect", "protect must be between 0 and 0.5.")

        input_path = None
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
                input_path = input_file.name
                input_file.write(audio.file.read())
                input_file.flush()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                output_path = output_file.name

            with app.state.lock:
                models = _refresh_models(app.state.rvc)
                if model not in models:
                    raise _error(404, "model_not_found", f"RVC model '{model}' was not found.")

                if app.state.rvc.current_model != model:
                    if app.state.rvc.current_model:
                        app.state.rvc.unload_model()
                    app.state.rvc.load_model(model)
                    app.state.active_params = None

                params = (
                    pitch,
                    f0_method,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    volume_envelope,
                    protect,
                )
                if app.state.active_params != params:
                    app.state.rvc.set_params(
                        f0up_key=pitch,
                        f0method=f0_method,
                        index_rate=index_rate,
                        filter_radius=filter_radius,
                        resample_sr=resample_sr,
                        rms_mix_rate=volume_envelope,
                        protect=protect,
                    )
                    app.state.active_params = params

                app.state.rvc.infer_file(input_path, output_path)
                with open(output_path, "rb") as output_file:
                    output_data = output_file.read()

            return Response(content=output_data, media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("RVC conversion failed")
            with app.state.lock:
                try:
                    app.state.rvc.unload_model()
                except Exception:
                    logger.exception("Failed to unload RVC model after conversion error")
                app.state.active_params = None
            raise _error(500, "conversion_failed", str(exc)) from exc
        finally:
            _remove_file(input_path)
            _remove_file(output_path)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(status_code=exc.status_code, content={"error": detail})
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "request_failed", "message": str(detail)}},
        )

    return app
