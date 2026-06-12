from pathlib import Path

from fastapi.testclient import TestClient

from rvc_python.api import create_app


class FakeRVC:
    def __init__(self):
        self.device = "cpu"
        self.current_model = None
        self.models = {"alpha": {"pth": "alpha.pth", "index": None}}
        self.params = {}
        self.load_calls = 0
        self.infer_calls = 0

    def list_models(self):
        return list(self.models)

    def refresh_models(self):
        return self.list_models()

    def load_model(self, model):
        self.current_model = model
        self.load_calls += 1

    def unload_model(self):
        self.current_model = None

    def set_params(self, **params):
        self.params = params

    def infer_file(self, input_path, output_path):
        self.infer_calls += 1
        Path(output_path).write_bytes(b"converted:" + Path(input_path).read_bytes())


def test_health_and_models():
    client = TestClient(create_app(FakeRVC()))

    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").json()["models"] == ["alpha"]


def test_convert_loads_model_and_returns_wav():
    rvc = FakeRVC()
    client = TestClient(create_app(rvc))

    response = client.post(
        "/v1/convert",
        files={"audio": ("input.wav", b"wave-data", "audio/wav")},
        data={"model": "alpha", "pitch": "2", "volume_envelope": "0.75"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == b"converted:wave-data"
    assert rvc.current_model == "alpha"
    assert rvc.load_calls == 1
    assert rvc.params["f0up_key"] == 2
    assert rvc.params["rms_mix_rate"] == 0.75


def test_missing_model_returns_structured_error():
    client = TestClient(create_app(FakeRVC()))

    response = client.post(
        "/v1/convert",
        files={"audio": ("input.wav", b"wave-data", "audio/wav")},
        data={"model": "missing"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
