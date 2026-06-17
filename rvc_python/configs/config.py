import json
import os
from multiprocessing import cpu_count

import torch

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from rvc_python.modules.ipex import ipex_init

        ipex_init()
except Exception:  # pylint: disable=broad-exception-caught
    pass
import logging

logger = logging.getLogger(__name__)

version_config_list = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/48k.json",
    "v2/32k.json",
]


class Config:
    def __init__(self,lib_dir,device,is_dml = False):
        self.lib_dir = lib_dir
        self.device = self.normalize_device(device)
        self.is_half = self.device.startswith(("cuda", "xpu"))
        self.use_jit = False
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        self.dml = is_dml
        self.instead = ""
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def normalize_device(device) -> str:
        requested = str(device or "auto").strip().lower()
        if requested in {"cpu", "cpu:0"}:
            return "cpu"
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            if Config.has_xpu():
                return "xpu:0"
            if Config.has_mps():
                return "mps"
            return "cpu"
        return requested

    def load_config_json(self) -> dict:
        d = {}
        for config_file in version_config_list:
            with open(f"{self.lib_dir}/configs/{config_file}", "r") as f:
                d[config_file] = json.load(f)
        return d

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @staticmethod
    def has_xpu() -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False

    def device_config(self) -> tuple:
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
            if ':' in self.device:
                i_device = int(self.device.split(":")[-1])
            else:
                i_device = 0
                self.device = "cuda:0"

            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("Found GPU %s, force to fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                logger.info("Using low-memory inference settings.")
        elif self.device.startswith("xpu"):
            if not self.has_xpu():
                raise RuntimeError("XPU was requested, but PyTorch cannot access an XPU device.")
            self.instead = self.device
            self.is_half = True
        elif self.device == "mps":
            if not self.has_mps():
                raise RuntimeError("MPS was requested, but PyTorch cannot access MPS.")
            self.is_half = False
            self.use_fp32_config()
        elif self.device == "cpu":
            logger.info("Using CPU inference")
            self.is_half = False
            self.use_fp32_config()
        else:
            raise ValueError(f"Unsupported inference device: {self.device}")

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        if self.dml:
            logger.info("Use DirectML instead")
            if (
                os.path.exists(
                    r"venv\Lib\site-packages\onnxruntime\capi\DirectML.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        r"venv\Lib\site-packages\onnxruntime",
                        r"venv\Lib\site-packages\onnxruntime-cuda",
                    )
                except:
                    pass
                try:
                    os.rename(
                        r"venv\Lib\site-packages\onnxruntime-dml",
                        r"venv\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
            # if self.device != "cpu":
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                logger.info(f"Use {self.instead} instead")
        print("is_half:%s, device:%s" % (self.is_half, self.device))
        return x_pad, x_query, x_center, x_max
