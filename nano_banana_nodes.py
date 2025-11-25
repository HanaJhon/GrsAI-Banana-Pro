"""
ComfyUI节点实现
定义 Nano Banana 图像生成节点（文生图 / 图生图 / 多图）
支持模型: nano-banana-pro
"""

import os
import tempfile
import logging
import base64
import json
from typing import Any, Tuple, Optional, Dict, List
from io import BytesIO

import torch
import numpy as np
from PIL import Image
import requests

# ======================
# 配置 & 工具函数
# ======================

class GrsaiConfig:
    API_HOSTS = {
        "domestic": "https://grsai.dakka.com.cn",
        "overseas": "https://api.grsai.com"
    }
    SUPPORTED_MODELS = ["nano-banana-pro", "nano-banana", "nano-banana-fast"]
    SUPPORTED_ASPECT_RATIOS = [
        "auto", "1:1", "16:9", "9:16", "4:3", "3:4",
        "3:2", "2:3", "5:4", "4:5", "21:9"
    ]
    SUPPORTED_IMAGE_SIZES = ["1K", "2K", "4K"]
    
    @staticmethod
    def get_api_key():
        return os.getenv("NANO_BANANA_API_KEY")
    
    API_KEY_ERROR = "请设置环境变量 NANO_BANANA_API_KEY"

def pil_to_tensor(pil_images: List[Image.Image]) -> torch.Tensor:
    """将 PIL 图像列表转为 ComfyUI tensor (N, H, W, C)"""
    if not pil_images:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    tensors = []
    for img in pil_images:
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    return torch.stack(tensors) if len(tensors) > 1 else tensors[0].unsqueeze(0)

def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """将 ComfyUI IMAGE tensor 转为 PIL 图像列表"""
    if tensor is None:
        return []
    tensor = tensor.cpu()
    if tensor.ndim == 4:
        return [
            Image.fromarray(
                np.clip(255.0 * t.numpy(), 0, 255).astype(np.uint8)
            ).convert("RGB")
            for t in tensor
        ]
    else:
        return []

def format_error_message(e: Exception) -> str:
    return str(e).split('\n')[0][:200]

def upload_file_to_grsai(api_key: str, file_path: str, host: str = "domestic") -> str:
    """上传文件到 GRSAI，返回公开 URL"""
    url = f"{GrsaiConfig.API_HOSTS[host]}/v1/upload"
    headers = {"Authorization": f"Bearer {api_key}"}
    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(url, headers=headers, files=files, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") == 0 and "data" in result and "url" in result["data"]:
        return result["data"]["url"]
    else:
        raise RuntimeError(f"Upload failed: {result.get('msg', 'Unknown')}")

# ======================
# 节点类
# ======================

class SuppressLogs:
    def __init__(self):
        self.loggers = ["httpx", "httpcore", "urllib3"]
        self.levels = {}

    def __enter__(self):
        for name in self.loggers:
            logger = logging.getLogger(name)
            self.levels[name] = logger.level
            logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, *args):
        for name, level in self.levels.items():
            logging.getLogger(name).setLevel(level)

class GrsaiNanoBananaProNode:
    FUNCTION = "execute"
    CATEGORY = "GrsAI/Nano Banana"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A ripe banana on a white surface, studio lighting, 4K"
                }),
                "model": (["nano-banana-pro"], {"default": "nano-banana-pro"}),  # 仅用 pro
                "imageSize": (GrsaiConfig.SUPPORTED_IMAGE_SIZES, {"default": "4K"}),
                "api_host": (["domestic", "overseas"], {"default": "domestic"}),
            },
            "optional": {
                "use_aspect_ratio": ("BOOLEAN", {"default": True}),
                "aspect_ratio": (GrsaiConfig.SUPPORTED_ASPECT_RATIOS, {"default": "auto"}),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "status")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def _error_result(self, msg: str) -> Tuple[torch.Tensor, str]:
        print(f"[Nano Banana] Error: {msg}")
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32), f"error: {msg}"

    def execute(self, prompt, model, imageSize, api_host, use_aspect_ratio, aspect_ratio, **kwargs):
        api_key = GrsaiConfig.get_api_key()
        if not api_key:
            return self._error_result(GrsaiConfig.API_KEY_ERROR)

        # 收集图像输入
        image_tensors = [kwargs.get(f"image_{i}") for i in range(1, 7)]
        image_tensors = [img for img in image_tensors if img is not None]

        uploaded_urls = []
        temp_files = []

        try:
            # 上传图像（如果有的话）
            if image_tensors:
                for tensor in image_tensors:
                    pil_list = tensor_to_pil(tensor)
                    if not pil_list:
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        pil_list[0].save(tmp, "PNG")
                        temp_files.append(tmp.name)
                        with SuppressLogs():
                            url = upload_file_to_grsai(api_key, tmp.name, api_host)
                        uploaded_urls.append(url)

            # 构建 payload
            payload = {
                "model": model,
                "prompt": prompt,
                "imageSize": imageSize,
                "webHook": "-1"  # 用轮询模式
            }

            if use_aspect_ratio:
                payload["aspectRatio"] = aspect_ratio

            if uploaded_urls:
                payload["urls"] = uploaded_urls

            # 调用 draw 接口
            base_url = GrsaiConfig.API_HOSTS[api_host]
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            with SuppressLogs():
                draw_resp = requests.post(
                    f"{base_url}/v1/draw/nano-banana",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
            draw_resp.raise_for_status()
            draw_data = draw_resp.json()
            if draw_data.get("code") != 0:
                return self._error_result(f"Submit failed: {draw_data.get('msg', 'Unknown')}")

            task_id = draw_data["data"]["id"]

            # 轮询结果
            for _ in range(120):
                with SuppressLogs():
                    result_resp = requests.post(
                        f"{base_url}/v1/draw/result",
                        json={"id": task_id},
                        headers=headers,
                        timeout=10
                    )
                result_resp.raise_for_status()
                result_data = result_resp.json()

                if result_data["code"] == -22:
                    import time
                    time.sleep(0.5)
                    continue
                if result_data["code"] != 0:
                    return self._error_result(f"Result error: {result_data.get('msg', 'Unknown')}")

                task = result_data["data"]
                if task["status"] == "succeeded" and task["results"]:
                    img_url = task["results"][0]["url"]
                    img_resp = requests.get(img_url, timeout=15)
                    img_resp.raise_for_status()
                    pil_img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                    tensor_out = pil_to_tensor([pil_img])
                    status = f"succeeded | ref: {len(uploaded_urls)} | {imageSize}"
                    return tensor_out, status

                elif task["status"] == "failed":
                    reason = task.get("failure_reason") or task.get("error") or "Unknown"
                    return self._error_result(f"Task failed: {reason}")

                import time
                time.sleep(0.5)

            return self._error_result("Timeout: Generation took too long")

        except Exception as e:
            return self._error_result(format_error_message(e))
        finally:
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)

# ======================
# 注册节点
# ======================

NODE_CLASS_MAPPINGS = {
    "Grsai_NanoBananaPro": GrsaiNanoBananaProNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grsai_NanoBananaPro": "🍌 GrsAI Nano Banana Pro",
}
