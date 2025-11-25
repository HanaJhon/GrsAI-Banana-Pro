"""
ComfyUI Nano Banana Pro 节点（单文件完整修复版）
- 支持文生图 & 图生图（最多6张参考图）
- 使用 GRSAI 官方上传流程（newUploadTokenZH → R2）
- 修复轮询超时和黑图问题
- 严格遵循 API 文档
"""

import os
import tempfile
import logging
import json
from typing import Any, Tuple, List, Optional
from io import BytesIO

import torch
import numpy as np
from PIL import Image
import requests

# ======================
# 配置
# ======================

class GrsaiConfig:
    API_HOSTS = {
        "domestic": "https://grsai.dakka.com.cn",
        "overseas": "https://api.grsai.com"
    }
    SUPPORTED_ASPECT_RATIOS = [
        "auto", "1:1", "16:9", "9:16", "4:3", "3:4",
        "3:2", "2:3", "5:4", "4:5", "21:9"
    ]
    SUPPORTED_IMAGE_SIZES = ["1K", "2K", "4K"]
    
    @staticmethod
    def get_api_key():
        return os.getenv("NANO_BANANA_API_KEY")
    
    API_KEY_ERROR = "请设置环境变量 NANO_BANANA_API_KEY"

# ======================
# 工具函数
# ======================

def pil_to_tensor(pil_images: List[Image.Image]) -> torch.Tensor:
    if not pil_images:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    tensors = []
    for img in pil_images:
        if img.width == 0 or img.height == 0:
            continue
        img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    if not tensors:
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    return torch.stack(tensors) if len(tensors) > 1 else tensors[0].unsqueeze(0)

def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
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
    return []

def format_error_message(e: Exception) -> str:
    return str(e).split('\n')[0][:200]

# ======================
# 上传逻辑（来自官方 upload.py）
# ======================

def get_upload_token_zh(api_key: str, file_ext: str = "png") -> dict:
    url = "https://grsai.dakka.com.cn/client/resource/newUploadTokenZH"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"sux": file_ext}
    resp = requests.post(url, headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()

def upload_file_zh(api_key: str, file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    ext = os.path.splitext(file_path)[1].lstrip(".").lower() or "png"
    token_data = get_upload_token_zh(api_key, ext)
    data = token_data["data"]
    upload_url = data["url"]
    key = data["key"]
    domain = data["domain"]
    token = data["token"]
    with open(file_path, "rb") as f:
        upload_resp = requests.post(
            upload_url,
            data={"token": token, "key": key},
            files={"file": f},
            timeout=120
        )
    upload_resp.raise_for_status()
    return f"{domain}/{key}"

# ======================
# 日志抑制
# ======================

class SuppressLogs:
    def __enter__(self):
        for name in ["httpx", "httpcore", "urllib3"]:
            logging.getLogger(name).setLevel(logging.WARNING)
        return self
    def __exit__(self, *args):
        pass

# ======================
# 主节点
# ======================

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
        print(f"[Nano Banana Pro] Error: {msg}")
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32), f"error: {msg}"

    def execute(self, prompt, imageSize, api_host, use_aspect_ratio, aspect_ratio, **kwargs):
        api_key = GrsaiConfig.get_api_key()
        if not api_key:
            return self._error_result(GrsaiConfig.API_KEY_ERROR)

        # 收集图像输入
        image_tensors = [kwargs.get(f"image_{i}") for i in range(1, 7)]
        image_tensors = [img for img in image_tensors if img is not None]

        uploaded_urls = []
        temp_files = []

        try:
            # 上传参考图
            if image_tensors:
                for tensor in image_tensors:
                    pil_list = tensor_to_pil(tensor)
                    if not pil_list:
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        pil_list[0].save(tmp, "PNG")
                        temp_files.append(tmp.name)
                        with SuppressLogs():
                            url = upload_file_zh(api_key, tmp.name)
                        uploaded_urls.append(url)
                print(f"[DEBUG] 已上传 {len(uploaded_urls)} 张参考图")

            # 构建绘图请求
            payload = {
                "model": "nano-banana-pro",
                "prompt": prompt,
                "imageSize": imageSize,
                "webHook": "-1"
            }
            if use_aspect_ratio:
                payload["aspectRatio"] = aspect_ratio
            if uploaded_urls:
                payload["urls"] = uploaded_urls

            # 提交任务
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
                return self._error_result(f"提交失败: {draw_data.get('msg', 'Unknown')}")

            task_id = draw_data["data"]["id"]
            print(f"[DEBUG] 任务已提交，ID: {task_id}")

            # 轮询结果（最多90秒）
            for _ in range(180):
                try:
                    with SuppressLogs():
                        result_resp = requests.post(
                            f"{base_url}/v1/draw/result",
                            json={"id": task_id},
                            headers=headers,
                            timeout=10
                        )
                    if result_resp.status_code == 404:
                        import time
                        time.sleep(0.5)
                        continue
                    result_resp.raise_for_status()
                    result_data = result_resp.json()

                    if result_data["code"] == -22:
                        import time
                        time.sleep(0.5)
                        continue
                    if result_data["code"] != 0:
                        return self._error_result(f"结果查询失败: {result_data.get('msg', 'Unknown')}")

                    task = result_data["data"]
                    print(f"[DEBUG] 状态: {task['status']}, 进度: {task['progress']}%")

                    if task["status"] == "succeeded":
                        if not task["results"]:
                            return self._error_result("任务成功但无图像结果")
                        img_url = task["results"][0]["url"]
                        print(f"[DEBUG] 下载图像: {img_url}")

                        try:
                            img_resp = requests.get(img_url, timeout=15)
                            img_resp.raise_for_status()
                            pil_img = Image.open(BytesIO(img_resp.content))
                            if pil_img.mode != "RGB":
                                pil_img = pil_img.convert("RGB")
                            if pil_img.width == 0 or pil_img.height == 0:
                                raise ValueError("图像为空")
                            tensor_out = pil_to_tensor([pil_img])
                            status = f"succeeded | refs: {len(uploaded_urls)} | {imageSize}"
                            print(f"[SUCCESS] 生成成功，尺寸: {pil_img.size}")
                            return tensor_out, status

                        except Exception as img_e:
                            return self._error_result(f"图像下载/处理失败: {format_error_message(img_e)}")

                    elif task["status"] == "failed":
                        reason = task.get("failure_reason") or task.get("error") or "Unknown"
                        return self._error_result(f"任务失败: {reason}")

                    import time
                    time.sleep(0.5)

                except Exception as e:
                    return self._error_result(f"轮询异常: {format_error_message(e)}")

            return self._error_result("Timeout: 生成耗时过长，请重试或降低分辨率")

        except Exception as e:
            return self._error_result(f"执行异常: {format_error_message(e)}")
        finally:
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)

# ======================
# 注册
# ======================

NODE_CLASS_MAPPINGS = {
    "Grsai_NanoBananaPro": GrsaiNanoBananaProNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Grsai_NanoBananaPro": "🍌 GrsAI Nano Banana Pro",
}
