import io
from typing import Any, List

from .base import BaseModelLoader


class TorchModelLoader(BaseModelLoader):
    """PyTorch 模型加载器，支持 .pt、.pth 文件（懒加载 torch）"""

    @property
    def supported_extensions(self) -> List[str]:
        return [".pt", ".pth"]

    def load(self, file_content: bytes) -> Any:
        try:
            import torch
        except ImportError as e:
            raise ValueError(
                f"PyTorch is required to load .pt/.pth models: {str(e)}"
            )
        try:
            buffer = io.BytesIO(file_content)
            model = torch.load(buffer, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load PyTorch model: {str(e)}")
        return model

    def detect_prediction_type(self, model: Any) -> str:
        if hasattr(model, "predict_proba") or hasattr(model, "predict"):
            return "classification" if hasattr(model, "predict_proba") else "regression"
        return "regression"
