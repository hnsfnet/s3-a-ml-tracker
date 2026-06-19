import pickle
from typing import Any, List

from .base import BaseModelLoader


class SklearnModelLoader(BaseModelLoader):
    """scikit-learn 模型加载器，支持 .pkl、.pickle、.joblib 文件"""

    @property
    def supported_extensions(self) -> List[str]:
        return [".pkl", ".pickle", ".joblib"]

    def load(self, file_content: bytes) -> Any:
        try:
            model = pickle.loads(file_content)
        except Exception as e:
            raise ValueError(f"Failed to unpickle scikit-learn model: {str(e)}")
        if not hasattr(model, "predict"):
            raise ValueError("Loaded object is not a valid model (missing predict method)")
        return model

    def validate_sklearn(self, file_content: bytes) -> bool:
        """验证是否为有效的 sklearn 模型（有 fit 和 predict 方法）"""
        try:
            model = self.load(file_content)
            return hasattr(model, "fit") and hasattr(model, "predict")
        except Exception:
            return False
