from abc import ABC, abstractmethod
from typing import Any, Optional, List


class BaseModelLoader(ABC):
    """模型加载器抽象基类，不同框架实现各自的加载器"""

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """返回支持的文件扩展名列表，例如 ['.pkl', '.joblib']"""
        pass

    @abstractmethod
    def load(self, file_content: bytes) -> Any:
        """从字节内容加载模型，返回模型实例"""
        pass

    def supports_file(self, file_path: str) -> bool:
        """判断是否支持该文件"""
        ext = file_path.lower()
        return any(ext.endswith(e) for e in self.supported_extensions)

    def get_classes(self, model: Any) -> Optional[List[Any]]:
        """获取分类模型的类别标签，没有则返回 None"""
        if hasattr(model, "classes_"):
            return list(model.classes_)
        return None

    def has_predict_proba(self, model: Any) -> bool:
        """判断模型是否支持概率预测"""
        return hasattr(model, "predict_proba")

    def detect_prediction_type(self, model: Any) -> str:
        """检测模型类型：classification 或 regression"""
        if self.has_predict_proba(model):
            return "classification"
        return "regression"
