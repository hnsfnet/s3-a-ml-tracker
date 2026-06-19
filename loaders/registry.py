from typing import Dict, Type, Optional
from pathlib import Path

from .base import BaseModelLoader
from .sklearn_loader import SklearnModelLoader


class ModelLoaderRegistry:
    """模型加载器注册表，根据文件后缀自动选择加载器"""

    _loaders: Dict[str, Type[BaseModelLoader]] = {}
    _instances: Dict[str, BaseModelLoader] = {}
    _default_loader: Optional[Type[BaseModelLoader]] = None

    @classmethod
    def register(cls, loader_cls: Type[BaseModelLoader], default: bool = False):
        """注册一个模型加载器"""
        instance = loader_cls()
        for ext in instance.supported_extensions:
            cls._loaders[ext.lower()] = loader_cls
            cls._instances[ext.lower()] = instance
        if default:
            cls._default_loader = loader_cls

    @classmethod
    def get_loader(cls, file_path: str) -> BaseModelLoader:
        """根据文件路径获取对应的加载器"""
        ext = Path(file_path).suffix.lower()
        if ext in cls._instances:
            return cls._instances[ext]
        if cls._default_loader:
            return cls._default_loader()
        raise ValueError(f"No model loader found for file extension: {ext}")

    @classmethod
    def get_loader_by_extension(cls, extension: str) -> Optional[BaseModelLoader]:
        """根据扩展名获取加载器"""
        ext = extension.lower()
        if not ext.startswith("."):
            ext = "." + ext
        return cls._instances.get(ext)

    @classmethod
    def list_supported_extensions(cls) -> list:
        """列出所有支持的扩展名"""
        return sorted(cls._loaders.keys())


ModelLoaderRegistry.register(SklearnModelLoader, default=True)
