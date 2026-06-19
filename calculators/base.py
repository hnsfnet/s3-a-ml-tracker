from abc import ABC, abstractmethod
from typing import Any, List, Dict


class BaseMetricCalculator(ABC):
    """指标计算器抽象基类，每个指标实现一个计算器"""

    @property
    @abstractmethod
    def name(self) -> str:
        """指标名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """指标的中文说明"""
        pass

    @property
    @abstractmethod
    def metric_type(self) -> str:
        """指标类型：classification 或 regression"""
        pass

    @abstractmethod
    def calculate(self, y_true: List[Any], y_pred: List[Any], **kwargs) -> Any:
        """计算指标值，返回指标结果（单个值、字典、数组等）"""
        pass

    def higher_is_better(self) -> bool:
        """指标是否越高越好，默认 True"""
        return True


class ClassificationMetricCalculator(BaseMetricCalculator):
    """分类指标计算器基类"""

    @property
    def metric_type(self) -> str:
        return "classification"


class RegressionMetricCalculator(BaseMetricCalculator):
    """回归指标计算器基类"""

    @property
    def metric_type(self) -> str:
        return "regression"
