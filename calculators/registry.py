from typing import Dict, Type, List, Optional, Any, Tuple
from .base import BaseMetricCalculator, ClassificationMetricCalculator, RegressionMetricCalculator
from .classification import (
    AccuracyCalculator,
    PrecisionCalculator,
    RecallCalculator,
    F1Calculator,
    ConfusionMatrixCalculator,
    SupportCalculator,
)
from .regression import (
    MSECalculator,
    MAECalculator,
    RMSECalculator,
    R2Calculator,
    ExplainedVarianceCalculator,
    MAPECalculator,
)


class MetricCalculatorRegistry:
    """指标计算器注册中心，支持动态注册和按名称查找"""

    _calculators: Dict[str, Type[BaseMetricCalculator]] = {}
    _instances: Dict[str, BaseMetricCalculator] = {}

    @classmethod
    def register(cls, calculator_cls: Type[BaseMetricCalculator]):
        """注册一个指标计算器"""
        instance = calculator_cls()
        cls._calculators[instance.name] = calculator_cls
        cls._instances[instance.name] = instance

    @classmethod
    def get(cls, name: str) -> Optional[BaseMetricCalculator]:
        """根据名称获取计算器实例"""
        return cls._instances.get(name)

    @classmethod
    def list_by_type(cls, metric_type: str) -> List[BaseMetricCalculator]:
        """按类型列出所有计算器"""
        return [
            inst for inst in cls._instances.values()
            if inst.metric_type == metric_type
        ]

    @classmethod
    def list_all(cls) -> List[BaseMetricCalculator]:
        """列出所有计算器"""
        return list(cls._instances.values())

    @classmethod
    def get_descriptions(cls, metric_type: Optional[str] = None) -> Dict[str, str]:
        """获取所有指标的中文说明"""
        result = {}
        calculators = cls.list_by_type(metric_type) if metric_type else cls.list_all()
        for calc in calculators:
            result[calc.name] = calc.description
        return result


MetricCalculatorRegistry.register(AccuracyCalculator)
MetricCalculatorRegistry.register(PrecisionCalculator)
MetricCalculatorRegistry.register(RecallCalculator)
MetricCalculatorRegistry.register(F1Calculator)
MetricCalculatorRegistry.register(ConfusionMatrixCalculator)
MetricCalculatorRegistry.register(SupportCalculator)
MetricCalculatorRegistry.register(MSECalculator)
MetricCalculatorRegistry.register(MAECalculator)
MetricCalculatorRegistry.register(RMSECalculator)
MetricCalculatorRegistry.register(R2Calculator)
MetricCalculatorRegistry.register(ExplainedVarianceCalculator)
MetricCalculatorRegistry.register(MAPECalculator)


def compute_classification_metrics(
    y_true: List[Any], y_pred: List[Any], metric_names: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """计算分类评估指标（兼容旧接口）"""
    calculators = MetricCalculatorRegistry.list_by_type("classification")
    if metric_names:
        calculators = [c for c in calculators if c.name in metric_names]

    labels = sorted(list(set(y_true) | set(y_pred)))
    metrics_dict: Dict[str, Any] = {}
    descriptions: Dict[str, str] = {}

    for calc in calculators:
        result = calc.calculate(y_true, y_pred, labels=labels)
        descriptions[calc.name] = calc.description

        if calc.name == "accuracy":
            metrics_dict["accuracy"] = result
        elif calc.name == "precision":
            metrics_dict["per_class"] = {
                label: {
                    "precision": result["per_class"].get(label, 0.0),
                }
                for label in result["per_class"]
            }
            metrics_dict["macro_avg_precision"] = result["macro_avg"]
            metrics_dict["weighted_avg_precision"] = result["weighted_avg"]
            descriptions["macro_avg_precision"] = "宏平均精确率：对每个类别计算精确率后取算术平均，不考虑样本量差异。"
            descriptions["weighted_avg_precision"] = "加权平均精确率：按每类样本量加权计算精确率，考虑类别不均衡。"
        elif calc.name == "recall":
            if "per_class" not in metrics_dict:
                metrics_dict["per_class"] = {}
            for label in result["per_class"]:
                if label not in metrics_dict["per_class"]:
                    metrics_dict["per_class"][label] = {}
                metrics_dict["per_class"][label]["recall"] = result["per_class"][label]
            metrics_dict["macro_avg_recall"] = result["macro_avg"]
            metrics_dict["weighted_avg_recall"] = result["weighted_avg"]
            descriptions["macro_avg_recall"] = "宏平均召回率：对每个类别计算召回率后取算术平均。"
            descriptions["weighted_avg_recall"] = "加权平均召回率：按每类样本量加权计算召回率。"
        elif calc.name == "f1":
            if "per_class" not in metrics_dict:
                metrics_dict["per_class"] = {}
            for label in result["per_class"]:
                if label not in metrics_dict["per_class"]:
                    metrics_dict["per_class"][label] = {}
                metrics_dict["per_class"][label]["f1"] = result["per_class"][label]
            metrics_dict["macro_avg_f1"] = result["macro_avg"]
            metrics_dict["weighted_avg_f1"] = result["weighted_avg"]
            descriptions["macro_avg_f1"] = "宏平均 F1：对每个类别计算 F1 后取算术平均。"
            descriptions["weighted_avg_f1"] = "加权平均 F1：按每类样本量加权计算 F1。"
        elif calc.name == "confusion_matrix":
            metrics_dict["confusion_matrix"] = result["matrix"]
            metrics_dict["labels"] = result["labels"]
        elif calc.name == "support":
            if "per_class" not in metrics_dict:
                metrics_dict["per_class"] = {}
            for label in result:
                if label not in metrics_dict["per_class"]:
                    metrics_dict["per_class"][label] = {}
                metrics_dict["per_class"][label]["support"] = result[label]

    return metrics_dict, descriptions


def compute_regression_metrics(
    y_true: List[float], y_pred: List[float], metric_names: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """计算回归评估指标（兼容旧接口）"""
    calculators = MetricCalculatorRegistry.list_by_type("regression")
    if metric_names:
        calculators = [c for c in calculators if c.name in metric_names]

    metrics_dict: Dict[str, Any] = {}
    descriptions: Dict[str, str] = {}

    for calc in calculators:
        result = calc.calculate(y_true, y_pred)
        metrics_dict[calc.name] = result
        descriptions[calc.name] = calc.description

    return metrics_dict, descriptions
