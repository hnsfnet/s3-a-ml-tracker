from typing import Any, List
import numpy as np
from .base import ClassificationMetricCalculator


class AccuracyCalculator(ClassificationMetricCalculator):
    """准确率计算器"""

    @property
    def name(self) -> str:
        return "accuracy"

    @property
    def description(self) -> str:
        return "准确率（Accuracy）：预测正确的样本数占总样本数的比例，越接近1越好。适用于类别均衡的场景。"

    def calculate(self, y_true: List[Any], y_pred: List[Any], **kwargs) -> float:
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y_true, y_pred))


class PrecisionCalculator(ClassificationMetricCalculator):
    """精确率计算器（返回每类精确率）"""

    @property
    def name(self) -> str:
        return "precision"

    @property
    def description(self) -> str:
        return "精确率（Precision）：被预测为正类的样本中真正是正类的比例，越接近1越好。用于关注误报代价高的场景（如垃圾邮件过滤）。"

    def calculate(self, y_true: List[Any], y_pred: List[Any], labels=None, **kwargs):
        from sklearn.metrics import precision_score
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        macro = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        weighted = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        return {
            "per_class": {str(labels[i]): float(per_class[i]) for i in range(len(labels))},
            "macro_avg": macro,
            "weighted_avg": weighted,
        }


class RecallCalculator(ClassificationMetricCalculator):
    """召回率计算器"""

    @property
    def name(self) -> str:
        return "recall"

    @property
    def description(self) -> str:
        return "召回率（Recall）：真正是正类的样本中被预测为正类的比例，越接近1越好。用于关注漏报代价高的场景（如疾病筛查）。"

    def calculate(self, y_true: List[Any], y_pred: List[Any], labels=None, **kwargs):
        from sklearn.metrics import recall_score
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        macro = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        weighted = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        return {
            "per_class": {str(labels[i]): float(per_class[i]) for i in range(len(labels))},
            "macro_avg": macro,
            "weighted_avg": weighted,
        }


class F1Calculator(ClassificationMetricCalculator):
    """F1 分数计算器"""

    @property
    def name(self) -> str:
        return "f1"

    @property
    def description(self) -> str:
        return "F1 分数：精确率和召回率的调和平均值，越接近1越好。综合评估模型在精确率和召回率上的表现，类别不均衡时更可靠。"

    def calculate(self, y_true: List[Any], y_pred: List[Any], labels=None, **kwargs):
        from sklearn.metrics import f1_score
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
        return {
            "per_class": {str(labels[i]): float(per_class[i]) for i in range(len(labels))},
            "macro_avg": macro,
            "weighted_avg": weighted,
        }


class ConfusionMatrixCalculator(ClassificationMetricCalculator):
    """混淆矩阵计算器"""

    @property
    def name(self) -> str:
        return "confusion_matrix"

    @property
    def description(self) -> str:
        return "混淆矩阵：行代表真实标签，列代表预测标签。对角线元素是预测正确的数量，非对角线是预测错误的数量，可直观看到模型易混淆的类别。"

    def calculate(self, y_true: List[Any], y_pred: List[Any], labels=None, **kwargs):
        from sklearn.metrics import confusion_matrix
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "matrix": cm.tolist(),
            "labels": [str(l) for l in labels],
        }


class SupportCalculator(ClassificationMetricCalculator):
    """样本数计算器"""

    @property
    def name(self) -> str:
        return "support"

    @property
    def description(self) -> str:
        return "样本数（Support）：每个类别在数据集中的实际样本数量。"

    def calculate(self, y_true: List[Any], y_pred: List[Any], labels=None, **kwargs):
        if labels is None:
            labels = sorted(list(set(y_true) | set(y_pred)))
        y_true_arr = np.array(y_true)
        support = {}
        for label in labels:
            support[str(label)] = int(np.sum(y_true_arr == label))
        return support
