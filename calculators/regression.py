from typing import List
import numpy as np
from .base import RegressionMetricCalculator


class MSECalculator(RegressionMetricCalculator):
    """均方误差计算器"""

    @property
    def name(self) -> str:
        return "mse"

    @property
    def description(self) -> str:
        return "均方误差（MSE）：预测值与真实值之差的平方的平均值，越小越好。对较大误差惩罚更重。"

    def higher_is_better(self) -> bool:
        return False

    def calculate(self, y_true: List[float], y_pred: List[float], **kwargs) -> float:
        from sklearn.metrics import mean_squared_error
        return float(mean_squared_error(y_true, y_pred))


class MAECalculator(RegressionMetricCalculator):
    """平均绝对误差计算器"""

    @property
    def name(self) -> str:
        return "mae"

    @property
    def description(self) -> str:
        return "平均绝对误差（MAE）：预测值与真实值之差的绝对值的平均值，越小越好。对异常值不敏感。"

    def higher_is_better(self) -> bool:
        return False

    def calculate(self, y_true: List[float], y_pred: List[float], **kwargs) -> float:
        from sklearn.metrics import mean_absolute_error
        return float(mean_absolute_error(y_true, y_pred))


class RMSECalculator(RegressionMetricCalculator):
    """均方根误差计算器"""

    @property
    def name(self) -> str:
        return "rmse"

    @property
    def description(self) -> str:
        return "均方根误差（RMSE）：MSE 的平方根，与目标变量单位一致，越小越好。直观反映预测偏差大小。"

    def higher_is_better(self) -> bool:
        return False

    def calculate(self, y_true: List[float], y_pred: List[float], **kwargs) -> float:
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))


class R2Calculator(RegressionMetricCalculator):
    """决定系数计算器"""

    @property
    def name(self) -> str:
        return "r2"

    @property
    def description(self) -> str:
        return "决定系数（R²）：衡量模型对数据变异性的解释能力，取值越接近1越好。1 表示完美预测，0 表示等价于均值预测，负数表示比均值还差。"

    def calculate(self, y_true: List[float], y_pred: List[float], **kwargs) -> float:
        from sklearn.metrics import r2_score
        return float(r2_score(y_true, y_pred))


class ExplainedVarianceCalculator(RegressionMetricCalculator):
    """解释方差计算器"""

    @property
    def name(self) -> str:
        return "explained_variance"

    @property
    def description(self) -> str:
        return "解释方差：模型解释的方差占总方差的比例，越接近1越好。"

    def calculate(self, y_true: List[float], y_pred: List[float], **kwargs) -> float:
        from sklearn.metrics import explained_variance_score
        return float(explained_variance_score(y_true, y_pred))


class MAPECalculator(RegressionMetricCalculator):
    """平均绝对百分比误差计算器"""

    @property
    def name(self) -> str:
        return "mape"

    @property
    def description(self) -> str:
        return "平均绝对百分比误差（MAPE）：预测误差占真实值百分比的平均值，越小越好。便于跨业务场景理解误差相对大小。"

    def higher_is_better(self) -> bool:
        return False

    def calculate(self, y_true: List[float], y_pred: List[float], **kwargs):
        y_true_arr = np.array(y_true, dtype=float)
        y_pred_arr = np.array(y_pred, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            non_zero = y_true_arr != 0
            if np.any(non_zero):
                mape = float(
                    np.mean(np.abs((y_true_arr[non_zero] - y_pred_arr[non_zero]) / y_true_arr[non_zero])) * 100
                )
            else:
                mape = None
        return mape
