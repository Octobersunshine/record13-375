import numpy as np
from scipy import stats
from typing import Tuple, Optional, Union, Callable


class DataTransformer:
    """
    数据变换服务类，支持多种数据预处理和方差稳定化变换方法。

    支持的变换：
    - 对数变换: ln, log10, log2
    - 平方根变换: sqrt
    - 倒数变换: reciprocal
    - Box-Cox 变换: boxcox
    """

    VALID_LOG_BASES = {'ln', 'log10', 'log2'}
    VALID_METHODS = {'ln', 'log10', 'log2', 'sqrt', 'reciprocal', 'boxcox'}

    def __init__(self):
        self._lambda_: Optional[float] = None
        self._shift_: Optional[float] = None
        self._method_: Optional[str] = None
        self._is_fitted_: bool = False

    def fit(self, data: np.ndarray, method: str, lambda_param: Optional[float] = None) -> 'DataTransformer':
        """
        拟合变换器，计算所需的参数。

        Args:
            data: 输入数据数组，一维或多维
            method: 变换方法，可选值见 VALID_METHODS
            lambda_param: Box-Cox 变换的 lambda 参数，None 表示自动估计

        Returns:
            self: 拟合后的变换器实例
        """
        data = np.asarray(data, dtype=np.float64)
        self._validate_method(method)
        self._method_ = method

        if method == 'boxcox':
            self._fit_boxcox(data, lambda_param)
        else:
            self._validate_input_positivity(data, method)
            self._lambda_ = None

        self._is_fitted_ = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        对数据应用变换。

        Args:
            data: 输入数据数组

        Returns:
            变换后的数据数组
        """
        self._check_fitted()
        data = np.asarray(data, dtype=np.float64)

        transform_map = {
            'ln': self._ln_transform,
            'log10': self._log10_transform,
            'log2': self._log2_transform,
            'sqrt': self._sqrt_transform,
            'reciprocal': self._reciprocal_transform,
            'boxcox': self._boxcox_transform,
        }

        return transform_map[self._method_](data)

    def fit_transform(self, data: np.ndarray, method: str, lambda_param: Optional[float] = None) -> np.ndarray:
        """
        拟合并变换数据的便捷方法。

        Args:
            data: 输入数据数组
            method: 变换方法
            lambda_param: Box-Cox 的 lambda 参数

        Returns:
            变换后的数据数组
        """
        self.fit(data, method, lambda_param)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        对变换后的数据进行逆变换，恢复原始尺度。

        Args:
            data: 变换后的数据数组

        Returns:
            逆变换后的数据数组（原始尺度）
        """
        self._check_fitted()
        data = np.asarray(data, dtype=np.float64)

        inverse_map = {
            'ln': self._ln_inverse,
            'log10': self._log10_inverse,
            'log2': self._log2_inverse,
            'sqrt': self._sqrt_inverse,
            'reciprocal': self._reciprocal_inverse,
            'boxcox': self._boxcox_inverse,
        }

        return inverse_map[self._method_](data)

    def _fit_boxcox(self, data: np.ndarray, lambda_param: Optional[float]) -> None:
        """拟合 Box-Cox 变换参数。"""
        flat_data = data.flatten()
        min_val = np.min(flat_data)

        if min_val <= 0:
            self._shift_ = 1.0 - min_val if min_val <= 0 else 0.0
        else:
            self._shift_ = 0.0

        shifted_data = flat_data + self._shift_

        if lambda_param is None:
            _, self._lambda_ = stats.boxcox(shifted_data)
        else:
            self._lambda_ = lambda_param

    @staticmethod
    def _ln_transform(data: np.ndarray) -> np.ndarray:
        DataTransformer._check_positive(data, '自然对数')
        return np.log(data)

    @staticmethod
    def _log10_transform(data: np.ndarray) -> np.ndarray:
        DataTransformer._check_positive(data, 'log10')
        return np.log10(data)

    @staticmethod
    def _log2_transform(data: np.ndarray) -> np.ndarray:
        DataTransformer._check_positive(data, 'log2')
        return np.log2(data)

    @staticmethod
    def _sqrt_transform(data: np.ndarray) -> np.ndarray:
        if np.any(data < 0):
            raise ValueError("平方根变换要求所有数据非负（>= 0）")
        return np.sqrt(data)

    @staticmethod
    def _reciprocal_transform(data: np.ndarray) -> np.ndarray:
        if np.any(data == 0):
            raise ValueError("倒数变换要求所有数据非零（≠ 0）")
        return 1.0 / data

    def _boxcox_transform(self, data: np.ndarray) -> np.ndarray:
        shifted = data + self._shift_
        if np.any(shifted <= 0):
            raise ValueError("Box-Cox 变换要求所有数据严格为正，请检查 shift 参数")

        lam = self._lambda_
        if np.isclose(lam, 0.0):
            return np.log(shifted)
        return (np.power(shifted, lam) - 1.0) / lam

    @staticmethod
    def _ln_inverse(data: np.ndarray) -> np.ndarray:
        return np.exp(data)

    @staticmethod
    def _log10_inverse(data: np.ndarray) -> np.ndarray:
        return np.power(10.0, data)

    @staticmethod
    def _log2_inverse(data: np.ndarray) -> np.ndarray:
        return np.power(2.0, data)

    @staticmethod
    def _sqrt_inverse(data: np.ndarray) -> np.ndarray:
        if np.any(data < 0):
            raise ValueError("平方根逆变换要求所有输入非负")
        return np.square(data)

    @staticmethod
    def _reciprocal_inverse(data: np.ndarray) -> np.ndarray:
        if np.any(data == 0):
            raise ValueError("倒数逆变换要求所有输入非零")
        return 1.0 / data

    def _boxcox_inverse(self, data: np.ndarray) -> np.ndarray:
        lam = self._lambda_
        if np.isclose(lam, 0.0):
            result = np.exp(data)
        else:
            result = np.power(lam * data + 1.0, 1.0 / lam)
        return result - self._shift_

    @property
    def lambda_(self) -> Optional[float]:
        """获取 Box-Cox 变换的 lambda 参数。"""
        return self._lambda_

    @property
    def shift_(self) -> Optional[float]:
        """获取 Box-Cox 变换的 shift 参数。"""
        return self._shift_

    @property
    def method_(self) -> Optional[str]:
        """获取当前使用的变换方法。"""
        return self._method_

    @property
    def is_fitted_(self) -> bool:
        """获取变换器是否已拟合。"""
        return self._is_fitted_

    def _check_fitted(self) -> None:
        if not self._is_fitted_:
            raise ValueError("变换器尚未拟合，请先调用 fit() 方法")

    def _validate_method(self, method: str) -> None:
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"不支持的变换方法: {method}。可选值: {sorted(self.VALID_METHODS)}"
            )

    @staticmethod
    def _validate_input_positivity(data: np.ndarray, method: str) -> None:
        if method in DataTransformer.VALID_LOG_BASES:
            DataTransformer._check_positive(data, method)

    @staticmethod
    def _check_positive(data: np.ndarray, name: str) -> None:
        if np.any(data <= 0):
            raise ValueError(f"{name}变换要求所有数据严格为正（> 0）")

    def get_params(self) -> dict:
        """获取变换器的所有参数。"""
        return {
            'method': self._method_,
            'lambda': self._lambda_,
            'shift': self._shift_,
            'is_fitted': self._is_fitted_,
        }

    def __repr__(self) -> str:
        if self._is_fitted_:
            return (f"DataTransformer(method='{self._method_}', "
                    f"lambda={self._lambda_}, shift={self._shift_}, fitted=True)")
        return "DataTransformer(fitted=False)"


class VarianceStabilizer:
    """
    方差稳定化分析器，用于评估和比较不同变换方法的方差稳定效果。
    """

    def __init__(self):
        self.results_: dict = {}

    def evaluate(self, data: np.ndarray, bins: int = 10) -> dict:
        """
        评估原始数据和各种变换方法的方差稳定效果。

        Args:
            data: 输入数据数组
            bins: 分组数量，用于计算均值-方差关系

        Returns:
            包含各种方法评估结果的字典
        """
        data = np.asarray(data, dtype=np.float64)
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]

        bin_size = len(sorted_data) // bins
        self.results_ = {}

        self.results_['original'] = self._calc_variance_stability(sorted_data, bins, bin_size)

        methods = ['ln', 'log10', 'log2', 'sqrt', 'reciprocal', 'boxcox']
        for method in methods:
            try:
                transformer = DataTransformer()
                transformed = transformer.fit_transform(data, method)
                sorted_transformed = transformed[sorted_indices]
                self.results_[method] = self._calc_variance_stability(
                    sorted_transformed, bins, bin_size
                )
                self.results_[method]['transformer'] = transformer.get_params()
            except (ValueError, np.linalg.LinAlgError) as e:
                self.results_[method] = {'error': str(e)}

        return self.results_

    @staticmethod
    def _calc_variance_stability(sorted_data: np.ndarray, bins: int, bin_size: int) -> dict:
        """计算分箱后的均值和方差。"""
        means = []
        variances = []
        for i in range(bins):
            start = i * bin_size
            end = start + bin_size if i < bins - 1 else len(sorted_data)
            bin_data = sorted_data[start:end]
            means.append(np.mean(bin_data))
            variances.append(np.var(bin_data, ddof=1))

        means = np.array(means)
        variances = np.array(variances)

        valid_mask = (variances > 0) & (means > 0) & np.isfinite(means) & np.isfinite(variances)
        if np.sum(valid_mask) >= 2:
            log_means = np.log(means[valid_mask])
            log_vars = np.log(variances[valid_mask])
            finite_mask = np.isfinite(log_means) & np.isfinite(log_vars)
            if np.sum(finite_mask) >= 2:
                try:
                    slope, _ = np.polyfit(log_means[finite_mask], log_vars[finite_mask], 1)
                    slope = float(slope)
                except (np.linalg.LinAlgError, ValueError):
                    slope = np.nan
            else:
                slope = np.nan
        else:
            slope = np.nan

        data_std = np.std(sorted_data, ddof=1)
        data_mean = np.mean(sorted_data)
        if np.isfinite(data_mean) and abs(data_mean) > 1e-10:
            cv = float(data_std / abs(data_mean))
        else:
            cv = np.nan

        return {
            'bin_means': means.tolist(),
            'bin_variances': variances.tolist(),
            'variance_mean_slope': float(slope) if np.isfinite(slope) else 1e10,
            'coefficient_of_variation': float(cv) if np.isfinite(cv) else 1e10,
            'overall_mean': float(np.mean(sorted_data)),
            'overall_variance': float(np.var(sorted_data, ddof=1)),
        }

    def get_best_method(self, metric: str = 'variance_mean_slope') -> Tuple[str, dict]:
        """
        获取最佳变换方法。

        Args:
            metric: 评估指标，'variance_mean_slope' 或 'coefficient_of_variation'

        Returns:
            (最佳方法名称, 该方法的评估结果)
        """
        if not self.results_:
            raise ValueError("请先调用 evaluate() 方法进行评估")

        best_method = 'original'
        best_score = float('inf')

        for method, result in self.results_.items():
            if 'error' in result:
                continue
            score = abs(result[metric])
            if score < best_score:
                best_score = score
                best_method = method

        return best_method, self.results_[best_method]

    def summary(self) -> str:
        """生成评估结果摘要。"""
        if not self.results_:
            return "尚未进行评估，请调用 evaluate() 方法。"

        lines = ["=" * 70, "方差稳定化评估摘要".center(70), "=" * 70]
        lines.append(f"{'方法':<12} {'斜率(方差-均值)':<18} {'变异系数':<14} {'状态':<12}")
        lines.append("-" * 70)

        for method, result in self.results_.items():
            if 'error' in result:
                lines.append(f"{method:<12} {'N/A':<18} {'N/A':<14} 失败: {result['error'][:20]}")
            else:
                slope = f"{result['variance_mean_slope']:.6f}"
                cv = f"{result['coefficient_of_variation']:.6f}"
                lines.append(f"{method:<12} {slope:<18} {cv:<14} 成功")

        lines.append("=" * 70)
        best_method, best_result = self.get_best_method()
        lines.append(f"最佳方法: {best_method}")
        lines.append(f"  斜率: {best_result['variance_mean_slope']:.6f}")
        lines.append(f"  变异系数: {best_result['coefficient_of_variation']:.6f}")
        lines.append("=" * 70)

        return "\n".join(lines)


def log_transform(data: np.ndarray, base: str = 'ln') -> np.ndarray:
    """
    便捷函数：对数变换。

    Args:
        data: 输入数据
        base: 对数基底，'ln', 'log10', 'log2'

    Returns:
        变换后的数据
    """
    transformer = DataTransformer()
    return transformer.fit_transform(data, base)


def sqrt_transform(data: np.ndarray) -> np.ndarray:
    """
    便捷函数：平方根变换。

    Args:
        data: 输入数据

    Returns:
        变换后的数据
    """
    transformer = DataTransformer()
    return transformer.fit_transform(data, 'sqrt')


def reciprocal_transform(data: np.ndarray) -> np.ndarray:
    """
    便捷函数：倒数变换。

    Args:
        data: 输入数据

    Returns:
        变换后的数据
    """
    transformer = DataTransformer()
    return transformer.fit_transform(data, 'reciprocal')


def boxcox_transform(data: np.ndarray, lambda_param: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """
    便捷函数：Box-Cox 变换。

    Args:
        data: 输入数据
        lambda_param: lambda 参数，None 表示自动估计

    Returns:
        (变换后的数据, lambda参数, shift参数)
    """
    transformer = DataTransformer()
    transformed = transformer.fit_transform(data, 'boxcox', lambda_param)
    return transformed, transformer.lambda_, transformer.shift_
