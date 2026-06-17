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
        elif method in self.VALID_LOG_BASES:
            self._fit_log(data)
        else:
            self._validate_input_constraints(data, method)
            self._lambda_ = None
            self._shift_ = 0.0

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
            self._shift_ = 1.0 - min_val
        else:
            self._shift_ = 0.0

        shifted_data = flat_data + self._shift_

        if lambda_param is None:
            _, self._lambda_ = stats.boxcox(shifted_data)
        else:
            self._lambda_ = lambda_param

    def _fit_log(self, data: np.ndarray) -> None:
        """拟合对数变换参数（自动计算平移量使数据为正）。"""
        flat_data = data.flatten()
        min_val = np.min(flat_data)

        if min_val <= 0:
            self._shift_ = 1.0 - min_val
        else:
            self._shift_ = 0.0

        self._lambda_ = None

    def _ln_transform(self, data: np.ndarray) -> np.ndarray:
        shifted = data + self._shift_
        if np.any(shifted <= 0):
            raise ValueError("自然对数变换要求所有数据严格为正（> 0），请检查 shift 参数或拟合数据")
        return np.log(shifted)

    def _log10_transform(self, data: np.ndarray) -> np.ndarray:
        shifted = data + self._shift_
        if np.any(shifted <= 0):
            raise ValueError("log10变换要求所有数据严格为正（> 0），请检查 shift 参数或拟合数据")
        return np.log10(shifted)

    def _log2_transform(self, data: np.ndarray) -> np.ndarray:
        shifted = data + self._shift_
        if np.any(shifted <= 0):
            raise ValueError("log2变换要求所有数据严格为正（> 0），请检查 shift 参数或拟合数据")
        return np.log2(shifted)

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

    def _ln_inverse(self, data: np.ndarray) -> np.ndarray:
        return np.exp(data) - self._shift_

    def _log10_inverse(self, data: np.ndarray) -> np.ndarray:
        return np.power(10.0, data) - self._shift_

    def _log2_inverse(self, data: np.ndarray) -> np.ndarray:
        return np.power(2.0, data) - self._shift_

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
    def _validate_input_constraints(data: np.ndarray, method: str) -> None:
        if method == 'sqrt':
            if np.any(data < 0):
                raise ValueError("平方根变换要求所有数据非负（>= 0）")
        elif method == 'reciprocal':
            if np.any(data == 0):
                raise ValueError("倒数变换要求所有数据非零（≠ 0）")

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
        """计算分箱后的均值和方差，以及偏度峰度等统计量。"""
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

        skewness = VarianceStabilizer._calc_skewness(sorted_data)
        kurtosis = VarianceStabilizer._calc_kurtosis(sorted_data)
        normality_score = VarianceStabilizer._calc_normality_score(skewness, kurtosis)

        return {
            'bin_means': means.tolist(),
            'bin_variances': variances.tolist(),
            'variance_mean_slope': float(slope) if np.isfinite(slope) else 1e10,
            'coefficient_of_variation': float(cv) if np.isfinite(cv) else 1e10,
            'overall_mean': float(np.mean(sorted_data)),
            'overall_variance': float(np.var(sorted_data, ddof=1)),
            'overall_std': float(np.std(sorted_data, ddof=1)),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'normality_score': float(normality_score),
        }

    @staticmethod
    def _calc_skewness(data: np.ndarray) -> float:
        """计算偏度（Pearson 偏度系数）。"""
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-15:
            return 0.0
        skew = np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
        return float(skew)

    @staticmethod
    def _calc_kurtosis(data: np.ndarray) -> float:
        """计算超额峰度（减去 3，正态分布为 0）。"""
        data = np.asarray(data, dtype=np.float64)
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-15:
            return 0.0
        kurt = (np.sum(((data - mean) / std) ** 4) * n * (n + 1)
                / ((n - 1) * (n - 2) * (n - 3))
                - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return float(kurt)

    @staticmethod
    def _calc_normality_score(skewness: float, kurtosis: float) -> float:
        """
        计算正态性综合得分。

        得分基于偏度和峰度偏离正态分布（0, 0）的程度。
        得分越低表示越接近正态分布。
        公式：sqrt(skew^2 + (kurt/2)^2)
        """
        return float(np.sqrt(skewness ** 2 + (kurtosis / 2.0) ** 2))

    def get_best_method(self, metric: str = 'normality_score') -> Tuple[str, dict]:
        """
        获取最佳变换方法。

        Args:
            metric: 评估指标，可选值:
                - 'normality_score': 正态性综合得分（默认，越低越好）
                - 'variance_mean_slope': 方差-均值斜率（越接近0越好）
                - 'coefficient_of_variation': 变异系数（越低越好）
                - 'skewness': 偏度绝对值（越低越好）
                - 'kurtosis': 峰度绝对值（越低越好）

        Returns:
            (最佳方法名称, 该方法的评估结果)
        """
        if not self.results_:
            raise ValueError("请先调用 evaluate() 方法进行评估")

        valid_metrics = {
            'normality_score', 'variance_mean_slope',
            'coefficient_of_variation', 'skewness', 'kurtosis'
        }
        if metric not in valid_metrics:
            raise ValueError(
                f"不支持的评估指标: {metric}。可选值: {sorted(valid_metrics)}"
            )

        best_method = 'original'
        best_score = float('inf')

        for method, result in self.results_.items():
            if 'error' in result:
                continue
            if metric == 'variance_mean_slope':
                score = abs(result[metric])
            elif metric in {'skewness', 'kurtosis'}:
                score = abs(result[metric])
            else:
                score = result[metric]

            if score < best_score:
                best_score = score
                best_method = method

        return best_method, self.results_[best_method]

    def recommend(self, goal: str = 'normality') -> dict:
        """
        智能推荐最佳变换方案。

        Args:
            goal: 优化目标，可选值:
                - 'normality': 正态化（使数据更接近正态分布，默认）
                - 'variance_stability': 方差稳定化
                - 'all': 综合评估所有维度

        Returns:
            推荐结果字典，包含最佳方法及详细说明
        """
        if not self.results_:
            raise ValueError("请先调用 evaluate() 方法进行评估")

        if goal == 'normality':
            method, result = self.get_best_method('normality_score')
            return {
                'recommended_method': method,
                'goal': 'normality',
                'score': result['normality_score'],
                'skewness': result['skewness'],
                'kurtosis': result['kurtosis'],
                'details': f"基于正态性得分推荐 {method} 变换，得分 {result['normality_score']:.4f}（越低越接近正态）",
                'transformer_params': result.get('transformer', {}),
            }
        elif goal == 'variance_stability':
            method, result = self.get_best_method('variance_mean_slope')
            return {
                'recommended_method': method,
                'goal': 'variance_stability',
                'score': result['variance_mean_slope'],
                'coefficient_of_variation': result['coefficient_of_variation'],
                'details': f"基于方差稳定化推荐 {method} 变换，斜率 {result['variance_mean_slope']:.4f}（越接近0越稳定）",
                'transformer_params': result.get('transformer', {}),
            }
        elif goal == 'all':
            rankings = {}
            metrics = ['normality_score', 'variance_mean_slope', 'coefficient_of_variation']

            for method, result in self.results_.items():
                if 'error' in result:
                    continue
                rank_sum = 0
                for metric in metrics:
                    if metric == 'variance_mean_slope':
                        score = abs(result[metric])
                    else:
                        score = result[metric]
                    all_scores = [
                        abs(r[metric]) if metric == 'variance_mean_slope' else r[metric]
                        for r in self.results_.values() if 'error' not in r
                    ]
                    rank = sum(1 for s in all_scores if s < score) + 1
                    rank_sum += rank
                rankings[method] = rank_sum

            best_method = min(rankings, key=rankings.get)
            best_result = self.results_[best_method]
            return {
                'recommended_method': best_method,
                'goal': 'all',
                'rank_sum': rankings[best_method],
                'normality_score': best_result['normality_score'],
                'variance_mean_slope': best_result['variance_mean_slope'],
                'coefficient_of_variation': best_result['coefficient_of_variation'],
                'details': f"综合多维度评估推荐 {best_method} 变换（排名和：{rankings[best_method]}）",
                'transformer_params': best_result.get('transformer', {}),
                'all_rankings': dict(sorted(rankings.items(), key=lambda x: x[1])),
            }
        else:
            raise ValueError(f"不支持的优化目标: {goal}。可选值: 'normality', 'variance_stability', 'all'")

    def summary(self, mode: str = 'full') -> str:
        """
        生成评估结果摘要。

        Args:
            mode: 摘要模式，可选值:
                - 'full': 完整摘要（所有指标）
                - 'normality': 仅显示正态化相关指标
                - 'variance': 仅显示方差稳定化相关指标

        Returns:
            格式化的摘要字符串
        """
        if not self.results_:
            return "尚未进行评估，请调用 evaluate() 方法。"

        if mode == 'full':
            return self._summary_full()
        elif mode == 'normality':
            return self._summary_normality()
        elif mode == 'variance':
            return self._summary_variance()
        else:
            raise ValueError(f"不支持的摘要模式: {mode}。可选值: 'full', 'normality', 'variance'")

    def _summary_full(self) -> str:
        """生成完整评估摘要。"""
        lines = []
        width = 90
        lines.append("=" * width)
        lines.append("数据变换效果评估摘要".center(width))
        lines.append("=" * width)

        header = (f"{'方法':<12} {'偏度':<12} {'峰度':<12} {'正态得分':<12} "
                  f"{'方差斜率':<14} {'变异系数':<14} {'状态':<10}")
        lines.append(header)
        lines.append("-" * width)

        for method, result in self.results_.items():
            if 'error' in result:
                err = result['error'][:18]
                lines.append(f"{method:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} "
                             f"{'N/A':<14} {'N/A':<14} 失败: {err}")
            else:
                skew = f"{result['skewness']:.6f}"
                kurt = f"{result['kurtosis']:.6f}"
                norm = f"{result['normality_score']:.6f}"
                slope = f"{result['variance_mean_slope']:.6f}"
                cv = f"{result['coefficient_of_variation']:.6f}"
                lines.append(f"{method:<12} {skew:<12} {kurt:<12} {norm:<12} "
                             f"{slope:<14} {cv:<14} 成功")

        lines.append("=" * width)

        best_norm, _ = self.get_best_method('normality_score')
        best_var, _ = self.get_best_method('variance_mean_slope')
        best_cv, _ = self.get_best_method('coefficient_of_variation')

        lines.append("🏆 最佳变换推荐:")
        lines.append(f"  • 正态化最佳: {best_norm}")
        lines.append(f"  • 方差稳定最佳: {best_var}")
        lines.append(f"  • 变异系数最低: {best_cv}")
        lines.append("=" * width)

        return "\n".join(lines)

    def _summary_normality(self) -> str:
        """生成正态化评估摘要。"""
        lines = []
        width = 70
        lines.append("=" * width)
        lines.append("正态化效果评估摘要".center(width))
        lines.append("=" * width)
        lines.append(f"{'方法':<12} {'偏度':<14} {'峰度':<14} {'正态得分':<14} {'状态':<10}")
        lines.append("-" * width)

        for method, result in self.results_.items():
            if 'error' in result:
                err = result['error'][:18]
                lines.append(f"{method:<12} {'N/A':<14} {'N/A':<14} {'N/A':<14} 失败: {err}")
            else:
                skew = f"{result['skewness']:.6f}"
                kurt = f"{result['kurtosis']:.6f}"
                norm = f"{result['normality_score']:.6f}"
                lines.append(f"{method:<12} {skew:<14} {kurt:<14} {norm:<14} 成功")

        lines.append("=" * width)
        best_method, best_result = self.get_best_method('normality_score')
        lines.append(f"🏆 最佳正态化变换: {best_method}")
        lines.append(f"   偏度: {best_result['skewness']:.6f}  (理想值: 0)")
        lines.append(f"   峰度: {best_result['kurtosis']:.6f}  (理想值: 0)")
        lines.append(f"   正态得分: {best_result['normality_score']:.6f} (越低越好)")
        lines.append("=" * width)

        return "\n".join(lines)

    def _summary_variance(self) -> str:
        """生成方差稳定化评估摘要。"""
        lines = []
        width = 70
        lines.append("=" * width)
        lines.append("方差稳定化评估摘要".center(width))
        lines.append("=" * width)
        lines.append(f"{'方法':<12} {'方差-均值斜率':<18} {'变异系数':<14} {'状态':<10}")
        lines.append("-" * width)

        for method, result in self.results_.items():
            if 'error' in result:
                err = result['error'][:18]
                lines.append(f"{method:<12} {'N/A':<18} {'N/A':<14} 失败: {err}")
            else:
                slope = f"{result['variance_mean_slope']:.6f}"
                cv = f"{result['coefficient_of_variation']:.6f}"
                lines.append(f"{method:<12} {slope:<18} {cv:<14} 成功")

        lines.append("=" * width)
        best_method, best_result = self.get_best_method('variance_mean_slope')
        lines.append(f"🏆 最佳方差稳定变换: {best_method}")
        lines.append(f"   方差-均值斜率: {best_result['variance_mean_slope']:.6f} (越接近0越好)")
        lines.append(f"   变异系数: {best_result['coefficient_of_variation']:.6f}")
        lines.append("=" * width)

        return "\n".join(lines)

    def comparison_table(self) -> str:
        """
        生成变换前后对比表。

        Returns:
            格式化的对比表字符串
        """
        if not self.results_:
            return "尚未进行评估，请调用 evaluate() 方法。"

        original = self.results_.get('original', {})
        if 'error' in original:
            return "原始数据评估失败，无法生成对比表。"

        width = 78
        lines = ["=" * width, "变换前后对比表".center(width), "=" * width]
        lines.append(f"{'指标':<20} {'原始':<16} {'最佳变换':<16} {'改善率':<12} {'方法':<10}")
        lines.append("-" * width)

        metric_labels = [
            ('skewness', '偏度'),
            ('kurtosis', '峰度'),
            ('normality_score', '正态得分'),
            ('variance_mean_slope', '方差-均值斜率'),
            ('coefficient_of_variation', '变异系数'),
        ]

        for metric, label in metric_labels:
            orig_val = abs(original[metric]) if metric in {'skewness', 'kurtosis', 'variance_mean_slope'} else original[metric]
            best_method, best_result = self.get_best_method(metric)
            best_val = abs(best_result[metric]) if metric in {'skewness', 'kurtosis', 'variance_mean_slope'} else best_result[metric]

            if orig_val > 1e-10:
                improvement = (orig_val - best_val) / orig_val * 100
                improvement_str = f"{improvement:+.2f}%"
            else:
                improvement_str = "N/A"

            orig_str = f"{original[metric]:.6f}"
            best_str = f"{best_result[metric]:.6f}"
            lines.append(f"{label:<20} {orig_str:<16} {best_str:<16} {improvement_str:<12} {best_method:<10}")

        lines.append("=" * width)
        return "\n".join(lines)


def log_transform(data: np.ndarray, base: str = 'ln') -> Tuple[np.ndarray, float]:
    """
    便捷函数：对数变换（自动平移非正数数据）。

    Args:
        data: 输入数据
        base: 对数基底，'ln', 'log10', 'log2'

    Returns:
        (变换后的数据, shift平移量)
    """
    transformer = DataTransformer()
    transformed = transformer.fit_transform(data, base)
    return transformed, transformer.shift_


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
