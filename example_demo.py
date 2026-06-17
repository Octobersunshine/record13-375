import numpy as np
from data_transformer import (
    DataTransformer,
    VarianceStabilizer,
    log_transform,
    sqrt_transform,
    reciprocal_transform,
    boxcox_transform,
)

np.random.seed(42)


def demo_basic_transforms():
    print("=" * 70)
    print("示例 1: 基本数据变换演示".center(70))
    print("=" * 70)

    data = np.random.lognormal(mean=1.0, sigma=0.8, size=1000)
    print(f"\n原始数据统计（对数正态分布）:")
    print(f"  均值: {np.mean(data):.4f}")
    print(f"  标准差: {np.std(data, ddof=1):.4f}")
    print(f"  最小值: {np.min(data):.4f}")
    print(f"  最大值: {np.max(data):.4f}")

    transformer = DataTransformer()

    ln_data = transformer.fit_transform(data.copy(), 'ln')
    print(f"\n 自然对数变换 (ln):")
    print(f"  均值: {np.mean(ln_data):.4f}")
    print(f"  标准差: {np.std(ln_data, ddof=1):.4f}")
    original_recovered = transformer.inverse_transform(ln_data)
    print(f"  逆变换误差: {np.max(np.abs(data - original_recovered)):.2e}")

    log10_data = transformer.fit_transform(data.copy(), 'log10')
    print(f"\n 常用对数变换 (log10):")
    print(f"  均值: {np.mean(log10_data):.4f}")
    print(f"  标准差: {np.std(log10_data, ddof=1):.4f}")

    log2_data = transformer.fit_transform(data.copy(), 'log2')
    print(f"\n 二进制对数变换 (log2):")
    print(f"  均值: {np.mean(log2_data):.4f}")
    print(f"  标准差: {np.std(log2_data, ddof=1):.4f}")

    sqrt_data = transformer.fit_transform(data.copy(), 'sqrt')
    print(f"\n 平方根变换 (sqrt):")
    print(f"  均值: {np.mean(sqrt_data):.4f}")
    print(f"  标准差: {np.std(sqrt_data, ddof=1):.4f}")
    original_recovered = transformer.inverse_transform(sqrt_data)
    print(f"  逆变换误差: {np.max(np.abs(data - original_recovered)):.2e}")

    reciprocal_data = transformer.fit_transform(data.copy(), 'reciprocal')
    print(f"\n 倒数变换 (reciprocal):")
    print(f"  均值: {np.mean(reciprocal_data):.6f}")
    print(f"  标准差: {np.std(reciprocal_data, ddof=1):.6f}")

    bc_data, bc_lambda, bc_shift = boxcox_transform(data.copy())
    print(f"\n Box-Cox 变换:")
    print(f"  最优 lambda: {bc_lambda:.6f}")
    print(f"  shift 参数: {bc_shift:.6f}")
    print(f"  均值: {np.mean(bc_data):.4f}")
    print(f"  标准差: {np.std(bc_data, ddof=1):.4f}")


def demo_boxcox_lambda():
    print("\n" + "=" * 70)
    print("示例 2: Box-Cox 变换不同 lambda 的效果".center(70))
    print("=" * 70)

    n = 500
    data = np.random.lognormal(mean=2.0, sigma=1.0, size=n)
    print(f"\n原始数据 (n={n}):")
    print(f"  偏度: {_skewness(data):.4f}")
    print(f"  峰度: {_kurtosis(data):.4f}")

    lambdas_to_test = [None, 0.0, 0.5, 1.0, -1.0]

    for lam in lambdas_to_test:
        transformer = DataTransformer()
        transformed = transformer.fit_transform(data.copy(), 'boxcox', lambda_param=lam)
        lam_used = transformer.lambda_
        label = f"lambda={lam_used:.4f}" if lam is None else f"lambda={lam}"
        print(f"\n  {label}:")
        print(f"    偏度: {_skewness(transformed):.4f}")
        print(f"    峰度: {_kurtosis(transformed):.4f}")
        print(f"    shift: {transformer.shift_:.4f}")


def demo_2d_array():
    print("\n" + "=" * 70)
    print("示例 3: 二维数组变换演示".center(70))
    print("=" * 70)

    data_2d = np.random.lognormal(mean=0.5, sigma=0.5, size=(100, 5))
    print(f"\n二维数组形状: {data_2d.shape}")
    print(f"各列原始均值: {np.mean(data_2d, axis=0).round(4)}")

    transformer = DataTransformer()
    transformed_2d = transformer.fit_transform(data_2d.copy(), 'ln')
    print(f"各列 ln 变换后均值: {np.mean(transformed_2d, axis=0).round(4)}")

    recovered_2d = transformer.inverse_transform(transformed_2d)
    error = np.max(np.abs(data_2d - recovered_2d))
    print(f"逆变换最大误差: {error:.2e}")


def demo_variance_stabilizer():
    print("\n" + "=" * 70)
    print("示例 4: 变换效果综合评估".center(70))
    print("=" * 70)

    n = 2000
    x = np.linspace(1, 100, n)
    mean_func = 0.1 * x ** 1.5
    var_func = 0.01 * x ** 2.5
    noise = np.random.normal(0, np.sqrt(var_func), n)
    data = mean_func + noise
    data = np.abs(data) + 0.01

    stabilizer = VarianceStabilizer()
    stabilizer.evaluate(data, bins=10)

    print("\n--- 完整评估摘要 ---")
    print(stabilizer.summary(mode='full'))

    print("\n--- 正态化评估摘要 ---")
    print(stabilizer.summary(mode='normality'))

    print("\n--- 方差稳定化评估摘要 ---")
    print(stabilizer.summary(mode='variance'))

    print("\n--- 变换前后对比表 ---")
    print(stabilizer.comparison_table())

    print("\n--- 智能推荐 ---")
    for goal in ['normality', 'variance_stability', 'all']:
        rec = stabilizer.recommend(goal=goal)
        print(f"\n目标 [{goal}]:")
        print(f"  推荐方法: {rec['recommended_method']}")
        print(f"  说明: {rec['details']}")


def demo_skewness_kurtosis_comparison():
    print("\n" + "=" * 70)
    print("示例 8: 偏度峰度变换效果对比".center(70))
    print("=" * 70)

    data_sets = {
        '对数正态分布': np.random.lognormal(mean=0.5, sigma=0.8, size=2000),
        '指数分布': np.random.exponential(scale=2.0, size=2000) + 0.1,
        '卡方分布(df=3)': np.random.chisquare(df=3, size=2000) + 0.1,
    }

    for name, data in data_sets.items():
        print(f"\n{name} (n={len(data)}):")
        print(f"  原始偏度: {_skewness(data):.4f}, 原始峰度: {_kurtosis(data):.4f}")

        stabilizer = VarianceStabilizer()
        stabilizer.evaluate(data, bins=10)

        rec = stabilizer.recommend(goal='normality')
        print(f"  最佳正态化变换: {rec['recommended_method']}")
        print(f"  变换后偏度: {rec['skewness']:.4f}, 变换后峰度: {rec['kurtosis']:.4f}")
        print(f"  正态得分: {rec['score']:.4f} (越低越接近正态)")


def demo_utility_functions():
    print("\n" + "=" * 70)
    print("示例 5: 便捷函数使用演示".center(70))
    print("=" * 70)

    data = np.random.exponential(scale=2.0, size=200) + 0.1
    print(f"\n原始指数分布数据统计:")
    print(f"  均值: {np.mean(data):.4f}")

    result_ln, shift_ln = log_transform(data, base='ln')
    print(f"\nlog_transform(data, base='ln'):")
    print(f"  shift: {shift_ln:.6f}")
    print(f"  均值: {np.mean(result_ln):.4f}")

    result_sqrt = sqrt_transform(data)
    print(f"\nsqrt_transform(data):")
    print(f"  均值: {np.mean(result_sqrt):.4f}")

    result_reciprocal = reciprocal_transform(data)
    print(f"\nreciprocal_transform(data):")
    print(f"  均值: {np.mean(result_reciprocal):.6f}")

    result_bc, lam, shift = boxcox_transform(data)
    print(f"\nboxcox_transform(data):")
    print(f"  lambda: {lam:.6f}")
    print(f"  shift: {shift:.6f}")
    print(f"  均值: {np.mean(result_bc):.4f}")


def demo_log_with_negative_data():
    print("\n" + "=" * 70)
    print("示例 7: 对数变换自动平移非正数数据".center(70))
    print("=" * 70)

    data_with_neg = np.array([-5.0, 0.0, 1.0, 2.5, 3.7, 10.2])
    print(f"\n原始数据（含负数和零）:")
    print(f"  {data_with_neg}")
    print(f"  最小值: {np.min(data_with_neg):.4f}")
    print(f"  最大值: {np.max(data_with_neg):.4f}")

    transformer = DataTransformer()
    ln_transformed = transformer.fit_transform(data_with_neg.copy(), 'ln')
    print(f"\n 自然对数变换结果:")
    print(f"  shift 平移量: {transformer.shift_:.6f}")
    print(f"  平移后最小值: {np.min(data_with_neg + transformer.shift_):.6f} (应 > 0)")
    print(f"  变换结果: {ln_transformed.round(6)}")

    recovered = transformer.inverse_transform(ln_transformed)
    print(f"\n 逆变换恢复:")
    print(f"  恢复结果: {recovered.round(6)}")
    print(f"  与原始数据最大误差: {np.max(np.abs(data_with_neg - recovered)):.2e}")

    for base in ['ln', 'log10', 'log2']:
        result, shift = log_transform(data_with_neg.copy(), base=base)
        print(f"\n {base} 变换 (便捷函数):")
        print(f"  shift = {shift:.6f}, 结果均值 = {np.mean(result):.6f}")

    data_all_neg = np.array([-10, -5, -3, -1])
    t = DataTransformer()
    result = t.fit_transform(data_all_neg.copy(), 'log10')
    recovered = t.inverse_transform(result)
    print(f"\n 全负数数组 log10 变换:")
    print(f"  原始: {data_all_neg}")
    print(f"  shift: {t.shift_:.6f}")
    print(f"  变换: {result.round(6)}")
    print(f"  恢复: {recovered.round(6)}")
    print(f"  误差: {np.max(np.abs(data_all_neg - recovered)):.2e}")


def demo_edge_cases():
    print("\n" + "=" * 70)
    print("示例 6: 边界情况与错误处理".center(70))
    print("=" * 70)

    print("\n测试非正数数据的对数变换（现在自动平移，不再抛错）:")
    data_with_neg = np.array([1, 2, -1, 4])
    result, shift = log_transform(data_with_neg, 'ln')
    print(f"  成功！shift = {shift:.6f}, 变换结果均值: {np.mean(result):.6f}")

    print("\n测试未拟合就调用 transform:")
    try:
        t = DataTransformer()
        t.transform(np.array([1, 2, 3]))
    except ValueError as e:
        print(f"  捕获异常: {e}")

    print("\n测试无效的变换方法:")
    try:
        t = DataTransformer()
        t.fit(np.array([1, 2, 3]), 'invalid_method')
    except ValueError as e:
        print(f"  捕获异常: {e}")

    print("\n测试包含零的倒数变换:")
    try:
        reciprocal_transform(np.array([1, 0, 3]))
    except ValueError as e:
        print(f"  捕获异常: {e}")


def _skewness(x: np.ndarray) -> float:
    """计算偏度。"""
    m3 = np.mean((x - np.mean(x)) ** 3)
    s3 = np.std(x, ddof=1) ** 3
    return m3 / s3 if s3 != 0 else 0.0


def _kurtosis(x: np.ndarray) -> float:
    """计算峰度。"""
    m4 = np.mean((x - np.mean(x)) ** 4)
    s4 = np.std(x, ddof=1) ** 4
    return m4 / s4 - 3.0 if s4 != 0 else 0.0


if __name__ == '__main__':
    demo_basic_transforms()
    demo_boxcox_lambda()
    demo_2d_array()
    demo_variance_stabilizer()
    demo_utility_functions()
    demo_edge_cases()
    demo_log_with_negative_data()
    demo_skewness_kurtosis_comparison()
    print("\n所有示例执行完成！")
