# Requires python >= 3.10
import numpy as np
from typing import Any, Callable

# ndarray 的批量操作
# eval_v = np.vectorize(lambda x:x+1); y=eval_v(x)  逐个元素进行操作
# np.apply_along_axis 按某个方向进行向量化操作


# 简单高斯分布进化
# 问题: 标准差与上一轮分布相关，因此只能处理静态情况，如果算法需要在中途改变置信，就没法适应
# 以及，估计是非缩放的，如果某些维度维度分量对函数的影响很大、变化很剧烈或恰巧相反，就会导致难以收敛
# 以及，下一轮的分布是基于上一轮的样本得到的，是有偏估计，其准确程度取决于上一轮样本的数量
def simple_gussian_es(dim: int, fitness: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
                      init_means: list[float] | None = None,
                      init_scales: list[float] | None = None,
                      # 在自动计算采样数量的情况下的采样策略(=1就是一倍，=2就是两倍，以此类推)，如果觉得一次性采样的点太少了，可以调大这个值，以减少代数
                      sample_size_scale: int = 1,
                      sample_size: int | None = None, elite_size: int | None = None,
                      max_generation=1000, stopfitness=1e-10,
                      on_generation_finish: Callable | None = None
                      ) -> tuple[np.ndarray[Any, np.dtype[np.float64]],  np.ndarray[Any, np.dtype[np.float64]]]:
    # 自动初始化均值和方差
    means = np.array(init_means) if init_means is not None else np.zeros(dim)
    scale = np.array(init_scales) if init_scales is not None else np.ones_like(means)

    # 自动选择数量
    sample_size = int(sample_size if sample_size is not None else (
        4 + np.floor(3 * np.log(dim) * sample_size_scale)))
    elite_size = int(elite_size if elite_size is not None else sample_size / 2)

    for generation in range(1, max_generation + 1):
        # 采样(行向量)
        x = np.random.normal(means, scale, (sample_size, len(means)))
        # 评估
        fitnesses: np.ndarray = np.apply_along_axis(
            fitness, 1, x)  # 批量逐行, 向量化操作
        # 前 elite_size 个 fitness 最小的 x 作为精英样本
        elite_index = np.argpartition(fitnesses, elite_size)[
            :elite_size]  # np.argpartition 前k个排序，获得索引
        elite_x = x[elite_index]
        best_fitness = float(fitnesses[elite_index[0]])
        del x, fitnesses
        # 更新高斯分布
        means = elite_x.mean(0)
        scale = elite_x.std(0)
        if on_generation_finish is not None:
            on_generation_finish(generation=generation, max_generation=max_generation,
                                 best_fitness=best_fitness, means=means, scale=scale)
        if best_fitness < stopfitness:
            break
    return (means, scale)


# 协方差矩阵自适应进化 CMA-ES
# 参考了 https://en.wikipedia.org/wiki/CMA-ES 的 Matlab 代码
# 写完才发现有人写过了: https://gist.github.com/yukoba/731bb555a7dc79cdfecf33904ee4e043 不过这个是列向量版本的
def cma_es(dim: int, fitness: Callable[[np.ndarray[Any, np.dtype[np.float64]]], float],
           init_means: list[float] | None = None,
           init_cov: np.ndarray | None = None,
           sigma=0.3,  # 步长
           d_sigma=1.09,  # simga 学习率(mu更新) 的迭代衰减参数
           # 几个学习率
           alpha_mu: float | list[float] | None = None, alpha_sigma: float | None = None, alpha_cp: float | None = None,
           alpha_c1: float | None = None, alpha_clambda: float | None = None,
           # 在自动计算采样数量的情况下的采样策略(=1就是一倍，=2就是两倍，以此类推)，如果觉得一次性采样的点太少了，可以调大这个值，以减少代数
           sample_size_scale: int = 1,
           sample_size: int | None = None, elite_size: int | None = None, max_generation=1000, stopfitness=1e-10,
           on_generation_finish: Callable | None = None
           ) -> tuple[np.ndarray[Any, np.dtype[np.float64]],  np.ndarray[Any, np.dtype[np.float64]]]:
    # 初始化均值(初始点)
    means = np.array(init_means) if init_means is not None else np.zeros(dim)

    # 自动选择数量
    sample_size = int(sample_size if sample_size is not None else (
        4 + np.floor(3 * np.log(dim) * sample_size_scale)))
    elite_size = int(
        elite_size if elite_size is not None else np.floor(sample_size / 2))

    # 样本权重
    if alpha_mu is None:
        weights: np.ndarray = np.log(
            (sample_size + 1) / 2) - np.log(np.arange(elite_size) + 1)
    elif isinstance(alpha_mu, list):
        weights: np.ndarray = np.array(
            [alpha_mu[i] if i < len(alpha_mu) else 1 for i in range(sample_size)])
    else:
        weights = np.full([1, dim], float(alpha_mu))
    weights /= weights.sum()

    # 自动选择学习率
    # 平方不等式 mueff ≤ elite_size即mu当且仅当weight相等时取等，所以近似就是elite_size，但是为什么要用mueff替换elite_size呢？
    mueff = float((weights.sum() ** 2) / (weights**2).sum())
    alpha_cp = float(alpha_cp if alpha_cp is not None else (
        4+mueff/dim)/(dim+4+2*mueff/dim))
    alpha_sigma = float(
        alpha_sigma if alpha_sigma is not None else (mueff+2)/(dim+mueff+5))
    alpha_c1 = float(alpha_c1 if alpha_c1 is not None else 2 /
                     ((dim+1.3)**2+mueff))
    alpha_clambda = float(alpha_clambda if alpha_clambda is not None else min(
        1-alpha_c1, 2*(mueff-2+1/mueff)/((dim+2)**2+mueff)))
    d_sigma = float(d_sigma if d_sigma is not None else (
        1 + 2*max(0, np.sqrt((mueff-1)/(dim+1))-1)+alpha_sigma))

    # 协方差矩阵初始化 #
    if init_cov is None:
        C = np.identity(dim)
        invsqrtC = np.identity(dim)  # I的逆开方还是I
    else:
        C = init_cov
        D, B = np.linalg.eigh(C, 'U')
        D: np.ndarray = np.sqrt(D.real)
        invsqrtC = B @ np.diag(1/D) @ B.T

    # 确定 invsqrtC 的更新频率, 每代都更新成本偏高
    invsqrtC_update_step = int(
        max(np.ceil(1 / ((alpha_c1 + alpha_clambda) * dim * 10)), 1))

    # E||N(0,I)||
    chiDim = np.sqrt(dim)*(1-1/(4*dim)+1/(21*dim**2))

    # 初始化路径
    p_c = np.zeros([dim, 1])
    p_sigma = np.zeros([dim, 1])

    # 迭代
    for generation in range(1, max_generation + 1):
        # 采样(行向量)
        x = np.random.multivariate_normal(means, sigma*sigma*C, sample_size)
        # 评估
        fitnesses = np.apply_along_axis(fitness, 1, x)  # 批量逐行, 向量化操作
        # 前 elite_size 个 fitness 最小的 x 作为精英样本
        elite_index = np.argpartition(fitnesses, elite_size)[
            :elite_size]  # np.argpartition 前k个排序，获得索引
        elite_x = x[elite_index]
        best_fitness = float(fitnesses[elite_index[0]])
        del elite_index, x, fitnesses

        # 更新期望
        means_old = means
        means = weights @ elite_x

        # 路径更新
        tmp = (means - means_old) / sigma
        p_sigma: np.ndarray = (1-alpha_sigma)*p_sigma + \
            np.sqrt(alpha_sigma*(2-alpha_sigma)*mueff)*invsqrtC@tmp
        n_p_sigma = np.linalg.norm(p_sigma)
        hsig = 1. if n_p_sigma / np.sqrt(1 - (1 - alpha_sigma)**(
            2*generation)*chiDim) < (1.4 + 2/(dim+1)) else 0.  # ???这是对路径做了什么判断
        p_c = (1-alpha_cp)*p_c + hsig * \
            np.sqrt(alpha_sigma*(2-alpha_cp)*mueff)*tmp
        del tmp

        # sigma 学习率更新
        sigma *= np.exp((n_p_sigma / chiDim - 1) * alpha_sigma / d_sigma)

        # 协方差矩阵C更新
        artmp = (elite_x - means_old) / sigma
        C = (1 - alpha_c1 - alpha_clambda) * C + \
            alpha_c1 * (p_c * p_c.T + (1 - hsig) * alpha_cp * (2 - alpha_cp) * C) + \
            alpha_clambda * artmp.T@np.diag(weights)@artmp
        del artmp

        # invsqrtC 的更新
        if generation % invsqrtC_update_step == 0:
            D, B = np.linalg.eigh(C, 'U')
            D: np.ndarray = np.sqrt(D.real)
            invsqrtC = B @ np.diag(1/D) @ B.T

        if on_generation_finish is not None:
            on_generation_finish(generation=generation, max_generation=max_generation,
                                 best_fitness=best_fitness, means=means, scales=C)

        # 精度够了或者看着不太能收敛就停止
        if best_fitness < stopfitness or D.max() > 1e7 * D.min():
            break

    return (means, C)

# 例子，求函数的局部实根
# 可以看出，在同样的采样策略下，CMA-ES的收敛速度比简单高斯策略快很多, 采样点过少时简单高斯甚至无法收敛


def print_generation(**kwargs):
    print('\t'.join([
        f'generation: {kwargs["generation"]}/{kwargs["max_generation"]}',
        f' | best_fitness={kwargs["best_fitness"]}',
        f' | means: {kwargs["means"]}',
    ]))


print('Simple Gussian:')
means, _ = simple_gussian_es(1, lambda x: abs(
    x[0]**5 + 3.4*4**x[0] - 6*x[0]**2 + 4*x[0] + 1), on_generation_finish=print_generation)
print(means)

print('CMA-ES:')
means, _ = cma_es(1, lambda x: abs(
    x[0]**5 + 3.4*4**x[0] - 6*x[0]**2 + 4*x[0] + 1), on_generation_finish=print_generation)
print(means)
