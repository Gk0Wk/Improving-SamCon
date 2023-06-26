import numpy as np


class CMAES():
    def __init__(self, dim: int,
                 sigma=0.3,  # 步长
                 d_sigma=1.09,  # simga 学习率(mu更新) 的迭代衰减参数
                 init_means: list[float] | None = None,
                 alpha_mu: float | list[float] | None = None,
                 alpha_sigma: float | None = None,
                 alpha_cp: float | None = None, alpha_c1: float | None = None,
                 alpha_clambda: float | None = None,
                 # 几个学习率
                 # 在自动计算采样数量的情况下的采样策略(=1就是一倍，=2就是两倍，以此类推)，如果觉得一次性采样的点太少了，可以调大这个值，以减少代数
                 sample_size_scale: int = 1,
                 sample_size: int | None = None, elite_size: int | None = None):
        self.generation = 0
        self.lokced = False
        self.sigma = sigma

        # 初始化均值(初始点) #
        self.means = np.array(
            init_means) if init_means is not None else np.random.rand(dim)

        # 自动选择数量 #
        self.sample_size = int(
            sample_size if sample_size is not None else (4 + np.floor(3 * np.log(dim) * sample_size_scale)))
        self.elite_size = int(
            elite_size if elite_size is not None else np.floor(sample_size / 2))

        # 样本权重 #
        if alpha_mu is None:
            weights: np.ndarray = np.log(
                (sample_size + 1) / 2) - np.log(np.arange(self.elite_size) + 1)
        elif isinstance(alpha_mu, list):
            weights: np.ndarray = np.array(
                [alpha_mu[i] if i < len(alpha_mu) else 1 for i in range(sample_size)])
        else:
            weights = np.full([1, dim], float(alpha_mu))
        self.weights = weights / weights.sum()

        # 自动选择学习率 #
        # 平方不等式 mueff ≤ elite_size即mu当且仅当weight相等时取等，
        # 所以近似就是elite_size，但是为什么要用mueff替换elite_size呢？
        self.mueff = float((self.weights.sum() ** 2) / (self.weights ** 2).sum())
        self.alpha_cp = float(
            alpha_cp if alpha_cp is not None else (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim))
        self.alpha_sigma = float(alpha_sigma if alpha_sigma is not None else (
            self.mueff + 2) / (dim + self.mueff + 5))
        self.alpha_c1 = float(
            alpha_c1 if alpha_c1 is not None else 2 / ((dim + 1.3) ** 2 + self.mueff))
        self.alpha_clambda = float(alpha_clambda if alpha_clambda is not None else min(1 - self.alpha_c1, 2 * (
            self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff)))
        self.d_sigma = float(d_sigma if d_sigma is not None else (
            1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.alpha_sigma))

        # 协方差矩阵初始化 #
        self.C = np.identity(dim)
        self.invsqrtC = np.identity(dim)  # I的逆开方还是I

        # 确定 invsqrtC 的更新频率, 每代都更新成本偏高 #
        self.invsqrtC_update_step = int(
            max(np.ceil(1 / ((self.alpha_c1 + self.alpha_clambda) * dim * 10)), 1))

        # E||N(0,I)||
        self.chiDim = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # 初始化路径
        self.p_c = np.zeros([dim, 1])
        self.p_sigma = np.zeros([dim, 1])

    def sample(self):
        """
        采样
        """
        return np.random.multivariate_normal(self.means, self.sigma*self.sigma*self.C, self.sample_size)

    def update(self, elite_in_order: np.ndarray):
        """
        使用精英样本进行更新，样本应当是行向量构成的矩阵且经过排序，优者在前
        """
        if self.locked:
            return

        # 检查尺寸
        if elite_in_order.shape[0] != self.elite_size or elite_in_order.shape[1] != self.dim:
            raise ValueError(f'elite_in_order shape should be [{self.elite_size}, {self.dim}]')

        # 更新期望
        means_old = self.means
        self.means = self.weights @ elite_in_order

        # 路径更新
        tmp = (self.means - means_old) / self.sigma
        self.p_sigma: np.ndarray = (1-self.alpha_sigma)*self.p_sigma + \
            np.sqrt(self.alpha_sigma*(2-self.alpha_sigma)*self.mueff)*self.invsqrtC@tmp
        n_p_sigma = np.linalg.norm(self.p_sigma)
        hsig = 1. if n_p_sigma / np.sqrt(1 - (1 - self.alpha_sigma)**(
            2*self.generation)*self.chiDim) < (1.4 + 2/(self.dim+1)) else 0.  # ???这是对路径做了什么判断
        self.p_c = (1-self.alpha_cp)*self.p_c + hsig * \
            np.sqrt(self.alpha_sigma*(2-self.alpha_cp)*self.mueff)*tmp
        del tmp

        # sigma 学习率更新
        self.sigma *= np.exp((n_p_sigma / self.chiDim - 1) * self.alpha_sigma / self.d_sigma)

        # 协方差矩阵C更新
        artmp = (elite_in_order - means_old) / self.sigma
        self.C = (1 - self.alpha_c1 - self.alpha_clambda) * self.C + \
            self.alpha_c1 * (self.p_c * self.p_c.T + (1 - hsig) * self.alpha_cp * (2 - self.alpha_cp) * self.C) + \
            self.alpha_clambda * artmp.T@np.diag(self.weights)@artmp
        del artmp

        # invsqrtC 的更新
        if self.generation % self.invsqrtC_update_step == 0:
            D, B = np.linalg.eigh(self.C, 'U')
            D: np.ndarray = np.sqrt(D.real)
            self.invsqrtC = B @ np.diag(1/D) @ B.T

        self.generation += 1

    def lock(self):
        """
        锁定优化器释放内存, 后续就不能 update 了，只能 sample
        """
        self.locked = True
        del self.elite_size
        del self.weights
        del self.mueff
        del self.alpha_cp
        del self.alpha_sigma
        del self.alpha_c1
        del self.alpha_clambda
        del self.d_sigma
        del self.invsqrtC
        del self.invsqrtC_update_step
        del self.chiDim
        del self.p_c
        del self.p_sigma
