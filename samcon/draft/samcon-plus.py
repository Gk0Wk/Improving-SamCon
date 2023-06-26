WINDOW_SIZE = 50

def f(reference_motion: list[any], nSample: int, nElite: int):
    # 适应器初始化
    cmaes_list = [None for _ in range()]
    def get_cmaes(iter: int):
        if (cmaes_list[iter] is None):
            cmaes_list[iter] = CMAES()
        return cmaes_list[iter]

    target_trajectory = []
    trial = 0 # 从 1 计数
    iter_window = [0, 1]
    while len(target_trajectory) < len(reference_motion):
        trial += 1
        # 建立窗口
        iter_window[1] = min(len(reference_motion), iter_window[0] + WINDOW_SIZE)
        # 逐帧优化
        for iter in range(*iter_window):
            cmaes = get_cmaes(iter)
            samples = cmaes.sample(nSample)
            # TODO: 模拟、优选
            ...
            target_trajectory.append(...)
            # 重建失败，本轮失败
            if reconstruction_failed:
                iter_window[1] = iter + 1
                break
            # 更新分布，最后一个(窗口末端)不用
            if iter < iter_window[1] - 1:
                cmaes.update(...)
        iter_window[1] -= 1
        # 窗口移动
        for iter in range(*iter_window):
            if not is_good_engouth(iter):
                break
            iter_window[0] += 1
