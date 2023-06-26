def initialize():
    ...


def pick_start_state(samples: list[sm_ele]) -> any:
    ...


def select_samples(samples: list[tuple[sample, float, state]], nSave: int) -> list[sm_ele]:
    ...


def job_of_master(origin_motion, nIter: int, nSample: int, nSave: int):
    SM: list[list[sm_ele]] = [initialize()]
    for i in range(1, nIter + 1):
        samples: list[tuple[sample, float, state]] = []
        # 阻塞，线程池执行指定任务 在SM[i-1]中取一个，生成nSample个样本
        pool.batch_work(nSample,
                        on_begin=lambda: ['WORK', pick_start_state(SM[i - 1]), i],
                        on_finish=lambda data: samples.append(data)
                        )
        SM.append(select_samples(samples, nSave))
    return search_beat_path(SM)


def job_of_worker():
    while True:
        cmd, start_state, curr_iter = receive_data_from_master()
        if cmd == 'WORK':
            sample = generate_sample(origin_motion, curr_iter)
            m, end_state = simulate(start_state, sample, origin_motion)
            cost = calculate_cost(m, origin_motion)
            send_result_to_master((sample, cost, end_state))
        elif cmd == 'CLOSE':
            break
        else:
            raise RuntimeError(f'Got unrecognized cmd {cmd}')
    send_result_to_master('CLOSE')
