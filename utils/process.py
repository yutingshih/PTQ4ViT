import multiprocessing
import os
import time

def process(pid, experiment_process, args_queue, n_gpu):
    """
    worker process.
    """
    gpu_id = pid % n_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    tot_run = 0
    while args_queue.qsize():
        test_args = args_queue.get()
        print(f"Run {test_args} on pid={pid} gpu_id={gpu_id}")
        experiment_process(**test_args)
        time.sleep(0.5)
        tot_run += 1
        # run_experiment(**args)
    print(f"{pid} tot_run {tot_run}")


def multiprocess(experiment_process, cfg_list=None, n_gpu=6):
    """
    run experiment processes on "n_gpu" cards via "n_gpu" worker process.
    "cfg_list" arranges kwargs for each test point, and worker process will fetch kwargs and carry out an experiment.
    """
    args_queue = multiprocessing.Queue()
    for cfg in cfg_list:
        args_queue.put(cfg)

    ps = []
    for pid in range(n_gpu):
        p = multiprocessing.Process(
            target=process, args=(pid, experiment_process, args_queue, n_gpu)
        )
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
