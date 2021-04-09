from abc import ABC, abstractmethod
from multiprocessing import Queue, cpu_count, Process

import numpy as np


class BaseProcess(ABC):
#     @abstractmethod
#     def simulate(self, net, start=1, end=None):
#         raise NotImplementedError
    @staticmethod
    def run_jobs(net, jobs, n_jobs=-1):
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            n_jobs = cpu_count()

        queue_size = len(jobs)
        job_queue = Queue(maxsize=1000)
        result_queue = Queue()

        worker_procs = [
            Process(
                target=_simulate_worker,
                args=(job_queue, result_queue, pid, net),
                daemon=True,
            )
            for pid in range(n_jobs)
        ]
        for proc in worker_procs:
            proc.start()

        for job in jobs:
            job_queue.put(job)

        results = []
        for _ in range(queue_size):
            results.append(result_queue.get())
        for _ in range(n_jobs):
            job_queue.put(None)

        for proc in worker_procs:
            proc.join()

        return results

    def simulate(self, net, start=1, end=None, n_sim=1, n_jobs=1, seed=None):
        if end is None:
            end = net.n_events

        if n_jobs == -1:
            n_jobs = cpu_count()

        if seed is not None:
            np.random.seed(seed)

        all_seeds = np.random.randint(0, 12365243, size=n_sim)

        res = np.zeros(end-start, dtype=float)
        if n_jobs == 1:
            for i_sim in range(n_sim):
                res += self._simulate(net, start, end, seed=all_seeds[i_sim])
        else:
            jobs = [
                {
                    "class": self.__class__,
                    "init_kwargs": self.todict(),
                    "sim_kwargs": {
#                         "net": net,
                        "start": start,
                        "end": end,
                        "seed": seed,
                    }
                }
                for seed in all_seeds
            ]
            all_res = self.run_jobs(net, jobs, n_jobs=n_jobs)
            for t in all_res:
                res += t[1]
        return res/n_sim

    def simulate_dt(self, net, dt, n_sim=1, n_jobs=1, seed=None):
        if seed is not None:
            np.random.seed(seed)

        assert dt <= net.n_events

        start_range = net.n_events-dt

        all_seeds = np.random.randint(0, 12365243, size=n_sim)
        res = np.zeros((n_sim, dt), dtype=float)
        if n_jobs == 1:
            for i_sim in range(n_sim):
                start = int(i_sim*start_range/n_sim)
                end = start + dt
                res[i_sim][:] = self._simulate(net, start, end, all_seeds[i_sim])
        else:
            starts = [int(i_sim*start_range/n_sim) for i_sim in range(n_sim)]
            jobs = [
                {
                    "class": self.__class__,
                    "init_kwargs": self.todict(),
                    "sim_kwargs": {
#                         "net": net,
                        "start": starts[i_sim],
                        "end": starts[i_sim] + dt,
                        "seed": seed,
                    },
                    "sim_id": i_sim,
                }
                for i_sim in range(n_sim)
            ]
            all_res = self.run_jobs(net, jobs, n_jobs=n_jobs)
            for i_sim, t in enumerate(all_res):
                res[i_sim][:] = t[1]
        return res

    def _simulate(self, net, start=0, end=None, seed=None):
        raise NotImplementedError

    def todict(self):
        return {}


def _simulate_worker(job_queue, output_queue, pid, net=None):
    while True:
        job = job_queue.get(block=True)
        if job is None:
            break
        cls = job["class"]
        init_kwargs = job["init_kwargs"]
        sim_kwargs = job["sim_kwargs"]

        process = cls(**init_kwargs)
        results = process.simulate(net, **sim_kwargs)
        output_queue.put((job, results))
