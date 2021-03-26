from abc import ABC, abstractmethod
from multiprocessing import Queue, cpu_count, Process


class BaseProcess(ABC):
    @abstractmethod
    def simulate(self, net, start=1, end=None):
        raise NotImplementedError

    @staticmethod
    def run_jobs(jobs, n_jobs=-1):
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            n_jobs = cpu_count()

        queue_size = len(jobs)
        job_queue = Queue()
        result_queue = Queue()

        for job in jobs:
            job_queue.put(job)

        worker_procs = [
            Process(
                target=_simulate_worker,
                args=(job_queue, result_queue),
                daemon=True,
            )
            for _ in range(n_jobs)
        ]
        for proc in worker_procs:
            proc.start()

        results = []
        for _ in range(queue_size):
            results.append(result_queue.get())
        for _ in range(n_jobs):
            job_queue.put(None)

        for proc in worker_procs:
            proc.join()

        return results


def _simulate_worker(job_queue, output_queue):
    while True:
        job = job_queue.get(block=True)
        if job is None:
            break
        cls = job["class"]
        try:
            args = job["args"]
        except KeyError:
            args = []
        try:
            kwargs = job["kwargs"]
        except KeyError:
            kwargs = {}

        sim_args = job["sim_args"]
        process = cls(*args, **kwargs)
        results = process.run_simulation(*sim_args)
        output_queue.put((job, results))
