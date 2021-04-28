from abc import ABC, abstractmethod
from multiprocessing import Queue, cpu_count, Process

import numpy as np
from synet.networks.base import BaseNetwork


class BaseMeasure(ABC):
    @staticmethod
    def run_jobs(jobs, target, n_jobs=-1):
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
                target=target,
                args=(job_queue, result_queue, pid),
                daemon=True,
            )
            for pid in range(n_jobs)
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

    def entropy_t(self, networks, dt, n_jobs=1):
        if isinstance(networks, BaseNetwork):
            return self._entropy_t(networks, dt)

        res = []
        if n_jobs == 1:
            for net in networks:
                res.append(self._entropy_t(net, dt))
        else:
            jobs = [
                {
                    "class": self.__class__,
                    "init_kwargs": self.todict(),
                    "entropy_kwargs": {
                        "net": networks[i_net],
                        "dt": dt,
                    },
                    "net_id": i_net,
                }
                for i_net in range(len(networks))
            ]
            all_res = self.run_jobs(jobs, target=_simulate_worker_t,
                                    n_jobs=n_jobs)
            sim_ids = np.argsort([r[0]["net_id"] for r in all_res])
            res = [all_res[sim_id][1] for sim_id in sim_ids]
        return res

    def entropy_dt(self, networks, max_dt, n_jobs=1):
        if isinstance(networks, BaseNetwork):
            return self._entropy_dt(networks, max_dt)

        for net in networks:
            assert max_dt <= net.n_events

        res = []

        if n_jobs == 1:
            for net in networks:
                res.append(self._entropy_dt(net, max_dt))
        else:
            jobs = [
                {
                    "class": self.__class__,
                    "init_kwargs": self.todict(),
                    "entropy_kwargs": {
                        "net": networks[i_net],
                        "max_dt": max_dt,
                    },
                    "net_id": i_net,
                }
                for i_net in range(len(networks))
            ]
            all_res = self.run_jobs(jobs, target=_simulate_worker_dt,
                                    n_jobs=n_jobs)
            sim_ids = np.argsort([r[0]["net_id"] for r in all_res])
            res = [all_res[sim_id][1] for sim_id in sim_ids]
        return res

    def _entropy_t(self, net, dt):
        last_events = np.full(net.n_agents, -1, dtype=int)
        entropy_avg = np.zeros(net.n_events)
        entropy_counts = np.zeros(net.n_events, dtype=int)
        n_events = net.n_events
        for t_start in range(0, n_events):
            agents = net.participants[t_start]

            t_end = min(t_start + dt, n_events)
            entropy = self.measure_entropy(net, t_start, t_end)
            all_prev_dt = t_start - last_events[agents]
            for prev_dt in all_prev_dt:
                inter_start = max(-prev_dt, -dt)
                inter_end = min(0, t_end-t_start-dt)
                src_start = inter_start + dt
                src_end = inter_end + dt
                dst_start = inter_start + dt//2 + t_start
                dst_end = inter_end + dt//2 + t_start
                entropy_avg[dst_start:dst_end] += entropy[src_start:src_end]
                entropy_counts[dst_start:dst_end] += 1
            last_events[agents] = t_start
        return entropy_avg

    def _entropy_dt(self, net, max_dt):
        n_events = net.n_events
        eq_start = n_events//10
        eq_end = 9*n_events//10

        entropy_avg = np.zeros(max_dt)
        for t_start in range(eq_start, eq_end-max_dt):
            t_end = t_start + max_dt
            entropy = self.measure_entropy(net, t_start, t_end)
            entropy_avg += entropy
        return entropy_avg/(eq_end-eq_start-max_dt)

    @abstractmethod
    def measure_entropy(self, net, start, end):
        raise NotImplementedError

    def todict(self):
        return {}


class BasePaintEntropy(BaseMeasure):
    def _entropy_dt(self, net, max_dt):
        n_events = net.n_events

        last_events = np.full(net.n_agents, -1, dtype=int)
        entropy_avg = np.zeros(max_dt+1)
        counts = np.zeros(max_dt+1, dtype=int)
        for t_start in range(n_events):
            t_end = min(t_start + max_dt, n_events)
            agents = net.participants[t_start]
            cur_entropy = self.measure_entropy(net, t_start, t_end)
            entropy = np.zeros(max_dt+1)
            entropy[:len(cur_entropy)] = cur_entropy
            all_prev_dt = t_start - last_events[agents]
            for prev_dt in all_prev_dt:
                new_sum = np.cumsum(entropy)
                new_counts = np.zeros(len(entropy), dtype=int)
                new_counts[:len(cur_entropy)] = 1
                new_counts[0] = 0
                new_counts = np.cumsum(new_counts)
                if prev_dt < max_dt:
                    new_sum[prev_dt-max_dt:] -= new_sum[:max_dt-prev_dt]
                    new_counts[prev_dt-max_dt:] -= new_counts[:max_dt-prev_dt]
                entropy_avg += new_sum
                counts += new_counts
            last_events[agents] = t_start

#         print(counts[1], n_events*5)
#         assert counts[0] == n_events*5
        assert counts[1] == n_events*5
        norm = (n_events-np.arange(max_dt+1))
        return entropy_avg/norm#, counts


def _simulate_worker_dt(job_queue, output_queue, pid):
    while True:
        job = job_queue.get(block=True)
        if job is None:
            break
        cls = job["class"]
        init_kwargs = job["init_kwargs"]
        sim_kwargs = job["entropy_kwargs"]

        measure = cls(**init_kwargs)
        results = measure._entropy_dt(**sim_kwargs)
        output_queue.put((job, results))


def _simulate_worker_t(job_queue, output_queue, pid):
    while True:
        job = job_queue.get(block=True)
        if job is None:
            break
        cls = job["class"]
        init_kwargs = job["init_kwargs"]
        sim_kwargs = job["entropy_kwargs"]

        measure = cls(**init_kwargs)
        results = measure._entropy_t(**sim_kwargs)
        output_queue.put((job, results))
