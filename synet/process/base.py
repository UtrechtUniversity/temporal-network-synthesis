from abc import ABC
from multiprocessing import Queue, cpu_count, Process

import numpy as np

from synet.networks.base import BaseNetwork


class BaseProcess(ABC):
    """Base class for general processes living on a network.

    In particular a simulation process that works with BaseNetwork.
    """
    @staticmethod
    def run_jobs(jobs, target, net, n_jobs=-1):
        """Parallel running of measure jobs.

        Arguments
        ---------
        jobs: dict
            Jobs to be processed in parallel
        target: function
            Function that executes these jobs.
        net: (iterable, synet.networks.BaseNetwork)
            Either a single network or a list/array of them.
        n_jobs: int
            Number of processes to use, -1 defaults to the number of cores.

        Returns
        -------
        results: list
            Unordered results of the computation.
        """
        if n_jobs is None:
            n_jobs = 1
        elif n_jobs == -1:
            n_jobs = cpu_count()

        queue_size = len(jobs)
        job_queue = Queue(maxsize=1000)
        result_queue = Queue()

        args = (job_queue, result_queue, net)

        worker_procs = [
            Process(
                target=target,
                args=args,
                daemon=True,
            )
            for _ in range(n_jobs)
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

    def simulate(self, net, start=0, end=None, n_sim=1, n_jobs=1, seed=None):
        """Simulate the process over a portion of the network.

        Arguments
        ---------
        net: synet.networks.BaseNetwork
            Network to simulate the process on over a specific interval.
        start: int
            Start of the interval.
        end: int
            End of the interval, if None -> n_events.
        n_sim: int
            Number of simulations over the same interval.
        n_jobs: int
            Number of processes to use.
        seed: int
            Use this seed to seed the simulations with.
        """
        if end is None:
            end = net.n_events

        if n_jobs == -1:
            n_jobs = cpu_count()

        if seed is not None:
            np.random.seed(seed)

        all_seeds = np.random.randint(0, 12365243, size=n_sim)

        res = np.zeros(end-start, dtype=float)
        if n_jobs == 1:
            # Single process
            for i_sim in range(n_sim):
                res += self._simulate(net, start, end, seed=all_seeds[i_sim])
        else:
            # Multi process
            jobs = [
                {
                    "class": self.__class__,
                    "init_kwargs": self.todict(),
                    "sim_kwargs": {
                        "start": start,
                        "end": end,
                        "seed": seed,
                    }
                }
                for seed in all_seeds
            ]
            all_res = self.run_jobs(jobs, _simulate_worker, net, n_jobs=n_jobs)
            for t in all_res:
                res += t[1]
        return res/n_sim

    def simulate_dt(self, network, dt, n_sim=1, n_jobs=1, seed=None):
        """Simulate the process over a fixed period of time.

        Varies the starting point of the simulation while keeping the
        time difference between starting and end point the same.

        Arguments
        ---------
        network: (iterable, BaseNetwork)
            Either a network or a list of networks to simulate on.
        dt: int
            Time difference between start of the simulation and end.
        n_sim: int
            Number of simulations to be done.
        n_jobs: int
            Number of worker processes to use.
        seed: int
            Seed to simulate with.
        """
        if not isinstance(network, BaseNetwork):
            return self._simulate_dt_networks(network, dt, n_sim=n_sim,
                                              n_jobs=n_jobs, seed=seed)
        return self._simulate_dt(network, dt, n_sim=n_sim, n_jobs=n_jobs,
                                 seed=seed)

    def _simulate_dt_networks(self, networks, dt, n_sim=1, n_jobs=1,
                              seed=None):
        """Simulate multiple networks over a fixed period of time."""
        n_network = len(networks)
        np.random.seed(seed)
        all_seeds = np.random.randint(0, 129873984, size=n_sim)
        jobs = [
            {
                "class": self.__class__,
                "init_kwargs": self.todict(),
                "sim_kwargs": {
                    "dt": dt,
                    "n_sim": n_sim,
                    "seed": all_seeds,
                    "n_jobs": 1,
                },
                "net_id": net_id,
            }
            for net_id in range(n_network)
        ]
        results = self.run_jobs(jobs, _simulate_network_worker, net=networks,
                                n_jobs=n_jobs)
        sorted_res = sorted(results, key=lambda x: x[0]["net_id"])
        return [r[1] for r in sorted_res]

    def _simulate_dt(self, net, dt, n_sim=1, n_jobs=1, seed=None):
        "Simulate a single network over a fixed period of time."
        assert dt <= net.n_events

        start_range = net.n_events-dt

        # Generate random seeds.
        np.random.seed(seed)
        all_seeds = np.random.randint(0, 12365243, size=n_sim)
        res = np.zeros((n_sim, dt), dtype=float)
        if n_jobs == 1:
            for i_sim in range(n_sim):
                start = int(i_sim*start_range/n_sim)
                end = start + dt
                res[i_sim][:] = self._simulate(net, start, end,
                                               all_seeds[i_sim])
        else:
            starts = [int(i_sim*start_range/n_sim) for i_sim in range(n_sim)]
            jobs = [
                {
                    "class": self.__class__,
                    "init_kwargs": self.todict(),
                    "sim_kwargs": {
                        "start": starts[i_sim],
                        "end": starts[i_sim] + dt,
                        "seed": all_seeds[i_sim],
                    },
                    "sim_id": i_sim,
                }
                for i_sim in range(n_sim)
            ]
            all_res = self.run_jobs(jobs, _simulate_worker, net, n_jobs=n_jobs)
            for cur_res in all_res:
                sim_id = cur_res[0]["sim_id"]
                res[sim_id][:] = cur_res[1]
        return res

    def _simulate(self, net, start=0, end=None, seed=None):
        "Functionality implemented by the process."
        raise NotImplementedError

    def todict(self):
        "Return the parameters of the simulation."
        return {}


def _simulate_worker(job_queue, output_queue, net):
    "Process simulation worker for a single network"
    while True:
        job = job_queue.get(block=True)
        if job is None:
            break
        cls = job["class"]
        init_kwargs = job["init_kwargs"]
        sim_kwargs = job["sim_kwargs"]

        process = cls(**init_kwargs)
        results = process._simulate(net, **sim_kwargs)
        output_queue.put((job, results))


def _simulate_network_worker(job_queue, output_queue, networks):
    "Process simulation worker for multiple networks."
    while True:
        job = job_queue.get(block=True)
        if job is None:
            break
        cls = job["class"]
        init_kwargs = job["init_kwargs"]
        sim_kwargs = job["sim_kwargs"]
        net = networks[job["net_id"]]

        process = cls(**init_kwargs)
        results = process.simulate_dt(net, **sim_kwargs)
        output_queue.put((job, results))
