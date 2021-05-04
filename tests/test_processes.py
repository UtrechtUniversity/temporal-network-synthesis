import pytest
import numpy as np

from synet.process.delay import DelayProcess
from synet.process.disease import DiseaseProcess
from synet.process.majority import MajorityProcess
from synet.process.predator import PredatorProcess
from synet.networks.random import random_network


@pytest.mark.parametrize(
    "process_class", [DelayProcess, DiseaseProcess, MajorityProcess,
                    PredatorProcess]
#     "process_class", [DiseaseProcess]
)
def test_process(process_class):
    process = process_class()
    seed = 129837
    dt = 20
    networks = [random_network(n_events=300, n_agents=20, event_size=5)
                for _ in range(2)]

    proc_results = process.simulate(networks[0], n_sim=10, seed=seed)
    proc_results_parallel = process.simulate(networks[0], n_sim=10, seed=seed,
                                             n_jobs=2)
    assert np.allclose(proc_results, proc_results_parallel)

    proc_results_dt = process.simulate_dt(
        networks, dt, n_sim=10, seed=seed)
    proc_results_dt_parallel = process.simulate_dt(
        networks, dt, n_sim=10, seed=seed, n_jobs=2)

    assert np.allclose(proc_results_dt, proc_results_dt_parallel)
    proc_results_dt_parallel_single = process.simulate_dt(
        networks[0], dt, n_sim=10, seed=seed, n_jobs=2
    )
    proc_results_dt_single = process.simulate_dt(
        networks[0], dt, n_sim=10, seed=seed, n_jobs=1
    )
    assert np.allclose(proc_results_dt_parallel_single, proc_results_dt_single)
    #     proc_results_indiv = [process.simulate(net) for net in networks]

