from collections import defaultdict

import numpy as np

from synet.analysis import entropy_dt


def get_measure(measure_name=None):
    from synet.config import measures
    if isinstance(measure_name, (tuple, list, np.ndarray)):
        return [measures[n] for n in measure_name]
    return measures[measure_name]


def apply_measures(networks, measures=None, max_dt=100):
    if measures is None:
        from synet.config import measures

    if isinstance(measures, (tuple, list)):
        measures = {m: get_measure(m) for m in measures}
    elif isinstance(measures, str):
        measures = {measures: get_measure(measures)}

    all_measure_results = defaultdict(lambda: [])
    for net in networks:
        for name, measure_f in measures.items():
            res = entropy_dt(net, max_dt=max_dt, entropy_game=measure_f)
            all_measure_results[name].append(res)
    return all_measure_results


def apply_process(networks, process, dt=100, n_sim=16, n_jobs=1):
    all_process_results = []
    for net in networks:
        res = process.simulate_dt(net, dt=dt, n_sim=n_sim, n_jobs=n_jobs)
        all_process_results.append(res)
    return all_process_results
