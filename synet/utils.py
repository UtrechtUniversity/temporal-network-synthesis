from collections import defaultdict

import numpy as np

from synet.analysis import entropy_dt, fixed_entropy_dt


def get_measure(measure_name, **kwargs):
    from synet.config import measures
    if isinstance(measure_name, (tuple, list, np.ndarray)):
        return [measures[n](**kwargs) for n in measure_name]
    return measures[measure_name](**kwargs)


def apply_measures(networks, measures=None, max_dt=100, n_jobs=1):
    if measures is None:
        from synet.config import measures as measure_classes
        measures = {key: value() for key, value in measure_classes.items()}

    if isinstance(measures, (tuple, list)):
        measures = {m: get_measure(m) for m in measures}
    elif isinstance(measures, str):
        measures = {measures: get_measure(measures)}

    all_measure_results = defaultdict(lambda: [])
    for name, measure in measures.items():
        all_measure_results[name] = measure.entropy_dt(
            networks=networks, max_dt=max_dt, n_jobs=n_jobs)
    return dict(all_measure_results)


def apply_process(networks, process, dt=100, n_sim=16, n_jobs=1):
    all_process_results = []
#     for net in networks:
#         res = process.simulate_dt(net, dt=dt, n_sim=n_sim, n_jobs=n_jobs)
#         all_process_results.append(res)
    return process.simulate_dt(networks, dt, n_sim=n_sim, n_jobs=n_jobs)
#     return all_process_results
