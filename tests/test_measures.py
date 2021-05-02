import pytest
import numpy as np

from synet.config import measures
from synet.networks.random import random_network

n_events = 50


@pytest.mark.parametrize(
    "measure_class", list(measures.values())
)
def test_basic_measures(measure_class):
    measure = measure_class()
    assert len(measure.name) > 0
    net = random_network(n_events=n_events)
    net2 = random_network(n_events=n_events)
    networks = [net, net2]
    for dt in [0, 1, 3, 6, 31]:
        entropy_t = measure.entropy_t(net, dt=dt, numba=False)
        entropy_t2 = measure.entropy_t(net2, dt=dt, numba=False)
        entropy_t_numba = measure.entropy_t(net, dt=dt, numba=True)
        entropy_t_parallel = measure.entropy_t(networks, dt=dt, numba=True,
                                               n_jobs=2)
        entropy_t_mono = measure.entropy_t(networks, dt=dt, numba=True,
                                           n_jobs=1)
        assert np.allclose(entropy_t, entropy_t_numba)
        assert np.allclose(entropy_t_parallel[0], entropy_t)
        assert np.allclose(entropy_t_parallel[1], entropy_t2)
        for i in range(2):
            assert np.allclose(entropy_t_mono[i], entropy_t_parallel[i])
        assert len(entropy_t) == n_events
        if dt == 0:
            assert np.all(entropy_t == 0)
        elif dt == 1:
            assert not np.all(entropy_t == 0)
        else:
            assert np.all(entropy_t[:dt//2] == 0)
            assert np.all(entropy_t[-dt//2+1:] == 0)
            assert np.all(entropy_t[dt//2:-dt//2] > 0)

    entropy_dt = measure.entropy_dt(net, max_dt=50)
    entropy_dt2 = measure.entropy_dt(net2, max_dt=50)
    entropy_dt_parallel = measure.entropy_dt(networks, max_dt=50, n_jobs=2)
    entropy_dt_mono = measure.entropy_dt(networks, max_dt=50, n_jobs=1)
    assert np.allclose(entropy_dt, entropy_dt_parallel[0])
    assert np.allclose(entropy_dt2, entropy_dt_parallel[1])
    assert np.allclose(entropy_dt_mono, entropy_dt_parallel)
