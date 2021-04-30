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
    for dt in [1, 3, 6, 31]:
        entropy_t = measure.entropy_t(net, dt=dt, numba=False)
        entropy_t_numba = measure.entropy_t(net, dt=dt, numba=True)
        assert np.allclose(entropy_t, entropy_t_numba)
        assert len(entropy_t) == n_events
        if dt > 1:
            assert np.all(entropy_t[:dt//2] == 0)
            assert np.all(entropy_t[-dt//2+1:] == 0)
            assert np.all(entropy_t[dt//2:-dt//2] > 0)
        else:
            assert not np.all(entropy_t == 0)
