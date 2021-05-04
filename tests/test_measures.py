import pytest
import numpy as np
from numpy import log

from synet.config import measures
from synet.networks.random import random_network
from synet.networks.user import UserNetwork
from synet.measures.paint import PaintEntropy
from synet.measures.paths import PathEntropy
from synet.measures.mixing import MixingEntropy

n_events = 50


def create_user_net():
    participants = np.array([
        [1, 2, 3],
        [1, 4, 5],
        [3, 5, 6],
        [1, 2, 4],
    ])
    n_agents = 8
    return UserNetwork.from_participants(participants, n_agents)


@pytest.mark.parametrize(
    "measure_class", list(measures.values())
)
def test_basic_measures(measure_class):
    measure = measure_class()
    assert len(measure.name) > 0
    net = random_network(n_events=n_events)
    net2 = random_network(n_events=n_events)
    networks = [net, net2]
    entropy_dt = measure.entropy_dt(net, max_dt=50)
    for dt in [0, 1, 2, 6, 31]:
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
        assert np.isclose(np.sum(entropy_t)/(n_events-dt+1), entropy_dt[dt])
        for i in range(2):
            assert np.allclose(entropy_t_mono[i], entropy_t_parallel[i])
        assert len(entropy_t) == n_events
        if dt == 0:
            assert np.all(entropy_t == 0)
        elif dt == 1:
            assert not np.all(entropy_t == 0)
        else:
            assert np.all(entropy_t[:dt//2] == 0)
            assert np.all(entropy_t[dt//2:-dt//2] > 0)
            if dt > 2:
                assert np.all(entropy_t[-dt//2+1:] == 0)

    entropy_dt2 = measure.entropy_dt(net2, max_dt=50)
    entropy_dt_parallel = measure.entropy_dt(networks, max_dt=50, n_jobs=2)
    entropy_dt_mono = measure.entropy_dt(networks, max_dt=50, n_jobs=1)
    assert np.allclose(entropy_dt, entropy_dt_parallel[0])
    assert np.allclose(entropy_dt2, entropy_dt_parallel[1])
    assert np.allclose(entropy_dt_mono, entropy_dt_parallel)


def test_boolean_entropy():
    net = create_user_net()
    measure = PaintEntropy()
    entropy_t_dt1 = measure.entropy_t(net, 1, numba=False)
    entropy_t_dt2 = measure.entropy_t(net, 2, numba=False)
    entropy_t_dt3 = measure.entropy_t(net, 3, numba=False)
    entropy_t_dt4 = measure.entropy_t(net, 4, numba=False)

    assert np.allclose(entropy_t_dt1, 3*log(3))
    assert np.allclose(
        entropy_t_dt2,
        [0, 3*log(5)+2*log(3), 3*log(5)+2*log(3), 6*log(3)])
    assert np.allclose(
        entropy_t_dt3,
        [0, 3*log(6)+2*log(5)+log(3), 3*log(6)+3*log(3), 0])
    assert np.allclose(entropy_t_dt4, [0, 0, 5*log(6)+log(3), 0])


def comp_entropy(*args):
    total_sum = np.sum([a[0]*a[1] for a in args])
    return -np.sum([a[0]*a[1]/total_sum*log(a[1]/total_sum) for a in args])


def test_path_entropy():
    net = create_user_net()
    measure = PathEntropy()
    entropy_t_dt1 = measure.entropy_t(net, 1, numba=False)
    entropy_t_dt2 = measure.entropy_t(net, 2, numba=False)
    entropy_t_dt3 = measure.entropy_t(net, 3, numba=False)
    entropy_t_dt4 = measure.entropy_t(net, 4, numba=False)

    manual_dt2 = [0, 3*log(5)+2*log(3), 3*log(5)+2*log(3), 6*log(3)]
    manual_dt3 = [
        0,
        3*comp_entropy((3, 1), (3, 2)) + 2*log(5) + log(3),
        3*comp_entropy((3, 1), (3, 2)) + 3*log(3),
        0
    ]
    manual_dt4 = [
        0,
        0,
        (3*comp_entropy((3, 2), (3, 3)) +
         2*comp_entropy((3, 1), (3, 2)) + log(3)),
        0
    ]

    assert np.allclose(entropy_t_dt1, 3*log(3))
    assert np.allclose(entropy_t_dt2, manual_dt2)
    assert np.allclose(entropy_t_dt3, manual_dt3)
    assert np.allclose(entropy_t_dt4, manual_dt4)


def test_mixing_entropy():
    net = create_user_net()
    measure = MixingEntropy()
    entropy_t_dt1 = measure.entropy_t(net, 1, numba=False)
    entropy_t_dt2 = measure.entropy_t(net, 2, numba=False)
    entropy_t_dt3 = measure.entropy_t(net, 3, numba=False)
    entropy_t_dt4 = measure.entropy_t(net, 4, numba=False)

    manual_dt2 = [0, 3*comp_entropy((2, 1/3), (3, 1/9)) + 2*log(3),
                  3*comp_entropy((2, 1/3), (3, 1/9)) + 2*log(3), 6*log(3)]
    manual_dt3 = [
        0,
        (3*comp_entropy((1, 1/3), (2, 1/9), (3, 4/27))
         + 2*comp_entropy((2, 1/3), (3, 1/9)) + log(3)),
        3*comp_entropy((3, 2/9), (3, 1/9)) + 3*log(3),
        0
    ]
    manual_dt4 = [
        0,
        0,
        (3*comp_entropy((3, 5/27), (3, 4/27)) +
         2*comp_entropy((3, 1/9), (3, 2/9)) + log(3)),
        0
    ]

    assert np.allclose(entropy_t_dt1, 3*log(3))
    assert np.allclose(entropy_t_dt2, manual_dt2)
    assert np.allclose(entropy_t_dt3, manual_dt3)
    assert np.allclose(entropy_t_dt4, manual_dt4)
