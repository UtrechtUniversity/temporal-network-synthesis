import numpy as np

from synet.networks.heterogeneous import HeterogeneousNetwork
from synet.networks.homogeneous import HomogeneousNetwork
from synet.networks.random import random_network, random_two_split_network
from unittest.mock import patch
from scipy.sparse.csc import csc_matrix

n_agents = 50
n_events = 150
event_size = 3


def check_network(net, n_sources=1):
    assert net.event_size == event_size
    assert net.n_agents == n_agents
    assert net.n_events == n_events
    if n_sources is not None:
        assert len(np.unique(net.event_sources)) == n_sources
        assert len(net.event_types) == n_sources

    assert set(list(net.event_types)) >= set(np.unique(net.event_sources))
    assert np.all(net.event_times < net.time_span)
    assert len(net.event_times) == net.n_events
    assert net.participants.shape == (net.n_events, net.event_size)
    assert np.all(net.participants < net.n_agents)
    assert np.all(net.participants >= 0)
    assert len(net.event_sources) == net.n_events
    assert len(net.agent_rates) == net.n_agents
    check_adjacency(net)


def check_adjacency(net):
    assert isinstance(net.A, csc_matrix)
    assert net.A.shape == (n_events+2, n_events+2)
    last_events = np.zeros(net.n_agents, dtype=int)
    print(net.A[0, :].todense())
    for i_event, agents in enumerate(net.participants):
        for agent in agents:
            assert net.A[last_events[agent], i_event+1] >= 1
        last_events[agents] = i_event+1

    assert np.all(net.A[last_events, net.n_events+1].todense() >= 1)
    assert np.sum(net.A, axis=None) == net.n_events*net.event_size + n_agents


def test_hetero_network():
    net = HeterogeneousNetwork(n_agents=n_agents, n_events=n_events,
                               event_size=event_size, heterogeniety=0.0)
    check_network(net)
    assert np.allclose(net.agent_rates, net.agent_rates[0])

    net = HeterogeneousNetwork(n_agents=n_agents, n_events=n_events,
                               event_size=event_size, heterogeniety=1.0)
    check_network(net)


@patch("matplotlib.pyplot.show")
def test_homo_network(_):
    net = HomogeneousNetwork(n_agents=n_agents, n_events=n_events,
                             event_size=event_size)
    assert np.allclose(net.agent_rates, net.agent_rates[0])
    check_network(net)
    net.plot()
    net.plot_matrix()


def test_random_network():
    for _ in range(10):
        net = random_network(n_agents=n_agents, n_events=n_events,
                             n_community=2, event_size=event_size)
        check_network(net, n_sources=None)


def test_two_split_network():
    for n_inter_events in [0, 10, 100]:
        net = random_two_split_network(
            n_agents=n_agents, n_events=n_events,
            event_size=event_size, n_inter_events=n_inter_events)
        check_network(net, n_sources=None)
        if n_inter_events == 0:
            assert len(np.unique(net.event_sources)) == 2
        else:
            assert len(np.unique(net.event_sources)) == 3
