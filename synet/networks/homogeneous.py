import numpy as np

from synet.networks.base import BaseNetwork
from synet.networks.utils import get_event_times


class HomogeneousNetwork(BaseNetwork):
    def __init__(self, n_agents=20, event_size=5, n_events=100,
                 time_span=None):
        """Homogeneous network

        In the homogeneous network, the intrinsic rates for each agent
        is the same.

        Arguments
        ---------
        n_agents: int
            Number of agents in the network.
        event_size: int
            Size of each event.
        n_events: int
            Number of events in the network.
        time_span: double
            Time span of the network, relevant for merging.
        """
        if time_span is None:
            time_span = n_events/n_agents

        self.n_agents = n_agents
        self.time_span = time_span
        self.agent_rates = np.ones(n_agents)

        participants = np.zeros((n_events, event_size), dtype=int)
        event_sources = np.zeros(n_events, dtype=int)
        for i in range(0, n_events):
            participants[i, :] = np.random.choice(
                n_agents, size=event_size, replace=False)

        self.event_sources = event_sources
        self.event_times = get_event_times(n_events, time_span)
        self.participants = participants
        self.event_types = {0: self.name}

    @property
    def name(self):
        return (f"homo_a{self.n_agents}s{self.event_size}e{self.n_events}"
                f"t{self.time_span}")
