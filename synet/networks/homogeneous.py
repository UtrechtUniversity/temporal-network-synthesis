import numpy as np

from synet.networks.base import BaseNetwork
from synet.networks.utils import get_event_times


class HomogeneousNetwork(BaseNetwork):
    def __init__(self, n_agents=20, event_size=5, n_events=100, time_span=None):
        if time_span is None:
            time_span = n_events/n_agents

        self.n_agents = n_agents
        self.time_span = time_span
        self.n_events = n_events
        self.event_size = event_size
        self.agent_rates = np.ones(n_agents)

        participants = np.zeros((n_events, event_size), dtype=int)
        event_sources = np.zeros(n_events, dtype=int)
        for i in range(0, n_events):
            participants[i, :] = np.random.choice(
                n_agents, size=event_size, replace=False)

        self.event_sources = event_sources
        self.event_types = {0: self.name}
        self.event_times = get_event_times(n_events, time_span)
        self.participants = participants

    @property
    def name(self):
        return (f"homo_a{self.n_agents}s{self.event_size}e{self.n_events}"
                f"t{self.time_span}")
