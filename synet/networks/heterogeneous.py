import numpy as np

from synet.networks.base import BaseNetwork
from synet.networks.utils import get_event_times


class HeterogeneousNetwork(BaseNetwork):
    def __init__(self, n_agents=20, event_size=5, n_events=100, time_span=None,
                 heterogeniety=1.0):
        """Heterogeneous network

        In the heterogeneous network, agent have different intrinsic rates.
        The rates are distributed as (1-h) + h*pareto(2), and then normalized
        to 1.

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
        heterogeniety: double
            Value between 0 and 1 that governs how different the event rates
            are for agents. Higher values indicate rates that are more diverse.
        """
        if time_span is None:
            time_span = n_events/n_agents

        self.n_agents = n_agents
        self.time_span = time_span

        self.heterogeniety = heterogeniety

        self.agent_rates = heterogeniety*np.random.pareto(2, self.n_agents)
        self.agent_rates += (1-heterogeniety)
        self.agent_rates /= np.mean(self.agent_rates)

        agent_prob = self.agent_rates/np.sum(self.agent_rates)
        participants = np.zeros((n_events, event_size), dtype=int)
        event_sources = np.zeros(n_events, dtype=int)
        for i in range(0, n_events):
            participants[i, :] = np.random.choice(
                n_agents, size=event_size, replace=False,
                p=agent_prob)

        self.event_sources = event_sources
        self.event_types = {0: self.name}
        self.event_times = get_event_times(n_events, time_span)
        self.participants = participants

    @property
    def name(self):
        return (f"hetero_a{self.n_agents}s{self.event_size}e{self.n_events}"
                f"t{self.time_span}h{self.heterogeniety:.3f}")
