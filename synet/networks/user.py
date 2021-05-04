import numpy as np

from synet.networks.base import BaseNetwork


class UserNetwork(BaseNetwork):
    name = "user"

    def __init__(self, participants, n_agents, time_span=None,
                 event_times=None, event_types=None, event_sources=None,
                 agent_rates=None):
        n_events = participants.shape[0]
        if time_span is None:
            time_span = n_events/n_agents
        if event_times is None:
            event_times = time_span*np.arange(n_events)/n_events
        if event_sources is None:
            event_sources = np.zeros(n_events)
        if event_types is None:
            event_types = {0: self.name}
        if agent_rates is None:
            agent_rates = np.ones(n_agents)

        self.participants = participants
        self.n_agents = n_agents
        self.time_span = time_span
        self.event_times = event_times
        self.event_types = event_types
        self.event_sources = event_sources
        self.agent_rates = agent_rates

    @classmethod
    def from_participants(cls, participants, n_agents, **kwargs):
        return cls(participants, n_agents, **kwargs)
