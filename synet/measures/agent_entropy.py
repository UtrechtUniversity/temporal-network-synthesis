import numpy as np
from numba import njit
from synet.measures.base import BaseMeasure


class AgentEntropy(BaseMeasure):
    """Entropy measure that doesn't take topology into account.

    It measures how agents are reached between two time steps.
    The computation is as follows:
    p = N_v/np.sum(N_v), with N_v the number of visits of agents.
    entropy = np.sum(p*log(p)).
    """
    name = "agent"

    def measure_entropy(self, net, start, end):
        return agent_entropy(net, start, end)


def agent_entropy(net, start=1, end=None, numba=True):
    if end is None:
        end = net.n_events

    agent_count = np.zeros(net.n_agents, dtype=int)
    entropy = np.zeros(end-start)

    if numba:
        return _numba_agent_entropy(net.participants, agent_count, entropy,
                                    start, end)
    return _python_agent_entropy(net, agent_count, entropy, start, end)


@njit
def _numba_agent_entropy(participants, agent_count, entropy, start, end):
    """Numba enabled computation of agent entropy."""
    for i_event in range(start, end):
        agent_count[participants[i_event]] += 1
        n_visit = agent_count[(agent_count > 0)]
        p = n_visit/np.sum(n_visit)
        entropy[i_event-start] = -np.sum(p*np.log(p))
    return entropy


def _python_agent_entropy(net, agent_count, entropy, start, end):
    """Perform computation of agent entropy in plain python."""
    for i_event in range(start, end):
        agent_count[net.participants[i_event]] += 1
        p = agent_count[(agent_count > 0)]
        p = p/np.sum(p)
        entropy[i_event-start] = -np.sum(p*np.log(p))
    return entropy
