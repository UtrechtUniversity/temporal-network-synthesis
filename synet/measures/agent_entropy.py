import numpy as np
from numba import njit


def agent_entropy(net, start=1, end=None, numba=True):
    if end is None:
        end = net.n_events

    agent_count = np.zeros(net.n_agents, dtype=int)
    entropy = np.zeros(end-start)

    if numba:
        return numba_agent_entropy(net.participants, agent_count, entropy,
                                   start, end)
    return python_agent_entropy(net, agent_count, entropy, start, end)


@njit
def numba_agent_entropy(participants, agent_count, entropy, start, end):
    for i_event in range(start, end):
        agent_count[participants[i_event]] += 1
        p = agent_count[(agent_count > 0)]
        p = p/np.sum(p)
        entropy[i_event-start] = -np.sum(p*np.log(p))
    return entropy


def python_agent_entropy(net, agent_count, entropy, start, end):
    for i_event in range(start, end):
        agent_count[net.participants[i_event]] += 1
        p = agent_count[(agent_count > 0)]
        p = p/np.sum(p)
        entropy[i_event-start] = -np.sum(p*np.log(p))
    return entropy
