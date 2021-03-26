import numpy as np


def agent_entropy(net, start=1, end=None):
    if end is None:
        end = net.n_events

    agent_count = np.zeros(net.n_agents, dtype=int)

    entropy = np.zeros(end-start)
    for i_event in range(start, end):
        agent_count[net.participants[i_event]] += 1
        p = agent_count[(agent_count > 0)]
        p = p/np.sum(p)
        entropy[i_event-start] = -np.sum(p*np.log(p))

    return entropy
