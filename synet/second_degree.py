import numpy as np


def overlap_parameter(participants, start, end, n_agents=-1):
    if n_agents == -1:
        n_agents = np.max(participants) + 1

    last_events = np.full(n_agents, -1, dtype=int)
    overlap = np.zeros(end-start, dtype=float)

    for i_event in range(start, end):
        agents = participants[i_event]
        previous_events = last_events[agents]
        previous_events = previous_events[previous_events != -1]
        n_unique = np.unique(previous_events)
        if len(n_unique):
            overlap[i_event-start] = len(n_unique)/len(previous_events)
        else:
            overlap[i_event-start] = 1
        last_events[agents] = i_event
    return overlap
