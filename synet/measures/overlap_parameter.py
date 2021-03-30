import numpy as np
from numba import njit


def overlap_parameter(net, start=1, end=None, numba=True):
    if end is None:
        end = net.n_events

    n_agents = net.n_agents
    last_events = np.full(n_agents, -1, dtype=int)
    overlap = np.zeros(end-start, dtype=float)
    if numba:
        return numba_overlap(
            net.participants, last_events, overlap, start, end)
    return python_overlap(net, last_events, overlap, start, end)

#     for i_event in range(start, end):
#         agents = net.participants[i_event]
#         previous_events = last_events[agents]
#         previous_events = previous_events[previous_events != -1]
#         n_unique = np.unique(previous_events)
#         if len(n_unique):
#             overlap[i_event-start] = len(n_unique)/len(previous_events)
#         else:
#             overlap[i_event-start] = 1
#         last_events[agents] = i_event
#     return overlap


@njit
def numba_overlap(participants, last_events, overlap, start, end):
    for i_event in range(start, end):
        agents = participants[i_event]
        previous_events = last_events[agents]
        previous_events = previous_events[previous_events != -1]
        n_total = previous_events.size
        n_unique = 0
        while len(previous_events) > 0:
            previous_events = previous_events[previous_events != previous_events[0]]
            n_unique += 1
#         n_unique = np.unique(previous_events)
        if n_unique:
            overlap[i_event-start] = n_unique/n_total
        else:
            overlap[i_event-start] = 1
        last_events[agents] = i_event
    return overlap


def python_overlap(net, last_events, overlap, start, end):
    for i_event in range(start, end):
        agents = net.participants[i_event]
        previous_events = last_events[agents]
        previous_events = previous_events[previous_events != -1]
        n_unique = np.unique(previous_events)
        if len(n_unique):
            overlap[i_event-start] = len(n_unique)/len(previous_events)
        else:
            overlap[i_event-start] = 1
        last_events[agents] = i_event
    return overlap
