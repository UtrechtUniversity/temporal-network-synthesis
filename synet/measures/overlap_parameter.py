import numpy as np
from numba import njit
from synet.measures.base import BaseMeasure


class OverlapParameter(BaseMeasure):
    """Overlap measure.

    Measures how much overlap there is between events some time apart.
    At t+dt, count the number of events of participants with a previous
    event within the interval [t, t+dt). Then measure the number of unique
    events in this set divided by the total number of previous events.
    """
    name = "overlap"

    def measure_entropy(self, net, start, end, **kwargs):
        return overlap_parameter(net, start, end, **kwargs)


def overlap_parameter(net, start=1, end=None, numba=True):
    if end is None:
        end = net.n_events

    n_agents = net.n_agents
    last_events = np.full(n_agents, -1, dtype=int)
    overlap = np.zeros(end-start+1, dtype=float)
    if numba:
        return _numba_overlap(
            net.participants, last_events, overlap, start, end)
    return _python_overlap(net, last_events, overlap, start, end)


@njit
def _numba_overlap(participants, last_events, overlap, start, end):
    "Numba overlap computation (fast)."
    for i_event in range(start, end):
        agents = participants[i_event]
        previous_events = last_events[agents]
        previous_events = previous_events[previous_events != -1]
        n_total = previous_events.size

        # Manual implementation of np.unique (not available under numpy).
        n_unique = 0
        while len(previous_events) > 0:
            previous_events = previous_events[
                previous_events != previous_events[0]]
            n_unique += 1
        if n_unique:
            overlap[i_event-start+1] = n_unique/n_total
        else:
            overlap[i_event-start+1] = 1
        last_events[agents] = i_event
    return overlap


def _python_overlap(net, last_events, overlap, start, end):
    "Python overlap computation (slow)"
    for i_event in range(start, end):
        agents = net.participants[i_event]
        previous_events = last_events[agents]
        previous_events = previous_events[previous_events != -1]
        n_unique = np.unique(previous_events)
        if len(n_unique):
            overlap[i_event-start+1] = len(n_unique)/len(previous_events)
        else:
            overlap[i_event-start+1] = 1
        last_events[agents] = i_event
    return overlap
