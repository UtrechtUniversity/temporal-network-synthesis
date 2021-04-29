from numba.core.decorators import njit
import numpy as np
from synet.measures.base import BasePaintEntropy


class PaintEntropy(BasePaintEntropy):
    """Boolean version of the paint game."""
    name = "paint"

    def measure_entropy(self, net, start, end):
        return paint_entropy(net, start, end)


def paint_entropy(net, start=0, end=None, numba=True):
    A = net.A
    if end is None:
        end = net.n_events

    entropy = np.full(end-start+1, -1, dtype=float)
    n_exit = np.array(A.sum(axis=1)).flatten()
    visited = np.full(end-start, False, dtype=np.bool)

    if numba:
        return _numba_paint_entropy(A.indptr, A.data, A.indices, entropy,
                                    visited, n_exit, start, end)
    return _python_paint_entropy(A, entropy, visited, n_exit, start, end)


@njit
def _numba_paint_entropy(A_indptr, A_data, A_indices, entropy, visited, n_exit,
                         start, end):
    "Numba version of boolean paint game (fast)."
    n_current_agents = n_exit[start]
    entropy[0] = 0
    entropy[1] = np.log(n_current_agents)
    visited[0] = True
    for dst_event in range(start+1, end):
        for src_pointer in range(A_indptr[dst_event], A_indptr[dst_event+1]):
            src_event = A_indices[src_pointer]
            n_agent = A_data[src_pointer]
            n_exit[src_event] -= n_agent
            if (src_event-start) >= 0 and visited[src_event-start]:
                n_current_agents -= n_agent
                visited[dst_event-start] = True
        if visited[dst_event-start]:
            n_current_agents += n_exit[dst_event]
        entropy[dst_event-start+1] = np.log(n_current_agents)
    return entropy


def _python_paint_entropy(A, entropy, visited, n_exit, start, end):
    "Python version of boolean paint game (slow)."
    n_current_agents = n_exit[start]
    entropy[0] = 0
    if start == end:
        return
    entropy[1] = np.log(n_current_agents)
    visited[0] = True
    for dst_event in range(start+1, end):
        for src_pointer in range(A.indptr[dst_event], A.indptr[dst_event+1]):
            src_event = A.indices[src_pointer]
            n_agent = A.data[src_pointer]
            n_exit[src_event] -= n_agent
            if (src_event-start) >= 0 and visited[src_event-start]:
                n_current_agents -= n_agent
                visited[dst_event-start] = True
        if visited[dst_event-start]:
            n_current_agents += n_exit[dst_event]

        entropy[dst_event-start] = np.log(n_current_agents)
    return entropy
