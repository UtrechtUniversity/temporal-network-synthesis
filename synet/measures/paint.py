from numba.core.decorators import njit
import numpy as np
from synet.measures.base import BaseMeasure


class PaintEntropy(BaseMeasure):
    """Boolean version of the paint game."""
    name = "paint"

    def measure_entropy(self, net, start, end, **kwargs):
        return paint_entropy(net, start, end, **kwargs)


def paint_entropy(net, start, end, numba=True):
    A = net.A

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
    n_current_agents = n_exit[start+1]
    entropy[0] = 0
    if start == end:
        return entropy
    entropy[1] = np.log(n_current_agents)
    visited[0] = True
    for dst_event in range(start+2, end+1):
        for src_pointer in range(A_indptr[dst_event], A_indptr[dst_event+1]):
            src_event = A_indices[src_pointer]
            n_agent = A_data[src_pointer]
            n_exit[src_event] -= n_agent
            if src_event >= start+1 >= 0 and visited[src_event-start-1]:
                n_current_agents -= n_agent
                visited[dst_event-start-1] = True
        if visited[dst_event-start-1]:
            n_current_agents += n_exit[dst_event]
        entropy[dst_event-start] = np.log(n_current_agents)
    return entropy


def _python_paint_entropy(A, entropy, visited, n_exit, start, end):
    "Python version of boolean paint game (slow)."
    n_current_agents = n_exit[start+1]
    entropy[0] = 0
    if start == end:
        return entropy
    entropy[1] = np.log(n_current_agents)
    visited[0] = True
    for dst_event in range(start+2, end+1):
        for src_pointer in range(A.indptr[dst_event], A.indptr[dst_event+1]):
            src_event = A.indices[src_pointer]
            n_agent = A.data[src_pointer]
            n_exit[src_event] -= n_agent
            if src_event >= start+1 and visited[src_event-start-1]:
                n_current_agents -= n_agent
                visited[dst_event-start-1] = True
        if visited[dst_event-start-1]:
            n_current_agents += n_exit[dst_event]

        entropy[dst_event-start] = np.log(n_current_agents)
    return entropy
