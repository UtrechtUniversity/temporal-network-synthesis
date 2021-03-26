from numba.core.decorators import njit
import numpy as np


def paint_entropy(net, start=1, end=None, numba=True):
    A = net.A
    if end is None:
        end = net.n_events

    entropy = np.full(end-start, -1, dtype=float)
    n_exit = np.array(A.sum(axis=1)).flatten()
    visited = np.full(end-start, False, dtype=np.bool)

    if numba:
        return numba_paint_entropy(A.indptr, A.data, A.indices, entropy,
                                   visited, n_exit, start, end)
    return python_paint_entropy(A, entropy, visited, n_exit, start, end)


@njit
def numba_paint_entropy(A_indptr, A_data, A_indices, entropy, visited, n_exit,
                        start, end):
    n_current_agents = n_exit[start]
    entropy[0] = np.log(n_current_agents)
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
        entropy[dst_event-start] = np.log(n_current_agents)
    return entropy


def python_paint_entropy(A, entropy, visited, n_exit, start, end):
    n_current_agents = n_exit[start]
    entropy[0] = np.log(n_current_agents)
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
