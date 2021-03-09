from numba.core.decorators import njit
import numpy as np

from synet.propagators.utils import add_value


def event_paint_game(A):
#     assert isinstance(A, csc_matrix)
    paint_matrix = np.full(A.shape, -1, dtype=float)
    np.fill_diagonal(paint_matrix, 0)
    return numba_event_paint_game(paint_matrix, A.indptr, A.indices)
    np.fill_diagonal(paint_matrix, 0)
    for i_col in range(len(A.indptr)-1):
        n_entry = A.indptr[i_col+1] - A.indptr[i_col]
        for i_row in A.indices[A.indptr[i_col]:A.indptr[i_col+1]]:
            for i_event in range(i_col):
                paint_matrix[i_col, i_event] = add_value(
                    paint_matrix[i_col, i_event], paint_matrix[i_row, i_event])
#             paint_matrix[i_col] += paint_matrix[i_row]

    return paint_matrix


@njit
def numba_event_paint_game(paint_matrix, indptr, indices):
    for i_col in range(len(indptr)-1):
        n_entry = indptr[i_col+1] - indptr[i_col]
        for i_row in indices[indptr[i_col]:indptr[i_col+1]]:
            for i_event in range(i_col):
                paint_matrix[i_col, i_event] = add_value(
                    paint_matrix[i_col, i_event], paint_matrix[i_row, i_event])
    return paint_matrix


def mem_event_paint_game(A):
    AT = A.T
    n_events = A.shape[0]
    active_sources = np.full(n_events, None, dtype=object)
    active_sources[0] = np.full(n_events, False, dtype=np.bool)
    paint_results = np.zeros(n_events, dtype=int)

    for dst_event in range(0, len(A.indptr)-1):
        connected = np.full(n_events, False, dtype=np.bool)
        connected[dst_event] = True
        for src_event in A.indices[A.indptr[dst_event]:A.indptr[dst_event+1]]:
            connected = np.logical_or(connected, active_sources[src_event])
            if A.indices[AT.indptr[src_event+1]-1] == dst_event:
                active_sources[src_event] = None
        active_sources[dst_event] = connected
        paint_results[1:dst_event] += np.flip(connected[1:dst_event]).astype(int)

    normalization = np.arange(A.shape[0]-2, 0, -1)
    return paint_results[1:-1]/normalization


def paint_entropy(A, start=0, end=None, numba=True):
    n_events = A.shape[0]
    if end is None:
        end = n_events

    entropy = np.full(end-start, -1, dtype=float)
    n_exit = np.array(A.sum(axis=1)).flatten()
    visited = np.full(end-start, False, dtype=np.bool)

    if numba:
        return numba_paint_entropy(A.indptr, A.data, A.indices, entropy,
                                   visited, n_exit, start, end)
    return python_paint_entropy(A, entropy, visited, n_exit, start, end)
#     n_current_agents = n_exit[start]
#     entropy[0] = np.log(n_current_agents)
#     visited[0] = True
#     for dst_event in range(start+1, end):
#         for src_pointer in range(A.indptr[dst_event], A.indptr[dst_event+1]):
#             src_event = A.indices[src_pointer]
#             n_agent = A.data[src_pointer]
#             n_exit[src_event] -= n_agent
#             if (src_event-start) >= 0 and visited[src_event-start]:
#                 n_current_agents -= n_agent
#                 visited[dst_event-start] = True
#         if visited[dst_event-start]:
#             n_current_agents += n_exit[dst_event]
#         entropy[dst_event-start] = np.log(n_current_agents)
#     return entropy


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
