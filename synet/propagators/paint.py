from numba.core.decorators import njit
import numpy as np

from synet.generators.utils import add_value


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
