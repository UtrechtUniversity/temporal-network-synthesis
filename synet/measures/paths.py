import numpy as np
from numba.core.decorators import njit

from synet.measures.utils import add_value, sub_value
from synet.measures.base import BaseMeasure


class PathEntropy(BaseMeasure):
    """Counting path paint game."""
    name = "path"

    def measure_entropy(self, net, start, end, **kwargs):
        return path_entropy(net, start, end, **kwargs)


@njit
def entropy_compute(log_sum_n_log_n, log_sum_n):
    if log_sum_n < -0.5:
        return -1
    if log_sum_n_log_n < -0.5:
        return log_sum_n
    else:
        return -np.exp(log_sum_n_log_n - log_sum_n) + log_sum_n


def path_entropy(net, start, end, numba=True):
    A = net.A
    n_events = A.shape[0]

    log_n_path = np.full(n_events, -1, dtype=float)
    n_exit = np.array(np.sum(A, axis=1)).flatten()
    entropy = np.full(end-start+1, -1, dtype=float)
    if numba:
        return _numba_path_entropy(
            A.indptr, A.data, A.indices, entropy, log_n_path,
            n_exit, start, end)
    return python_path_entropy(A, entropy, log_n_path, n_exit, start, end)


@njit
def _numba_path_entropy(A_indptr, A_data, A_indices, entropy, log_n_path,
                        n_exit, start, end):
    "Numba computation of the path paint game (fast)."
    log_sum_n_log_n = -1
    log_sum_n = np.log(n_exit[start])

    entropy[0] = 0
    if start == end:
        return entropy

    log_n_path[start] = 0
    entropy[1] = entropy_compute(log_sum_n_log_n, log_sum_n)
    for dst_event in range(start+1, end):
        for src_pointer in range(A_indptr[dst_event], A_indptr[dst_event+1]):
            src_event = A_indices[src_pointer]
            n_agent = A_data[src_pointer]
            n_exit[src_event] -= n_agent

            if log_n_path[src_event] > -1:
                log_n_path[dst_event] = add_value(
                    log_n_path[dst_event],
                    np.log(n_agent)+log_n_path[src_event])
                log_sum_n = sub_value(
                    log_sum_n, np.log(n_agent)+log_n_path[src_event])
                if log_n_path[src_event] >= np.log(2)-1e-6:
                    log_sum_n_log_n = sub_value(
                        log_sum_n_log_n,
                        np.log(n_agent) + log_n_path[src_event]
                        + np.log(log_n_path[src_event]))
        if log_n_path[dst_event] > -1 and n_exit[dst_event]:
            log_sum_n = add_value(
                log_sum_n, np.log(n_exit[dst_event]) + log_n_path[dst_event])
            if log_n_path[dst_event] >= np.log(2)-1e-6:
                log_sum_n_log_n = add_value(
                    log_sum_n_log_n, np.log(n_exit[dst_event])
                    + log_n_path[dst_event] + np.log(log_n_path[dst_event]))
        entropy[dst_event-start+1] = entropy_compute(log_sum_n_log_n,
                                                     log_sum_n)
    return entropy


def python_path_entropy(A, entropy, log_n_path, n_exit, start, end):
    "Python path computation of the path paint game (slow)."
    log_sum_n_log_n = -1
    log_sum_n = np.log(n_exit[start])

    entropy[0] = 0
    if start == end:
        return entropy
    log_n_path[start] = 0
    entropy[1] = entropy_compute(log_sum_n_log_n, log_sum_n)
    for dst_event in range(start+1, end):
        for src_pointer in range(A.indptr[dst_event], A.indptr[dst_event+1]):
            src_event = A.indices[src_pointer]
            n_agent = A.data[src_pointer]
            n_exit[src_event] -= n_agent

            if log_n_path[src_event] > -1:
                log_n_path[dst_event] = add_value(
                    log_n_path[dst_event],
                    np.log(n_agent)+log_n_path[src_event])
                log_sum_n = sub_value(
                    log_sum_n, np.log(n_agent)+log_n_path[src_event])
                if log_n_path[src_event] >= np.log(2)-1e-6:
                    log_sum_n_log_n = sub_value(
                        log_sum_n_log_n,
                        np.log(n_agent) + log_n_path[src_event]
                        + np.log(log_n_path[src_event]))
        if log_n_path[dst_event] > -1 and n_exit[dst_event]:
            log_sum_n = add_value(
                log_sum_n, np.log(n_exit[dst_event]) + log_n_path[dst_event])
            if log_n_path[dst_event] >= np.log(2)-1e-6:
                log_sum_n_log_n = add_value(
                    log_sum_n_log_n,
                    np.log(n_exit[dst_event]) + log_n_path[dst_event]
                    + np.log(log_n_path[dst_event]))
        entropy[dst_event-start+1] = entropy_compute(log_sum_n_log_n,
                                                     log_sum_n)
    return entropy
