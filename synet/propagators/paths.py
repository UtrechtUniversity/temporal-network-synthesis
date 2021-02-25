import numpy as np

from synet import add_value, sub_value


def entropy_game(A, start=0, end=None):
    n_events = A.shape[0]
    if end is None:
        end = n_events
    n_path = np.zeros(n_events, dtype=float)
    n_exit = np.array(np.sum(A, axis=1)).flatten()
    entropy = np.zeros(end-start, dtype=float)
    sum_n_log_n = 0
    sum_n = n_exit[start]
    n_path[start] = 1

    entropy[0] = -sum_n_log_n/sum_n + np.log(sum_n)
    for dst_event in range(start+1, end):
        for src_pointer in range(A.indptr[dst_event], A.indptr[dst_event+1]):
            src_event = A.indices[src_pointer]
            n_agent = A.data[src_pointer]
            n_exit[src_event] -= n_agent
            n_path[dst_event] += n_agent*n_path[src_event]
            if n_path[src_event] > 0:
                sum_n -= n_agent*n_path[src_event]
                sum_n_log_n -= n_agent*n_path[src_event]*np.log(n_path[src_event])
        if n_path[dst_event] > 0:
            sum_n += n_exit[dst_event]*n_path[dst_event]
            sum_n_log_n += n_exit[dst_event]*n_path[dst_event]*np.log(n_path[dst_event])
        if sum_n > 0.9:
            entropy[dst_event-start] = -sum_n_log_n/sum_n + np.log(sum_n)
    return entropy


def entropy_compute(log_sum_n_log_n, log_sum_n):
    if log_sum_n < -0.5:
        return -1
    if log_sum_n_log_n < -0.5:
        return log_sum_n
    else:
        return -np.exp(log_sum_n_log_n - log_sum_n) + log_sum_n


def path_entropy(A, start=0, end=None):
    n_events = A.shape[0]
    if end is None:
        end = n_events
    log_n_path = np.full(n_events, -1, dtype=float)
    n_exit = np.array(np.sum(A, axis=1)).flatten()
    entropy = np.full(end-start, -1, dtype=float)
    log_sum_n_log_n = -1
    log_sum_n = np.log(n_exit[start])
    log_n_path[start] = 0

    entropy[0] = entropy_compute(log_sum_n_log_n, log_sum_n)
    for dst_event in range(start+1, end):
        for src_pointer in range(A.indptr[dst_event], A.indptr[dst_event+1]):
            src_event = A.indices[src_pointer]
            n_agent = A.data[src_pointer]
            n_exit[src_event] -= n_agent

            if log_n_path[src_event] > -1:
                log_n_path[dst_event] = add_value(
                    log_n_path[dst_event], np.log(n_agent)+log_n_path[src_event])
                log_sum_n = sub_value(
                    log_sum_n, np.log(n_agent)+log_n_path[src_event])
                if log_n_path[src_event] >= np.log(2)-1e-6:
                    log_sum_n_log_n = sub_value(
                        log_sum_n_log_n, np.log(n_agent) + log_n_path[src_event]
                        + np.log(log_n_path[src_event]))
        if log_n_path[dst_event] > -1 and n_exit[dst_event]:
            log_sum_n = add_value(log_sum_n, np.log(n_exit[dst_event]) + log_n_path[dst_event])
            if log_n_path[dst_event] >= np.log(2)-1e-6:
                log_sum_n_log_n = add_value(log_sum_n_log_n, np.log(n_exit[dst_event])
                                            + log_n_path[dst_event] + np.log(log_n_path[dst_event]))
        entropy[dst_event-start] = entropy_compute(log_sum_n_log_n, log_sum_n)
    return entropy


