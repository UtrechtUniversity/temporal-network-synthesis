import numpy as np


def double_path_entropy(A, start, end):
    n_events = A.shape[0]
    AT = A.T

    n_path = np.zeros(n_events)
    n_path[start] = 1

    for dst_event in range(start+1, end+1):
        for src_pointer in range(A.indptr[dst_event], A.indptr[dst_event+1]):
            src_event = A.indices[src_pointer]
            n_agent = A.data[src_pointer]
            n_path[dst_event] += n_agent*n_path[src_event]
#             if src_event > start:
#                 n_entry[dst_event] -= n_agent

    sum_n = 0
    sum_n_log_n = 0
    for dst_event in range(end, start-1, -1):
        for src_pointer in range(AT.indptr[dst_event], AT.indptr[dst_event+1]):
            src_event = AT.indices[src_pointer]
            n_agent = AT.data[src_pointer]
            if src_event < start and n_path[dst_event] > 0:
                sum_n += n_agent*n_path[dst_event]
                sum_n_log_n += n_agent*n_path[dst_event]*np.log(n_path[dst_event])
            n_path[src_event] += n_agent*n_path[src_event]
    return -sum_n_log_n/sum_n + np.log(sum_n)
