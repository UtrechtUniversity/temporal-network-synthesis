import numpy as np


def mixing_entropy(A, start=0, end=None):
    n_events = A.shape[0]
    if end is None:
        end = n_events

    entropy = np.full(end-start, -1, dtype=float)
    n_exit = np.array(A.sum(axis=1)).flatten()
    current_p_log_p = np.log(n_exit[start])
    paint_fraction = np.zeros(end-start)

    paint_fraction[0] = 1
    entropy[0] = current_p_log_p

    for dst_event in range(start+1, end):
        for src_pointer in range(A.indptr[dst_event], A.indptr[dst_event+1]):
            src_event = A.indices[src_pointer]
            n_agent = A.data[src_pointer]
            if (src_event-start) >= 0 and paint_fraction[src_event-start] > 0:
                p = paint_fraction[src_event-start]/n_exit[src_event]
                current_p_log_p -= - n_agent*p*np.log(p)
                paint_fraction[dst_event-start] += n_agent*p

        p_dst = paint_fraction[dst_event-start]
        if p_dst:
            p_dst /= n_exit[dst_event]
            current_p_log_p += - n_exit[dst_event]*p_dst*np.log(p_dst)
        entropy[dst_event-start] = current_p_log_p

    return entropy
