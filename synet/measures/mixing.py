from numba import njit
import numpy as np
from synet.measures.base import BasePaintEntropy


class MixingEntropy(BasePaintEntropy):
    """Mixing or paint-conserving entropy."""
    name = "mixing"

    def measure_entropy(self, net, start, end, **kwargs):
        return mixing_entropy(net, start, end, **kwargs)


def mixing_entropy(net, start=1, end=None, numba=True):
    A = net.A
    if end is None:
        end = net.n_events

    entropy = np.full(end-start+1, -1, dtype=float)
    n_exit = np.array(A.sum(axis=1)).flatten()
    current_p_log_p = np.log(n_exit[start])
    paint_fraction = np.zeros(end-start)

    if numba:
        return _numba_mixing_entropy(
            A.indptr, A.data, A.indices, entropy, paint_fraction,
            current_p_log_p, n_exit, start, end)
    return _python_mixing_entropy(A, entropy, paint_fraction, current_p_log_p,
                                  n_exit, start, end)


@njit
def _numba_mixing_entropy(
        A_indptr, A_data, A_indices, entropy, paint_fraction, current_p_log_p,
        n_exit, start, end):
    """Numba computation (much faster)."""
    paint_fraction[0] = 1
    entropy[0] = 0
    entropy[1] = current_p_log_p

    for dst_event in range(start+1, end):
        for src_pointer in range(A_indptr[dst_event], A_indptr[dst_event+1]):
            src_event = A_indices[src_pointer]
            n_agent = A_data[src_pointer]
            if (src_event-start) >= 0 and paint_fraction[src_event-start] > 0:
                p = paint_fraction[src_event-start]/n_exit[src_event]
                current_p_log_p -= - n_agent*p*np.log(p)
                paint_fraction[dst_event-start] += n_agent*p

        p_dst = paint_fraction[dst_event-start]
        if p_dst:
            p_dst /= n_exit[dst_event]
            current_p_log_p += - n_exit[dst_event]*p_dst*np.log(p_dst)
        entropy[dst_event-start+1] = current_p_log_p

    return entropy


def _python_mixing_entropy(
        A, entropy, paint_fraction, current_p_log_p, n_exit, start, end):
    "Python variant."
    paint_fraction[0] = 1
    entropy[0] = 0
    entropy[1] = current_p_log_p

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
        entropy[dst_event-start+1] = current_p_log_p

    return entropy
