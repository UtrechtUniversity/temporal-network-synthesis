import numpy as np
from numba.core.decorators import njit


@njit
def add_value(log_val1, log_val2):
    if log_val1 == -1:
        return log_val2
    if log_val2 == -1:
        return log_val1

    log_Z = max(log_val1, log_val2)
    exp_val = np.exp(log_val1-log_Z) + np.exp(log_val2-log_Z)
    return log_Z + np.log(exp_val)


@njit
def sub_value(log_val1, log_val2):
    '''log_val1 - log_val2'''
    if log_val1 == -1 or log_val1 < log_val2:
        return -1
    if log_val2 == -1:
        return log_val1

    exp_val = 1 - np.exp(log_val2-log_val1)
    log_val = log_val1 + np.log(exp_val)
    if log_val < np.log(0.5):
        return -1
    return log_val
