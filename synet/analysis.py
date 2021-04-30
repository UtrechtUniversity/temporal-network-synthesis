import numpy as np


def alpha_eff(entropy):
    n_measure = len(entropy)
    t_start = np.arange(n_measure-2)+1
    t_end = np.arange(n_measure-2)+2
    t_avg = (t_start+t_end)/2
    alpha_eff = np.log(entropy[1:-1]/entropy[2:])/np.log(t_start/t_end)
    return t_avg, alpha_eff
