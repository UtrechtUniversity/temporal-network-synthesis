import numpy as np
from numba.core.decorators import njit


@njit
def norm(dx, sigma):
    return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-dx**2/sigma**2)


@njit
def add_results(results, counts, entropy, t_start, dt, window_t, window_dt):
    for t_entropy in range(len(entropy)):
        t_real_mean = (2*t_start+t_entropy)/2
        current_dt = t_entropy/2
        dt_diff = current_dt - dt
        if current_dt < dt - 2*window_dt or current_dt > dt + 2*window_dt:
            continue
        dt_factor = norm(dt_diff, dt)
        t_res_start = max(0, int(t_real_mean - 2*window_t+0.5))
        t_res_end = min(len(results), int(t_real_mean + 2*window_t+0.5))
        for t in range(t_res_start, t_res_end):
            t_factor = norm(t_real_mean - t, window_t)
            results[t] += entropy[t_entropy]*t_factor*dt_factor
            counts[t] += t_factor*dt_factor


def entropy_windows(A, dt, entropy_game, window_t_frac=0.1, window_dt_frac=0.3):
    n_events = A.shape[0]
    eq_start = n_events//10
    eq_end = 9*n_events//10
    max_dt = dt*2
    results = np.zeros(n_events)
    counts = np.zeros(n_events)

    window_t = max(1, round(dt*window_t_frac))
    window_dt = max(1, round(dt*window_dt_frac))

    for t_start in range(eq_start, eq_end-max_dt):
        t_end = t_start + max_dt
        entropy = entropy_game(A, t_start, t_end)
        add_results(results, counts, entropy, t_start, dt, window_t=window_t, window_dt=window_dt)
    return results/counts


def entropy_dt(A, max_dt, entropy_game, window_t_frac=0.1, window_dt_frac=0.3):
    n_events = A.shape[0]
    eq_start = n_events//10
    eq_end = 9*n_events//10

    entropy_avg = np.zeros(max_dt)
    for t_start in range(eq_start, eq_end-max_dt):
        t_end = t_start + max_dt
        entropy = entropy_game(A, t_start, t_end)
        entropy_avg += entropy
    return entropy_avg/(eq_end-eq_start-max_dt)
