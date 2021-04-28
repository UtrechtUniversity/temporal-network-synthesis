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


def entropy_windows(net, dt, entropy_game, window_t_frac=0.1, window_dt_frac=0.3):
    n_events = net.n_events
    eq_start = n_events//10
    eq_end = 9*n_events//10
    max_dt = dt*2
    results = np.zeros(n_events)
    counts = np.zeros(n_events)

    window_t = max(1, round(dt*window_t_frac))
    window_dt = max(1, round(dt*window_dt_frac))

    for t_start in range(eq_start, eq_end-max_dt):
        t_end = t_start + max_dt
        entropy = entropy_game(net, t_start, t_end)
        add_results(results, counts, entropy, t_start, dt, window_t=window_t, window_dt=window_dt)
    return results/counts


def entropy_dt(net, max_dt, entropy_game):
    n_events = net.n_events
    eq_start = n_events//10
    eq_end = 9*n_events//10

    entropy_avg = np.zeros(max_dt)
    for t_start in range(eq_start, eq_end-max_dt):
        t_end = t_start + max_dt
        entropy = entropy_game(net, t_start, t_end)
        entropy_avg += entropy
    return entropy_avg/(eq_end-eq_start-max_dt)


def fixed_entropy_dt(net, max_dt, entropy_game):
    n_events = net.n_events

    last_events = np.zeros(net.n_agents, dtype=int)
    entropy_avg = np.zeros(max_dt)
    for t_start in range(n_events-max_dt):
        t_end = t_start + max_dt
        agents = net.participants[t_start]
        entropy = entropy_game(net, t_start, t_end)
        all_prev_dt = t_start - last_events[agents]
        for prev_dt in all_prev_dt:
            new_sum = np.cumsum(entropy)
            if prev_dt < max_dt:
                new_sum[prev_dt-max_dt:] -= new_sum[:max_dt-prev_dt]
            entropy_avg += new_sum
        last_events[agents] = t_start
    return entropy_avg/(n_events-max_dt)


def alpha_eff(entropy):
    n_measure = len(entropy)
    t_start = np.arange(n_measure-2)+1
    t_end = np.arange(n_measure-2)+2
    t_avg = (t_start+t_end)/2
    alpha_eff = np.log(entropy[1:-1]/entropy[2:])/np.log(t_start/t_end)
    return t_avg, alpha_eff
