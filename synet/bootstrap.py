import numpy as np
from scipy.stats.stats import spearmanr


def create_net_boot(n_network, n_bootstrap):
    network_idx = np.empty((n_bootstrap, n_network), dtype=int)
    for i in range(n_bootstrap):
        boot_idx = np.random.choice(n_network, size=n_network,
                                    replace=True)
        network_idx[i] = boot_idx
    return network_idx


def create_process_bootstrap(process_results, n_bootstrap=100, network_idx=None):
    if isinstance(process_results, dict):
        n_network = len(process_results[list(process_results)[0]])

        if network_idx is None:
            network_idx = create_net_boot(n_network, n_bootstrap)
        return {n: create_process_bootstrap(p, n_bootstrap=n_bootstrap,
                                            network_idx=network_idx)
                for n, p in process_results.items()}

    n_network = len(process_results)
    if network_idx is None:
        network_idx = create_net_boot(n_network, n_bootstrap)

    n_sample = process_results[0].shape[0]
    avg_process_results = np.zeros((n_network, n_sample))
    boot_process_results = np.zeros((n_bootstrap, n_network))
    for i_net in range(n_network):
        for i_sample in range(n_sample):
            avg_process_results[i_net, i_sample] = np.mean(
                process_results[i_net][i_sample])

    for i_bootstrap in range(n_bootstrap):
        cur_net_idx = network_idx[i_bootstrap]
        cur_sample_idx = np.random.choice(
            n_sample, size=n_sample, replace=True)
        res = np.mean(avg_process_results[cur_net_idx][:, cur_sample_idx], axis=1)
        boot_process_results[i_bootstrap] = res
    return boot_process_results, network_idx


def create_xcor_boot_matrix(boot_proc_results, measure_results):
    avg_meas_results = {
        name: np.array([np.mean(r) for r in res])
        for name, res in measure_results.items()
    }
    proc_names = list(boot_proc_results)
    meas_names = list(measure_results)
    cor_names = (list(boot_proc_results), list(avg_meas_results))
    n_boot = boot_proc_results[proc_names[0]][0].shape[0]
    xcor_matrix = np.zeros((len(boot_proc_results), len(avg_meas_results), n_boot))
    for proc_name, proc_res in boot_proc_results.items():
        proc_data, net_idx = proc_res
        for meas_name, meas_res in avg_meas_results.items():
            correlations = []
            n_boot = proc_data.shape[0]
            for i_boot in range(n_boot):
                cor = spearmanr(proc_data[i_boot], meas_res[net_idx[i_boot]]).correlation
                correlations.append(cor)
            idx = proc_names.index(proc_name), meas_names.index(meas_name)
            xcor_matrix[idx] = correlations
    return xcor_matrix, cor_names


def avg_correlations(xcor_matrix, meas_names):
    avg_xcor_matrix = np.mean(xcor_matrix, axis=0)
    n_boot = avg_xcor_matrix.shape[1]
    median = n_boot//2
    lower = n_boot//50
    upper = n_boot - 1 - lower
    return {
        meas_names[i]: {
            "lower": [avg_xcor_matrix[i, lower]],
            "median": [avg_xcor_matrix[i, median]],
            "upper": [avg_xcor_matrix[i, upper]],
            "sem": [np.std(avg_xcor_matrix[i])]
        }
        for i in range(len(meas_names))
    }
