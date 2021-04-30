from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import spearmanr


def plot_matrix(laplacian):
    laplacian_copy = laplacian.copy()
    np.fill_diagonal(laplacian_copy, 0)

    plt.imshow(laplacian_copy)
    plt.show()


def plot_community(A, block_list=None):
    colors = ["red", "blue", "green", "yellow"]
    rows, cols = np.where(A == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.OrderedGraph()
    gr.add_edges_from(edges)
    reverse_nodes = {x: i for i, x in enumerate(gr.nodes)}
    labels = {i: str(i) for i in range(A.shape[0])}
    color_map = np.array(A.shape[0]*["red"], dtype=object)
    for i_block, block in enumerate(block_list):
        for node in block:
            color_map[reverse_nodes[node]] = colors[i_block]
    nx.draw(gr, labels=labels, node_color=color_map, with_labels=True)
    plt.show()


def plot_entropy_with_events(*results, events):
    assert len(results) > 0

    res_min = 1e7
    res_max = -1e7
    for res in results:
        plt.plot(res)
        res_min = min(np.nanmin(res[res > 0]), res_min)
        res_max = max(np.nanmax(res[res > 0]), res_max)

    plt.ylim(res_min, res_max)
    plt.vlines(events, res_min, res_max, color='red', linestyle="--")
    plt.show()


def plot_process_results(process_results, log_x_scale=False, title=""):
    if isinstance(process_results, dict):
        for name, proc in process_results.items():
            plt.title(title + " " + name)
            plot_process_results(proc, log_x_scale)
        return

    for res in process_results:
        plt.plot(np.mean(res, axis=0))

    if log_x_scale:
        plt.xscale("log")
    plt.show()


def plot_measure_results(measure_results):
    for measure_name, all_res in measure_results.items():
        plt.title(measure_name)
        for res in all_res:
            plt.plot(res)
        plt.show()


def plot_correlation_network(process_results, measure_results, n_resample=30):
    avg_process_results = np.array([np.mean(r) for r in process_results])
    avg_measure_results = {name: np.array([np.mean(r) for r in all_r])
                           for name, all_r in measure_results.items()}
    n_network = len(avg_process_results)

    for name, m_result in avg_measure_results.items():
        network_select = []
        all_correlations = []
        for cur_network_select in range(3, n_network+1):
            network_select.append(cur_network_select)
            correlations = []
            for _ in range(n_resample):
                select_idx = np.random.choice(
                    n_network, size=cur_network_select, replace=True)
                if len(np.unique(select_idx)) == 1:
                    continue
                cor = spearmanr(avg_process_results[select_idx],
                                m_result[select_idx]).correlation
                correlations.append(cor)
            all_correlations.append(np.mean(correlations))
        all_correlations = np.array(all_correlations)
        network_select = np.array(network_select)
        plt.plot(1/network_select, all_correlations, label=name)
    plt.legend()
    plt.xlim(0, 1/3)
    plt.show()


def bootstrap_sim(process_results, n_bootstrap=100, n_resample=None):
    if n_resample is None:
        n_resample = process_results[0].shape[0]
    n_network = len(process_results)

    bootstrap_results = np.zeros((n_network, n_bootstrap))
    for i_net in range(n_network):
        n_sample = process_results[i_net].shape[0]
        for i_bootstrap in range(n_bootstrap):
            sample_idx = np.random.choice(n_sample, size=n_resample,
                                          replace=True)
            new_results = process_results[i_net][sample_idx]
            new_results += 1e-7*(np.random.randn(*new_results.shape)-0.5)
            bootstrap_results[i_net, i_bootstrap] = np.mean(new_results)
    return bootstrap_results


def plot_bootstrap_sim(process_results, measure_results, n_bootstrap=100):
    n_sim = process_results[0].shape[0]
    middle = n_bootstrap//2
    edge = n_bootstrap//50
    all_n_resample = np.unique((n_sim*(3+np.arange(10))/12).astype(int))
    for measure_name, res in measure_results.items():
        meas_res = [np.mean(r) for r in res]
        y_min = []
        y_max = []
        y_median = []
        for n_resample in all_n_resample:
            bootstrap_results = bootstrap_sim(
                process_results, n_bootstrap=n_bootstrap,
                n_resample=n_resample)
            all_cor = np.zeros(n_bootstrap)
            for i_boot in range(n_bootstrap):
                proc_res = bootstrap_results[:, i_boot]
                all_cor[i_boot] = spearmanr(proc_res, meas_res).correlation
            sorted_cor = np.sort(all_cor)
            y_min.append(sorted_cor[edge])
            y_median.append(sorted_cor[middle])
            y_max.append(sorted_cor[n_bootstrap-edge-1])

        y_error = np.vstack((np.array(y_median)-np.array(y_min),
                             np.array(y_max)-np.array(y_median)))
        plt.errorbar(1/np.array(all_n_resample), y_median, yerr=y_error,
                     label=measure_name)
    plt.legend()
    plt.show()


def plot_process_v_measure(process_results, measure_results):
    x_axis = [np.mean(r) for r in process_results]
    for measure_name, res in measure_results.items():
        y_res = np.array([np.mean(r) for r in res])
        y_res -= y_res.min()
        y_res /= y_res.max()
        cor = spearmanr(x_axis, y_res).correlation
        plt.scatter(x_axis, y_res, label=f"{measure_name}: {cor: .3f}")
    plt.legend()
    plt.show()


def plot_measure_v_process(measure_results, process_results):
    meas = measure_results[list(measure_results)[0]]
    x_axis = [np.mean(r) for r in meas]
    for process_name, res in process_results.items():
        y_res = np.array([np.mean(r) for r in res])
        y_res -= y_res.min()
        y_res /= y_res.max()
        cor = spearmanr(x_axis, y_res).correlation
        plt.scatter(x_axis, y_res, label=f"{process_name}: {cor: .3f}")
    plt.legend()
    plt.show()


def plot_pvm_dt(process_results, measure_results):
    x_axis = [np.mean(r) for r in process_results]
    plt.xlabel("dt")
    plt.ylabel("correlation")
    for measure_name, res in measure_results.items():
        all_cor = []
        try:
            n_dt = len(res[0])
        except TypeError:
            continue
        for i in range(n_dt):
            y_vals = np.array([r[i] for r in res])
            cor = spearmanr(x_axis, y_vals).correlation
            all_cor.append(cor)
        plt.plot(all_cor, label=f"{measure_name} (max={np.nanmax(all_cor)})")
    plt.legend()
    plt.show()


def plt_pvm_alpha(process_results, measure_results, dt):
    x_vals = np.array([np.mean(r) for r in process_results])
    t_start = dt + 1
    t_end = dt + 2
    entropy_start = np.array([meas[t_start] for meas in measure_results])
    entropy_end = np.array([meas[t_end] for meas in measure_results])
    y_vals = np.log(entropy_start/entropy_end)/np.log(t_start/t_end)
    return spearmanr(x_vals, y_vals).correlation
