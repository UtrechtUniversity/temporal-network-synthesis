import networkx as nx
from matplotlib import pyplot as plt
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


def plot_entropy_game(*results, events):
    assert len(results) > 0

    res_min = 1e7
    res_max = -1e7
    for res in results:
        plt.plot(res)
        res_min = min(np.nanmin(res), res_min)
        res_max = max(np.nanmax(res), res_max)

    plt.vlines(events, res_min, res_max, color='red', linestyle="--")
    plt.show()


def plot_process_results(process_results, log_x_scale=False):
    for res in process_results:
        plt.plot(res)

    if log_x_scale:
        plt.xscale("log")
    plt.show()


def plot_measure_results(measure_results):
    for measure_name, all_res in measure_results.items():
        plt.title(measure_name)
        for res in all_res:
            plt.plot(res)
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


def plot_pvm_dt(process_results, measure_results):
    x_axis = [np.mean(r) for r in process_results]
    for measure_name, res in measure_results.items():
        all_cor = []
        for i in range(len(res[0])):
            y_vals = np.array([r[i] for r in res])
            cor = spearmanr(x_axis, y_vals).correlation
            all_cor.append(cor)
        plt.plot(all_cor, label=f"{measure_name}")
    plt.legend()
    plt.show()
