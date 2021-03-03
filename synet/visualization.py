import networkx as nx
from matplotlib import pyplot as plt
import numpy as np


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
