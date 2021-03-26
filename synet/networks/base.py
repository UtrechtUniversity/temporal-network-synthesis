# from abc import ABC

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse.csc import csc_matrix


class BaseNetwork():
    _A = None

    def plot(self):
        rows, cols = self.A.nonzero()
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.OrderedGraph()
        gr.add_edges_from(edges)
        reverse_nodes = {x: i for i, x in enumerate(gr.nodes)}
        labels = {i: str(i) for i in range(self.A.shape[0])}

        color_map = np.zeros(self.n_events+1)
        for i_event in np.arange(1, self.n_events+1):
            color_map[reverse_nodes[i_event]] = self.event_sources[i_event-1]+1

        color_map[reverse_nodes[0]] = 0
        norm = max(0.01, np.max(color_map))
        color_map /= norm
        nx.draw(gr, cmap=plt.get_cmap("gist_rainbow"), labels=labels,
                node_color=color_map, with_labels=True)
        plt.show()

    def plot_matrix(self):
        laplacian_copy = self.A.todense()
        np.fill_diagonal(laplacian_copy, 0)

        plt.imshow(laplacian_copy)
        plt.show()

    @property
    def A(self):
        if self._A is None:
            self._A = self.compute_adjacency()
        return self._A

    def compute_adjacency(self):
        A_rows = np.empty(self.event_size*(self.n_events+1), dtype=int)
        A_cols = np.empty_like(A_rows)
        A_data = np.ones_like(A_rows)
        n_data = 0
        last_event = np.zeros(self.n_agents, dtype=int)

        for i_event in range(1, self.n_events+1):
            cur_participants = self.participants[i_event-1, :]
            previous_events, counts = np.unique(last_event[cur_participants],
                                                return_counts=True)

            n_links = len(previous_events)
            A_rows[n_data: n_data+n_links] = previous_events
            A_cols[n_data: n_data+n_links] = i_event
            A_data[n_data: n_data+n_links] = counts
            n_data += n_links
            last_event[cur_participants] = i_event
        A_rows.resize(n_data)
        A_cols.resize(n_data)
        A_data.resize(n_data)
        A_sparse = csc_matrix((A_data, (A_rows, A_cols)),
                              shape=(self.n_events+1, self.n_events+1))
        return A_sparse
