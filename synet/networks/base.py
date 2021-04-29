import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse.csc import csc_matrix


class BaseNetwork():
    """ Basis for networks classes.

    It mainly contains some functionality for visualization and
    the computation of adjacency matrices. The following data members are
    assumed to be set:

    --- necessary ---
    n_agents (int): The number of agents in the network.
    event_types (dict[int: str]): The name of each event type
        (i.e. how it was generated).
    event_sources (np.ndarray[int]): The event type id for each of the events.
    participants (np.ndarray[float, float]): agent ids present at each event.

    --- optional ---
    agent_rates (np.ndarray[float]): The rate at which each agent participates
        in events.
    event_times (np.ndarray[float]): Time each event takes place.
    time_span (np.ndarray[float]): Time over which events take place
        (not equal to min/max of event_times necessarily).
    """
    _A = None

    def plot(self):
        """ Plot the network with the networkx library.

        Works mainly for smaller networks.
        """
        rows, cols = self.A.nonzero()
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.OrderedGraph()
        gr.add_edges_from(edges)
        reverse_nodes = {x: i for i, x in enumerate(gr.nodes)}
        labels = {i: str(i) for i in range(self.A.shape[0])}
        labels[0] = "s"
        labels[self.n_events+1] = "t"

        color_map = np.zeros(self.n_events+2)
        for i_event in np.arange(1, self.n_events+1):
            color_map[reverse_nodes[i_event]] = self.event_sources[i_event-1]+1

        color_map[reverse_nodes[0]] = 0
        color_map[reverse_nodes[self.n_events+1]] = 0
        norm = max(0.01, np.max(color_map))
        color_map /= norm
        nx.draw(gr, cmap=plt.get_cmap("gist_rainbow"), labels=labels,
                node_color=color_map, with_labels=True)
        plt.show()

    def plot_matrix(self):
        "Plot the laplacian as a 2D image."
        laplacian_copy = self.A.todense()
        np.fill_diagonal(laplacian_copy, 0)

        plt.imshow(laplacian_copy)
        plt.show()

    @property
    def event_size(self):
        "Get the event size of the network."
        return self.participants.shape[1]

    @property
    def n_events(self):
        "Get the number of events in the network."
        return self.participants.shape[0]

    @property
    def A(self):
        "Compute the adjacency matrix if necessary"
        if self._A is None:
            self._A = self.compute_adjacency()
        return self._A

    def compute_adjacency(self):
        """Compute the adjacency matrix from the event graph.

        Returns
        -------
        adjaceny matrix: csc_matrix
            Upper triangular matrix.
        """
        A_rows = np.empty(2*self.n_agents + self.event_size*self.n_events,
                          dtype=int)
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

        last_connections = last_event[last_event != 0]
        previous_events, counts = np.unique(last_connections,
                                            return_counts=True)
        n_links = len(previous_events)
        A_rows[n_data: n_data+n_links] = previous_events
        A_cols[n_data: n_data+n_links] = self.n_events+1
        A_data[n_data: n_data+n_links] = counts
        n_data += n_links
        A_rows.resize(n_data)
        A_cols.resize(n_data)
        A_data.resize(n_data)
        A_sparse = csc_matrix((A_data, (A_rows, A_cols)),
                              shape=(self.n_events+2, self.n_events+2))
        return A_sparse
