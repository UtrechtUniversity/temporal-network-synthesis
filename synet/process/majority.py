import numpy as np

from synet.process.base import BaseProcess


class MajorityProcess(BaseProcess):
    def __init__(self, n_color=3):
        self.n_color = n_color

    def _simulate(self, net, start=0, end=None, seed=None):
        np.random.seed(seed)
        results = simulate_majority(
            net, start, end, n_color=self.n_color)
        return results

    def todict(self):
        return {"n_color": self.n_color}


def entropy(counts, n_agents):
    nz_counts = counts[counts>0]
    p = nz_counts/n_agents
    return np.sum(-p*np.log(p))


def simulate_majority(net, start, end, n_color=3):
    n_agents = net.n_agents
    participants = net.participants

    current_colors = np.zeros(n_agents, dtype=int)
    for i_color in range(1, n_color):
        current_colors[i_color::n_color] = i_color
    np.random.shuffle(current_colors)

    cur_col_counts = np.zeros(n_color, dtype=int)
    for i_color in range(n_color):
        cur_col_counts[i_color] = len(np.where(current_colors == i_color)[0])
    majority_count = participants.shape[1]//2 + 1
    res_entropy = np.empty(end-start, dtype=float)

    for dst_event in range(start, end):
        cur_agents = participants[dst_event]
        colors = current_colors[cur_agents]
        col, counts = np.unique(colors, return_counts=True)
        if np.max(counts) >= majority_count:
            max_col = np.argmax(counts)
            new_color = col[max_col]
            current_colors[cur_agents] = new_color
            for i_col in range(len(col)):
                cur_col_counts[col[i_col]] -= counts[i_col]
            cur_col_counts[col[max_col]] += participants.shape[1]
        res_entropy[dst_event-start] = entropy(cur_col_counts, n_agents)
    return np.log(n_agents) - res_entropy
