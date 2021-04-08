import numpy as np

from synet.process.base import BaseProcess


class PredatorProcess(BaseProcess):
    def __init__(self):
        pass

    def _simulate(self, net, start=0, end=None, seed=None):
        np.random.seed(seed)
        if end is None:
            end = net.n_events
        return simulate_predator(net, start, end)


def simulate_predator(net, start, end):
    n_agents = net.n_agents
    participants = net.participants

    agent_types = np.zeros(n_agents, dtype=int)
    agent_types[:n_agents//2] = 1
    np.random.shuffle(agent_types)
    event_size = net.event_size
    cur_predators = np.sum(agent_types)
    results = np.empty(end-start, dtype=float)

    for dst_event in range(start, end):
        cur_agents = participants[dst_event]
        predator_idx = cur_agents[np.where(agent_types[cur_agents] == 1)[0]]
        prey_idx = cur_agents[np.where(agent_types[cur_agents] == 0)[0]]
        n_predators = len(predator_idx)
        n_prey = len(prey_idx)
        new_predators = min(event_size, 2*n_predators, n_prey//2)
        if new_predators > n_predators:
            new_predator_idx = np.random.choice(prey_idx, size=new_predators-n_predators, replace=False)
            agent_types[new_predator_idx] = 1
        elif new_predators < n_predators:
            new_prey_idx = np.random.choice(predator_idx, size=n_predators-new_predators, replace=False)
            agent_types[new_prey_idx] = 0

        cur_predators += (new_predators-n_predators)
        results[dst_event-start] = cur_predators/n_agents
    return 1-results
