import numpy as np
from synet.networks.heterogeneous import HeterogeneousNetwork
from synet.networks.homogeneous import HomogeneousNetwork
from synet.networks.utils import merge_networks


def random_network(n_events=100, n_agents=40):
    networks = []

    n_inter_events = round(n_events*np.random.rand())
    n_intra_events = n_events-n_inter_events
#     n_events_1 = round((0.25+0.5*np.random.rand())*n_events)
    n_agents_1 = round((0.15+0.7*np.random.rand())*n_agents)
    n_agents_1 = max(10, n_agents_1)
    n_agents_1 = min(n_agents-10, n_agents_1)
    n_agents_2 = n_agents-n_agents_1

    min_events_1 = round(n_intra_events*0.5*n_agents_1/n_agents)
    min_events_2 = round(n_intra_events*0.5*n_agents_2/n_agents)
    max_events_1 = round(n_intra_events*2*n_agents_1/n_agents)
    max_events_2 = round(n_intra_events*2*n_agents_2/n_agents)
    min_events = max(min_events_1, n_intra_events-max_events_2)
    max_events = min(max_events_1, n_intra_events-min_events_2)

    n_events_1 = np.random.randint(min_events, max_events+1)
#     n_events_group_1 = events_per_agent_1*n_agents_group_1
    n_events_group = [n_events_1, n_intra_events-n_events_1]
    n_agents_group = [n_agents_1, n_agents_2]
    assert np.sum(n_events_group) == n_intra_events
    assert np.sum(n_agents_group) == n_agents
    for i in range(2):
        if np.random.rand() < 0.5:
            net = HomogeneousNetwork(n_agents=n_agents_group[i],
                                     n_events=n_events_group[i], time_span=1)
        else:
            heterogeniety = 0.2+0.8*np.random.rand()
            net = HeterogeneousNetwork(
                n_agents=n_agents_group[i], n_events=n_events_group[i],
                time_span=1, heterogeniety=heterogeniety)
        networks.append(net)

    final_network = merge_networks(*networks, n_events=n_inter_events)
    assert final_network.n_events == n_events
    return final_network
