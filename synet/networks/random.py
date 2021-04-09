import numpy as np
from synet.networks.heterogeneous import HeterogeneousNetwork
from synet.networks.homogeneous import HomogeneousNetwork
from synet.networks.utils import merge_networks


def random_network(n_events=100, n_agents=40):
    """ Generate a random network.

    Generates random networks with a set number of events and agents.
    The following constraints hold:
    - The number of agents in each event is equal to 5.
    - There are two communities with unique agents.
    - A community is heterogeneous/homogeneous with probability 0.5 each.
    - A community has at least 10 members.
    - The number of events in each community varies, depending on the
        number of agents and is slightly random.
    - No simulataneous events.
    - Events have no time duration.

    Arguments
    ---------
    n_events: int
        Number of events in the final network.
    n_agents: int
        Number of agents in the final network.

    Returns
    -------
    network: BaseNetwork
        Network with specified number of agents and events.
    """
    n_inter_events = round(n_events*np.random.rand())
    n_intra_events = n_events-n_inter_events

    # Number of agents in communities 1 and 2.
    n_agents_1 = round((0.15+0.7*np.random.rand())*n_agents)
    n_agents_1 = max(10, n_agents_1)
    n_agents_1 = min(n_agents-10, n_agents_1)
    n_agents_2 = n_agents-n_agents_1
    n_agents_group = [n_agents_1, n_agents_2]

    # Compute the number of events in each community.
    min_events_1 = round(n_intra_events*0.5*n_agents_1/n_agents)
    min_events_2 = round(n_intra_events*0.5*n_agents_2/n_agents)
    max_events_1 = round(n_intra_events*2*n_agents_1/n_agents)
    max_events_2 = round(n_intra_events*2*n_agents_2/n_agents)
    min_events = max(min_events_1, n_intra_events-max_events_2)
    max_events = min(max_events_1, n_intra_events-min_events_2)
    n_events_1 = np.random.randint(min_events, max_events+1)
    n_events_group = [n_events_1, n_intra_events-n_events_1]

    assert np.sum(n_events_group) == n_intra_events
    assert np.sum(n_agents_group) == n_agents

    # Generate the individual networks.
    networks = []
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

    # Merge the networks.
    final_network = merge_networks(*networks, n_events=n_inter_events)
    assert final_network.n_events == n_events
    return final_network
