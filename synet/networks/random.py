import numpy as np
from synet.networks.heterogeneous import HeterogeneousNetwork
from synet.networks.homogeneous import HomogeneousNetwork
from synet.networks.utils import merge_networks


def random_network(n_events=100, n_agents=100, n_community=2, seed=None):
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
    seed: int
        Seed the random number generator
    Returns
    -------
    network: BaseNetwork
        Network with specified number of agents and events.
    """
    n_inter_events = round(n_events*np.random.rand())
    n_intra_events = n_events-n_inter_events

    # Number of agents in communities 1 and 2.
    min_agent = 10
    assert n_agents >= n_community*min_agent

    n_agents_group = distribute_randomly(
        n_agents-n_community*min_agent, n_community, max_rate=4) + min_agent
    n_events_group = distribute_randomly(
        n_intra_events, n_community, base_rates=n_agents_group, max_rate=2)

    assert np.sum(n_events_group) == n_intra_events
    assert np.sum(n_agents_group) == n_agents

    # Generate the individual networks.
    networks = []
    for i in range(n_community):
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


def distribute_randomly(n_items, n_community, base_rates=None, max_rate=3):
    if base_rates is None:
        base_rates = np.ones(n_community)

    item_rates = (1+(max_rate-1)*np.random.rand(n_community))
    item_rates /= np.sum(item_rates)
    n_item_group_fp = item_rates*n_items
    n_item_group = n_item_group_fp.astype(int)
    n_item_rest = n_item_group_fp-n_item_group

    n_left = round(np.sum(n_item_rest))
    for _ in range(n_left):
        i_group = np.random.choice(n_community, p=n_item_rest/np.sum(n_item_rest))
        n_item_group[i_group] += 1
        n_item_rest[i_group] = 0
    return n_item_group


def random_two_split_network(n_events, n_inter_events, n_agents):
    n_intra_events = n_events-n_inter_events
    net1 = HomogeneousNetwork(n_events=n_intra_events//2, n_agents=n_agents//2)
    net2 = HomogeneousNetwork(n_events=n_intra_events-n_intra_events//2,
                              n_agents=n_agents-n_agents//2)
    net = merge_networks(net1, net2, n_events=n_inter_events)
    return net

