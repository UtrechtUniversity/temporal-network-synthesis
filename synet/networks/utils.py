from synet.networks.base import BaseNetwork
from copy import deepcopy

import numpy as np


def merge_networks(*networks, n_events=0):
    """Merge two networks into one.

    Optionally add more inter community links to the network.

    Arguments
    ---------
    networks: BaseNetwork
        The networks to be merged.
    n_events: int
        The number of events added as inter community events.

    Returns
    -------
    network: BaseNetwork
        The merged network.
    """
    event_types = deepcopy(networks[0].event_types)
    cur_n_net_id = len(event_types)
    cur_n_agents = networks[0].n_agents
    participants = [networks[0].participants]
    event_sources = [networks[0].event_sources]
    event_times = [networks[0].event_times]
    agent_rates = [networks[0].agent_rates]
    for net in networks[1:]:
        event_sources.append(net.event_sources+len(event_types))
        participants.append(net.participants+cur_n_agents)
        event_times.append(net.event_times)
        agent_rates.append(net.agent_rates)
        for net_key in net.event_types.values():
            event_types[cur_n_net_id] = _find_key(event_types, net_key)
            cur_n_net_id += 1
        cur_n_agents += net.n_agents

    time_span = np.max([net.time_span for net in networks])
    event_size = np.max([net.event_size for net in networks])

    agent_rates = np.concatenate(agent_rates)
    agent_prob = agent_rates/np.sum(agent_rates)
    event_times.append(get_event_times(n_events, time_span))
    merge_participants = np.zeros((n_events, event_size))
    for i_event in range(n_events):
        merge_participants[i_event, :] = np.random.choice(
            cur_n_agents, size=event_size, replace=False,
            p=agent_prob)
    participants.append(merge_participants)
    event_types[cur_n_net_id] = _find_key(event_types, "merge")
    event_sources.append(np.full(n_events, cur_n_net_id, dtype=int))
    cur_n_net_id += 1

    new_network = BaseNetwork()
    new_network.event_times = np.concatenate(event_times)
    time_order = np.argsort(new_network.event_times)
    new_network.event_times = new_network.event_times[time_order]
    new_participants = np.concatenate(participants).astype(int)
    new_network.participants = new_participants[time_order]
    new_network.event_sources = np.concatenate(event_sources)[time_order]
    new_network.event_types = event_types
    new_network.n_agents = cur_n_agents
    new_network.time_span = time_span
    new_network.agent_rates = agent_rates
    return new_network


def get_event_times(n_events, time_span):
    """ Generate event times.

    Uses exponentially distributed times between events.

    Arguments
    ---------
    n_events: int
        Number of events to assign times for.
    """
    event_times = np.cumsum(np.random.exponential(size=n_events+1))
    event_times *= time_span/event_times[-1]
    return event_times[:-1]


def _find_key(event_types, old_key):
    """Helper function for finding new names"""
    base_key = old_key.split("-")[0]
    new_key = base_key
    i_try = 0
    while new_key in event_types.values():
        i_try += 1
        new_key = base_key + f"-{i_try}"
    return new_key
