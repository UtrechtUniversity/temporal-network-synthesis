from synet.networks.base import BaseNetwork
from copy import deepcopy

import numpy as np


def merge_networks(*networks, n_events=0):
    event_types = deepcopy(networks[0].event_types)
    cur_n_net_id = len(event_types)
    cur_n_agents = networks[0].n_agents
    participants = [networks[0].participants]
    event_sources = [networks[0].event_sources]
    event_times = [networks[0].event_times]
    for net in networks[1:]:
        event_sources.append(net.event_sources+len(event_types))
        participants.append(net.participants+cur_n_agents)
        event_times.append(net.event_times)
        for net_id, net_key in net.event_types.items():
            event_types[cur_n_net_id] = _find_key(event_types, net_key)
            cur_n_net_id += 1
        cur_n_agents += net.n_agents

    time_span = np.max([net.time_span for net in networks])
    event_size = np.max([net.event_size for net in networks])

    event_times.append(get_event_times(n_events, time_span))
    merge_participants = np.zeros((n_events, event_size))
    for i_event in range(n_events):
        merge_participants[i_event, :] = np.random.choice(
            cur_n_agents, size=event_size, replace=False)
    participants.append(merge_participants)
    event_types[cur_n_net_id] = _find_key(event_types, "merge")
    event_sources.append(np.full(n_events, cur_n_net_id, dtype=int))
    cur_n_net_id += 1

    new_network = BaseNetwork()
    new_network.event_times = np.concatenate(event_times)
    time_order = np.argsort(new_network.event_times)
    new_network.event_times = new_network.event_times[time_order]
    new_network.participants = np.concatenate(participants).astype(int)[time_order]
    new_network.event_sources = np.concatenate(event_sources)[time_order]
    new_network.event_types = event_types
    new_network.n_agents = cur_n_agents
    new_network.time_span = time_span
    new_network.n_events = n_events + np.sum([n.n_events for n in networks])
    new_network.event_size = event_size
    return new_network


def get_event_times(n_events, time_span):
    event_times = np.cumsum(np.random.exponential(size=n_events+1))
    event_times *= time_span/event_times[-1]
    return event_times[1:]


def _find_key(event_types, old_key):
    base_key = old_key.split("-")[0]
    new_key = base_key
    i_try = 0
    while new_key in event_types.values():
        i_try += 1
        new_key = base_key + f"-{i_try}"
    return new_key
