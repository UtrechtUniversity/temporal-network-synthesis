import numpy as np
from scipy.sparse.csc import csc_matrix


def create_temporal_network(n_base_community=2, part_per_community=20,
                            event_size=5,
                            p_intern=0.8, n_events=100):
    n_part = n_base_community*part_per_community
    n_community = n_base_community + 1
#     A = np.zeros((n_events+1, n_events+1))
    last_event = np.zeros(n_part, dtype=int)
    p_community = np.ones(n_community)/n_community
    communities = []

    for i in range(n_base_community):
        p_community[i] = p_intern/n_base_community
        communities.append(np.arange(i*part_per_community, (i+1)*part_per_community))
    p_community[-1] = 1-p_intern
    communities.append(np.arange(n_part))

    # A[0, :] = 1
    event_participants = np.zeros((n_events+1, event_size), dtype=int)
    event_types = np.zeros(n_events+1, dtype=int)
    event_types[0] = n_base_community
    A_rows = np.empty(event_size*(n_events+1), dtype=int)
    A_cols = np.empty_like(A_rows)
    A_data = np.ones_like(A_rows)
    n_data = 0
    for i in range(1, n_events+1):
        i_community = np.random.choice(np.arange(n_community), p=p_community)
        participants = np.random.choice(communities[i_community],
                                        size=event_size,
                                        replace=False)
        event_participants[i, :] = participants
        previous_events, counts = np.unique(last_event[participants], return_counts=True)
        last_event[participants] = i
        n_links = len(previous_events)
        A_rows[n_data: n_data+n_links] = previous_events
        A_cols[n_data: n_data+n_links] = i
        A_data[n_data: n_data+n_links] = counts
        n_data += n_links
#         A[previous_events, i] = 1
        event_types[i] = i_community

    A_rows.resize(n_data)
    A_cols.resize(n_data)
    A_data.resize(n_data)
    A_sparse = csc_matrix((A_data, (A_rows, A_cols)), shape=(n_events+1, n_events+1))
    event_list = []
    for i in range(n_community):
        event_list.append(np.where(event_types == i)[0])
    return A_sparse, event_list, event_participants
