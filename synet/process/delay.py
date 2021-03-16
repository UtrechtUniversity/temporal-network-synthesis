import numpy as np


def simulate_delay(event_participants, start, end, n_agents=-1, p_delay=0.03,
                   tau_delay=1, ):
    if n_agents == -1:
        n_agents = np.max(event_participants) + 1

    n_zeros = n_agents//2
    n_ones = n_agents - n_zeros
    delayed = np.zeros(n_agents, dtype=np.int32)
    delayed[:n_ones] = 1
    np.random.shuffle(delayed)
    total_delay = np.zeros(end-start, dtype=np.float32)

    current_total_delay = 0
    n_current_delayed = 0
    delay_resolve_time = np.full(n_agents, -1, dtype=np.int32)
    agent_pointer = np.full(n_agents, -1, dtype=np.int32)
    tau = tau_delay*n_agents
    max_delay = round(3*tau)

    ring_buffer = np.zeros((max_delay+1, n_agents), dtype=np.int32)
    ring_pointer = np.zeros(max_delay+1, dtype=np.int32)

    cur_t = 0

    def check_delay():
        nonlocal n_current_delayed
        test_n_current_delay = 0
        for agent_id in range(n_agents):
            test_n_current_delay += (delay_resolve_time[agent_id] != -1)
        assert test_n_current_delay == n_current_delayed, f"{test_n_current_delay}, {n_current_delayed}"

        nonlocal current_total_delay
        test_total_delay = 0
        for agent_id in range(n_agents):
            if delay_resolve_time[agent_id] == -1:
                continue
            delay = (delay_resolve_time[agent_id] - cur_t + (max_delay+1)) % (max_delay+1)
            test_total_delay += delay
        assert test_total_delay == current_total_delay, f"total: {test_total_delay} {current_total_delay}"

    def remove_delay(agent_id):
        nonlocal n_current_delayed, cur_t, current_total_delay
        old_time_slot = delay_resolve_time[agent_id]
        if old_time_slot == -1:
            return

        old_pointer = agent_pointer[agent_id]
        if old_pointer != ring_pointer[old_time_slot]-1:
            other_agent = ring_buffer[old_time_slot][ring_pointer[old_time_slot]-1]
            ring_buffer[old_time_slot][old_pointer] = other_agent
            agent_pointer[other_agent] = old_pointer
        ring_pointer[old_time_slot] -= 1
        delay_resolve_time[agent_id] = -1
        n_current_delayed -= 1
        old_delay = (old_time_slot - cur_t + max_delay + 1) % (max_delay+1)
        current_total_delay -= old_delay
#         check_delay()

    def set_delay(agent_id, time_slot=-1):
        nonlocal n_current_delayed, cur_t, current_total_delay
        if time_slot < 0:
            delay = round(np.random.exponential(scale=tau))
            delay = min(max_delay, delay)
            time_slot = (cur_t + delay) % (max_delay+1)
#             print(f"new_delay: {delay}")
        else:
            delay = (time_slot - cur_t + max_delay + 1) % (max_delay+1)
        delay_resolve_time[agent_id] = time_slot
        ring_buffer[time_slot][ring_pointer[time_slot]] = agent_id
        agent_pointer[agent_id] = ring_pointer[time_slot]
        ring_pointer[time_slot] += 1
        n_current_delayed += 1
        current_total_delay += delay
#         check_delay()

#     check_delay()

    for agent_id in np.where(delayed == 1)[0]:
        set_delay(agent_id)

    for dst_event in range(start, end):
        # Remove agents that get 0 delay
        for pointer_id in range(ring_pointer[cur_t]):
            agent_id = ring_buffer[cur_t][pointer_id]
            delay_resolve_time[agent_id] = -1
        n_current_delayed -= ring_pointer[cur_t]
        ring_pointer[cur_t] = 0
        agents = event_participants[dst_event]
        cur_resolves = delay_resolve_time[agents]
        delay_ids = np.where(cur_resolves != -1)[0]
        if len(delay_ids):
            cur_delays = (cur_resolves[delay_ids]-cur_t + (max_delay+1)) % (max_delay+1)
            current_delay = np.max(cur_delays)
            for agent_id in agents[delay_ids]:
                remove_delay(agent_id)
            new_time_slot = (cur_t + current_delay) % (max_delay+1)
            for agent_id in agents:
                set_delay(agent_id, new_time_slot)
        else:
            current_delay = 0
            new_time_slot = cur_t

        if np.random.rand() < p_delay:
            delta_delay = round(np.random.exponential(scale=tau))
            new_delay = min(max_delay, delta_delay+current_delay)
            new_time_slot = (cur_t+new_delay) % (max_delay+1)
            if current_delay:
                for agent_id in agents:
                    remove_delay(agent_id)
            if new_delay:
                for agent_id in agents:
                    set_delay(agent_id, time_slot=new_time_slot)
        cur_t = (cur_t+1) % (max_delay+1)
        current_total_delay -= n_current_delayed
        total_delay[dst_event-start] = current_total_delay/n_agents/n_agents
#         check_delay()
    return total_delay
