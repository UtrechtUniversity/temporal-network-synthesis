import numba as nb
from numba import njit
from numba import jit
import numpy as np

from synet.process.base import BaseProcess


class DiseaseProcess(BaseProcess):
    "Disease like process."
    name = "disease"

    def __init__(self, disease_time=2, disease_dt=1, p_infected=0.2):
        """Initialize disease like process.

        Arguments
        ---------
        disease_time: float
            Average time for agents to recover.
        disease_dt: float
            Difference between minimum and maximum recovery time.
        p_infected: float
            Base infection probability with one infected agent in an event.
        """
        self.disease_time = disease_time
        self.disease_dt = disease_dt
        self.p_infected = p_infected

    def _simulate(self, net, start=0, end=None, seed=None):
        "Simulate the disease process."
        np.random.seed(seed)
        results = _simulate_disease(
            net.participants, net.n_agents, start, end,
            disease_time=self.disease_time, disease_dt=self.disease_dt,
            p_infected=self.p_infected)
        return results

    def todict(self):
        return {
            "disease_time": self.disease_time,
            "disease_dt": self.disease_dt,
            "p_infected": self.p_infected,
        }


@njit
def _simulate_disease(participants, n_agents, start, end,
                      disease_time=2, disease_dt=1, p_infected=0.2):
    "Simulate the disease process."
    n_zeros = n_agents//2
    n_ones = n_agents - n_zeros
    cur_agents_state = np.zeros(n_agents, dtype=nb.int32)
    cur_agents_state[:n_ones] = 1
    np.random.shuffle(cur_agents_state)
    n_infected = np.zeros(end-start, dtype=nb.int32)

    # Initialize buffers and constants.
    base_disease_steps = int(n_agents*(disease_time - 0.5*disease_dt)+0.5)
    dt_disease_steps = int(n_agents*disease_dt+0.5)
    max_disease_steps = base_disease_steps + dt_disease_steps
    ring_buffer = np.zeros((max_disease_steps+1, n_agents), dtype=nb.int32)
    ring_pointer = np.zeros(max_disease_steps+1, dtype=nb.int32)

    # Infect half of the agents.
    for agent_id in np.where(cur_agents_state == 1)[0]:
        t = np.random.randint(max_disease_steps+1)
        ring_buffer[t][ring_pointer[t]] = agent_id
        ring_pointer[t] += 1

    cur_t = 0
    cur_infected = n_ones
    for dst_event in range(start, end):
        # Infect agents through our current event.
        n_event_infected = 0
        for agent_id in participants[dst_event]:
            n_event_infected += cur_agents_state[agent_id]
        cur_p_infection = 1-(1-p_infected)**n_event_infected
        for agent_id in participants[dst_event]:
            if (cur_agents_state[agent_id] != 1
                    and np.random.rand() < cur_p_infection):
                cur_agents_state[agent_id] = 1
                cure_time = cur_t + base_disease_steps + np.random.randint(
                    dt_disease_steps)
                cure_time = (cure_time) % (max_disease_steps+1)
                ring_buffer[cure_time][ring_pointer[cure_time]] = agent_id
                ring_pointer[cure_time] += 1
                cur_infected += 1

        # Remove agents from the count that get better.
        for agent_id in ring_buffer[cur_t][:ring_pointer[cur_t]]:
            cur_agents_state[agent_id] = 0
        cur_infected -= ring_pointer[cur_t]
        ring_pointer[cur_t] = 0
        n_infected[dst_event-start] = cur_infected
        if cur_infected == 0:
            break
        cur_t = (cur_t+1) % (max_disease_steps+1)
    return n_infected


@jit
def _simulate_disease_start(event_participants, start, end, n_agents=-1,
                            disease_time=2, disease_dt=1, p_infected=0.2):
    # Currently unused.
    if n_agents == -1:
        n_agents = np.max(event_participants)+1
    cur_agents_state = np.zeros(n_agents, dtype=nb.int32)
    n_infected = np.zeros(end-start, dtype=nb.int32)

    base_disease_steps = int(n_agents*(disease_time - 0.5*disease_dt)+0.5)
    dt_disease_steps = int(n_agents*disease_dt+0.5)
    max_disease_steps = base_disease_steps + dt_disease_steps
    ring_buffer = np.zeros((max_disease_steps+1, n_agents), dtype=nb.int32)
    ring_pointer = np.zeros(max_disease_steps+1, dtype=nb.int32)

    start_agent = np.random.choice(event_participants[start])
    t = base_disease_steps + np.random.randint(dt_disease_steps)
    ring_buffer[t][0] = start_agent
    ring_pointer[t] = 1
    cur_agents_state[start_agent] = 1

    cur_t = 0
    cur_infected = 1
    for dst_event in range(start, end):
        n_event_infected = 0
        for agent_id in event_participants[dst_event]:
            n_event_infected += cur_agents_state[agent_id]
        cur_p_infection = 1-(1-p_infected)**n_event_infected
        for agent_id in event_participants[dst_event]:
            if (cur_agents_state[agent_id] != 1
                    and np.random.rand() < cur_p_infection):
                cur_agents_state[agent_id] = 1
                cure_time = cur_t + base_disease_steps + np.random.randint(
                    dt_disease_steps)
                cure_time = (cure_time) % (max_disease_steps+1)
                ring_buffer[cure_time][ring_pointer[cure_time]] = agent_id
                ring_pointer[cure_time] += 1
                cur_infected += 1

        for agent_id in ring_buffer[cur_t][:ring_pointer[cur_t]]:
            cur_agents_state[agent_id] = 0
        cur_infected -= ring_pointer[cur_t]
        ring_pointer[cur_t] = 0
        n_infected[dst_event-start] = cur_infected
        if cur_infected == 0:
            break
        cur_t = (cur_t+1) % (max_disease_steps+1)
    return n_infected
