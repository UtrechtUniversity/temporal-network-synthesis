from pathlib import Path
from pandas import DataFrame
import pickle
from scipy.sparse import save_npz

def write_network(data_dir, A_sparse, event_types, event_participants):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    adjacency_fp = Path(data_dir, "adjacency.npz")
    agent_fp = Path(data_dir, "agents.csv")
    event_type_fp = Path(data_dir, "event_type.pkl")

    save_npz(adjacency_fp, A_sparse)
    with open(event_type_fp, "wb") as f:
        pickle.dump(event_types, f)
    df = DataFrame()
    for agent in range(event_participants.shape[1]):
        df[f"agent{agent}"] = event_participants[:, agent]
    df["time"] = np.arange(event_participants.shape[0], dtype=int)
    df.to_csv(agent_fp)
