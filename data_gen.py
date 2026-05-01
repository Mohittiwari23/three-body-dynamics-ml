import numpy as np
import os
from glob import glob

# ---------- CONFIG ----------
WINDOW_SIZE = 10
STRIDE = 10

DATA_PATH = "data/trajectories/trajectories"
SAVE_PATH = "data/traj_data"

os.makedirs(SAVE_PATH, exist_ok=True)


# ---------- GET NEXT CHUNK ID ----------
def get_next_chunk_id(save_path):
    existing = glob(os.path.join(save_path, "*.npz"))
    if not existing:
        return 0

    ids = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in existing]
    return max(ids) + 1


# ---------- PROCESS ONE FILE ----------
def process_file(path):
    data = np.load(path)

    traj = data["traj"]
    time = data["time"]

    # ---------- VALIDATION ----------
    if traj.ndim != 3 or traj.shape[1:] != (3, 2):
        print(f"Invalid traj shape: {path}")
        return None

    if len(traj) < 2:
        print(f"Too short (raw): {path}")
        return None

    # ---------- SUBSAMPLING ----------
    traj = traj[::STRIDE]
    time = time[::STRIDE]

    if len(traj) < 2:
        print(f"Too short after stride: {path}")
        return None

    # ---------- TIME CHECK ----------
    dt_array = np.diff(time)

    if len(dt_array) == 0 or np.any(dt_array <= 0):
        print(f"Invalid time data: {path}")
        return None

    dt = np.mean(dt_array)

    if not np.isfinite(dt):
        print(f"Invalid dt: {path}")
        return None

    # ---------- VELOCITY ----------
    vel = (traj[1:] - traj[:-1]) / dt
    pos = traj[:-1]

    if len(pos) == 0:
        print(f"No valid position after processing: {path}")
        return None

    # ---------- COMBINE ----------
    state = np.concatenate([pos, vel], axis=-1)

    if state.size == 0:
        print(f"Empty state: {path}")
        return None

    state = state.reshape(state.shape[0], -1)

    return state

# ---------- CREATE SEQUENCES ----------
def create_sequences(state):
    X, y = [], []

    for i in range(len(state) - WINDOW_SIZE):
        X.append(state[i:i + WINDOW_SIZE])
        y.append(state[i + WINDOW_SIZE] - state[i + WINDOW_SIZE - 1])

    return np.array(X), np.array(y)


# ---------- MAIN ----------
files = glob(os.path.join(DATA_PATH, "*.npz"))

if len(files) == 0:
    raise ValueError("No files found in DATA_PATH")

all_X, all_y = [], []

for file in files:
    state = process_file(file)

    if state is None:
        continue

    if len(state) <= WINDOW_SIZE:
        continue

    X, y = create_sequences(state)

    if len(X) == 0:
        continue

    all_X.append(X)
    all_y.append(y)

X_chunk = np.concatenate(all_X, axis=0)
y_chunk = np.concatenate(all_y, axis=0)

chunk_id = get_next_chunk_id(SAVE_PATH)

save_file = os.path.join(SAVE_PATH, f"chunk_{chunk_id}.npz")

np.savez(save_file, X=X_chunk, y=y_chunk)

print(f"Saved {save_file}")
print("Shape:", X_chunk.shape)