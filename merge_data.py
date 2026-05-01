import numpy as np
import glob

CHUNK_PATH = "data/traj_data/*.npz"

chunk_files = sorted(glob.glob(CHUNK_PATH))

all_X, all_y = [], []

for file in chunk_files:
    data = np.load(file)

    X = data["X"]
    y = data["y"]

    print(f"{file}: X={X.shape}, y={y.shape}")

    # sanity check
    if X.ndim != 3 or y.ndim != 2:
        print(f"Skipping invalid chunk: {file}")
        continue

    all_X.append(X)
    all_y.append(y)

# ---------- MERGE ----------
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

print("\nFinal merged dataset:")
print("X:", X.shape)
print("y:", y.shape)

mean = X.mean(axis=(0,1), keepdims=True)
std = X.std(axis=(0,1), keepdims=True) + 1e-8

X = (X - mean) / std
y = y / std.squeeze(0)

print("Normalization complete.")

np.savez("final_dataset.npz", X=X, y=y, mean=mean, std=std)