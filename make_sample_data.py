import numpy as np
import glob

# ── adjust these paths to your local setup ──
CHUNK_DIR   = "data/traj_data"
TRAJ_DIR    = "data/dataset3/trajectories"
OUTPUT_DIR  = "data/sample"

SAMPLES_PER_CHUNK = 30000   # 30k × 3 chunks = 90k total
TRAJ_COUNT        = 10      # how many raw trajectory files to copy over

# ─────────────────────────────────────────────────────────────────────────────

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. sample from chunks → one small training file ──
chunk_files = sorted(glob.glob(f"{CHUNK_DIR}/*.npz"))
print(f"chunks found: {len(chunk_files)}")

all_X, all_y = [], []

for path in chunk_files:
    d   = np.load(path)
    idx = np.random.choice(len(d["X"]), min(SAMPLES_PER_CHUNK, len(d["X"])), replace=False)
    all_X.append(d["X"][idx])
    all_y.append(d["y"][idx])
    print(f"  sampled {len(idx)} from {os.path.basename(path)}")

X = np.concatenate(all_X)
y = np.concatenate(all_y)

out = f"{OUTPUT_DIR}/train_sample.npz"
np.savez_compressed(out, X=X, y=y)

size_mb = os.path.getsize(out) / 1e6
print(f"\ntraining file saved: {out}")
print(f"shape: X={X.shape}  y={y.shape}")
print(f"size:  {size_mb:.1f} MB")

# ── 2. copy a few raw trajectory files for eval ──
traj_files = sorted(glob.glob(f"{TRAJ_DIR}/*.npz"))[:TRAJ_COUNT]
print(f"\ncopying {len(traj_files)} trajectory files for eval...")

import shutil
for path in traj_files:
    dst = f"{OUTPUT_DIR}/{os.path.basename(path)}"
    shutil.copy(path, dst)
    print(f"  {os.path.basename(path)}")

print("\ndone. upload the entire sample/ folder to Drive.")
