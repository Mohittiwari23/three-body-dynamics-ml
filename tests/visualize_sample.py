"""
visualize_sample.py
====================
Visualise any sample from the Phase 3 three-body dataset.

Usage
-----
  # By CSV row position (0-indexed)
  python visualize_sample.py --row 42

  # By outcome class
  python visualize_sample.py --outcome stable
  python visualize_sample.py --outcome ejection
  python visualize_sample.py --outcome chaotic

  # By regime
  python visualize_sample.py --regime hierarchical
  python visualize_sample.py --regime compact_equal --outcome ejection

  # Random sample (any outcome)
  python visualize_sample.py --random

  # Save to file instead of showing
  python visualize_sample.py --row 42 --save output.mp4

  # List available samples
  python visualize_sample.py --list

Defaults
--------
  csv_path  : data/dataset3/metadata3.csv  (relative to this script)
  traj_dir  : data/dataset3/trajectories/
"""

from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

# ── path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent         # one level up from three_body/
sys.path.insert(0, str(PROJECT_DIR))

from three_body.visualize3 import animate_3body

DEFAULT_CSV  = PROJECT_DIR / "data" / "dataset3" / "metadata3.csv"
DEFAULT_TRAJ = PROJECT_DIR / "data" / "dataset3" / "trajectories"

OUTCOME_COLOURS = {
    "stable":   "\033[92m",   # green
    "ejection": "\033[91m",   # red
    "chaotic":  "\033[93m",   # yellow
    "collision":"\033[95m",   # magenta
}
RESET = "\033[0m"


# ── result wrapper ──────────────────────────────────────────────────────────

class SimResult:
    """
    Minimal wrapper around a loaded .npz trajectory file.
    animate_3body requires: traj, time, E_hist, L_hist, E0, L0, MEGNO_final
    """
    def __init__(self, npz, megno_final: float):
        self.traj        = npz["traj"].astype(np.float64)   # (T, 3, 2)
        self.E_hist      = npz["E_hist"].astype(np.float64) # (T,)
        self.L_hist      = npz["L_hist"].astype(np.float64) # (T,)
        self.time        = npz["time"].astype(np.float64)   # (T,)
        self.E0          = float(self.E_hist[0])
        self.L0          = float(self.L_hist[0])
        self.MEGNO_final = megno_final
        self.vel_hist    = None   # not stored; animate_3body falls back to gradient


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found at {csv_path}\n"
                 f"Run dataset_generator3.py first to generate the dataset.")
    return pd.read_csv(csv_path)


def _traj_path(traj_dir: Path, idx: int) -> Path:
    """
    Trajectory files are named by simulation IDX, not by CSV row position.
    There are gaps in IDX because some simulations were rejected by the
    quality gate (dL_max > 0.02 or dE_max > 0.1). Always use the idx
    column from the CSV, not the row number, to find the file.
    """
    return traj_dir / f"traj_{idx:05d}.npz"


def _check_traj(path: Path) -> bool:
    if not path.exists():
        print(f"WARNING: trajectory file not found: {path}")
        print("         The dataset CSV and trajectories/ folder must be in the same directory.")
        print("         Re-run dataset_generator3.py with the same seed to regenerate.")
        return False
    return True


def _print_sample_info(row: pd.Series, csv_row_pos: int):
    outcome = row["outcome"]
    colour  = OUTCOME_COLOURS.get(outcome, "")
    print()
    print("─"*52)
    print(f"  Sample    : CSV row {csv_row_pos}  (simulation idx {int(row['idx'])})")
    print(f"  Regime    : {row['regime']}")
    print(f"  Outcome   : {colour}{outcome.upper()}{RESET}  "
          f"(class {int(row['outcome_class'])})")
    print()
    print(f"  Initial conditions:")
    print(f"    epsilon_total = {row['epsilon_total']:.4f}  (specific energy, all < 0)")
    print(f"    h_total       = {row['h_total']:.4f}  (specific ang. momentum)")
    print(f"    q12           = {row['q12']:.4f}  (inner mass ratio, ≤ 1)")
    print(f"    v3_frac       = {row['v3_frac']:.4f}  (outer velocity / v_circ)")
    print(f"    e_inner       = {row['e_inner']:.4f}  (inner binary eccentricity)")
    print(f"    r12_init      = {row['r12_init']:.4f}  (inner binary periapsis)")
    print(f"    r3_sep        = {row['r3_sep']:.4f}  (outer body distance)")
    print(f"    m1={row['m1']:.2f}  m2={row['m2']:.2f}  m3={row['m3']:.2f}")
    print()
    print(f"  Labelling metadata (NOT ML features):")
    print(f"    MEGNO         = {row['MEGNO']:.3f}  (> 3 → chaotic label)")
    print(f"    dE_max_w20    = {row['dE_max_w20']:.2e}  (energy conservation quality)")
    print(f"    e12_std_w20   = {row['e12_std_w20']:.4f}  (inner binary perturbation)")
    print("─"*52)
    print()


def _select_row(df: pd.DataFrame, args) -> tuple[pd.Series, int]:
    """
    Returns (row, csv_row_position).
    Selection priority: --row > --outcome+--regime filter > --random.
    """
    if args.row is not None:
        if args.row >= len(df) or args.row < 0:
            sys.exit(f"ERROR: --row {args.row} out of range (0 to {len(df)-1})")
        return df.iloc[args.row], args.row

    # Build filter mask
    mask = pd.Series([True] * len(df), index=df.index)
    if args.outcome:
        valid = ["stable","ejection","chaotic","collision"]
        if args.outcome not in valid:
            sys.exit(f"ERROR: --outcome must be one of {valid}")
        mask &= df["outcome"] == args.outcome
    if args.regime:
        valid_r = df["regime"].unique().tolist()
        if args.regime not in valid_r:
            sys.exit(f"ERROR: --regime must be one of {valid_r}")
        mask &= df["regime"] == args.regime

    candidates = df[mask]
    if len(candidates) == 0:
        sys.exit(f"ERROR: no samples match the filter "
                 f"(outcome={args.outcome}, regime={args.regime})")

    chosen = candidates.sample(1, random_state=None).iloc[0]
    csv_pos = df.index.get_loc(chosen.name)
    return chosen, int(csv_pos)


def _list_samples(df: pd.DataFrame):
    print(f"\nDataset: {len(df)} samples\n")
    print(f"{'Row':>5}  {'idx':>5}  {'Regime':>15}  {'Outcome':>12}  "
          f"{'MEGNO':>7}  {'v3_frac':>7}  {'q12':>6}")
    print("─"*68)
    for i, (_, row) in enumerate(df.iterrows()):
        colour  = OUTCOME_COLOURS.get(row['outcome'], '')
        print(f"{i:5d}  {int(row['idx']):5d}  {row['regime']:>15}  "
              f"{colour}{row['outcome']:>12}{RESET}  "
              f"{row['MEGNO']:7.2f}  {row['v3_frac']:7.4f}  {row['q12']:6.4f}")
        if i >= 49:
            print(f"  ... (showing first 50 of {len(df)})")
            print(f"  Use --outcome / --regime to filter, or --row N for a specific sample")
            break
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise a sample from the Phase 3 three-body dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",     type=Path, default=DEFAULT_CSV,
                        help=f"Path to metadata3.csv (default: {DEFAULT_CSV})")
    parser.add_argument("--traj_dir",type=Path, default=DEFAULT_TRAJ,
                        help=f"Path to trajectories/ folder (default: {DEFAULT_TRAJ})")
    parser.add_argument("--row",     type=int,  default=None,
                        help="CSV row index (0-based)")
    parser.add_argument("--outcome", type=str,  default=None,
                        choices=["stable","ejection","chaotic","collision"],
                        help="Filter by outcome label")
    parser.add_argument("--regime",  type=str,  default=None,
                        choices=["hierarchical","asymmetric","compact_equal","scatter"],
                        help="Filter by sampling regime")
    parser.add_argument("--random",  action="store_true",
                        help="Pick a random sample (no filter)")
    parser.add_argument("--save",    type=str,  default=None,
                        help="Save animation to this .mp4 path instead of showing")
    parser.add_argument("--list",    action="store_true",
                        help="List available samples (first 50)")
    parser.add_argument("--stride",  type=int,  default=10,
                        help="Animation frame stride (higher = faster, default 10)")
    parser.add_argument("--trail",   type=int,  default=400,
                        help="Trail length in frames (lower = faster, default 400)")
    args = parser.parse_args()

    # ── load dataset ──────────────────────────────────────────────────────
    df = _load_csv(args.csv)

    if args.list:
        _list_samples(df)
        return

    # ── select sample ─────────────────────────────────────────────────────
    row, csv_row_pos = _select_row(df, args)
    _print_sample_info(row, csv_row_pos)

    # ── load trajectory ───────────────────────────────────────────────────
    # IMPORTANT: use row['idx'], NOT csv_row_pos.
    # The idx column is the simulation index; trajectories have gaps because
    # some simulations were rejected by the quality gate and not written to CSV.
    traj_file = _traj_path(args.traj_dir, int(row["idx"]))
    if not _check_traj(traj_file):
        sys.exit(1)

    npz    = np.load(traj_file)
    result = SimResult(npz, megno_final=float(row["MEGNO"]))

    nsteps = result.traj.shape[0]
    tspan  = result.time[-1]
    print(f"  Trajectory: {nsteps:,} steps  |  t_total = {tspan:.2f}")
    print(f"  Shape:      {result.traj.shape}  (steps × bodies × 2D coords)")
    print()

    # ── animate ───────────────────────────────────────────────────────────
    save_path = args.save if args.save else None
    animate_3body(
        result,
        stride    = args.stride,
        trail_len = args.trail,
        save_path = save_path,
    )


if __name__ == "__main__":
    main()