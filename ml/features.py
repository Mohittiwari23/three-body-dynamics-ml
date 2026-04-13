"""
ml/features.py
==============
Feature group definitions and extraction utilities.

Feature groups
--------------
  PHYSICS_NORM   — normalised, rotation-invariant (ε, h, μ, q, r₀)
  PHYSICS_RAW    — raw (E0, L0, μ, r0) without normalisation
  DYNAMICS       — conservation diagnostics + MEGNO_clean
  ALL_PHYSICS    — PHYSICS_NORM + DYNAMICS
  ORBITAL_ELEM   — (e, p, a, r_min, r_ratio) — Pinheiro 2025 baseline
  TRAJECTORY     — scalar summaries from trajectory

Quality filter
--------------
  Use dL_max < 1e-6 (NOT dE_max).
  dE_max explodes for parabolic orbits (E₀ ≈ 0), destroying 97% of that
  class. Angular momentum is conserved to machine precision for all orbit
  types — dL_max is reliable across circular/elliptical/parabolic/hyperbolic.

MEGNO_clean
-----------
  Raw MEGNO can be negative for high-eccentricity orbits that haven't
  converged in 3 periods (63/2000 samples). Clip to [0, 10].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Column group definitions
# ---------------------------------------------------------------------------

PHYSICS_NORM = ["epsilon", "h", "mu", "q", "r0"]
PHYSICS_RAW  = ["E0", "L0", "mu", "r0"]
DYNAMICS     = ["dE_max", "dE_slope", "dL_max", "MEGNO_clean", "e_inst_std"]
TRAJECTORY   = ["traj_r_mean", "traj_r_std",
                "traj_r_min_num", "traj_r_max_num",
                "traj_vr_max", "traj_vt_mean"]
ORBITAL_ELEM = ["e", "p", "a", "r_min", "r_ratio"]
ALL_PHYSICS  = PHYSICS_NORM + DYNAMICS
RAW_INPUTS   = ["alpha", "v_perp", "omega", "M_total", "q", "r0"]

REGRESSION_TARGET     = "e"
CLASSIFICATION_TARGET = "orbit_class"
CLASS_NAMES           = ["circular", "elliptical", "parabolic", "hyperbolic"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract(df: pd.DataFrame, feature_set: list[str]) -> np.ndarray:
    missing = [f for f in feature_set if f not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return df[feature_set].values.astype(np.float64)


def extract_regression_target(df: pd.DataFrame) -> np.ndarray:
    return df[REGRESSION_TARGET].values.astype(np.float64)


def extract_classification_target(df: pd.DataFrame) -> np.ndarray:
    return df[CLASSIFICATION_TARGET].values.astype(np.int64)


# ---------------------------------------------------------------------------
# MEGNO cleaning
# ---------------------------------------------------------------------------

def add_megno_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MEGNO_clean = clip(MEGNO, 0, 10). Handles negatives and NaN.
    Negative MEGNO is a convergence artifact for high-e ellipticals with
    short integration windows, not a physical signal.
    """
    df = df.copy()
    if "MEGNO" in df.columns:
        df["MEGNO_clean"] = df["MEGNO"].clip(lower=0.0, upper=10.0).fillna(2.0)
    else:
        df["MEGNO_clean"] = 2.0
    return df


# ---------------------------------------------------------------------------
# Quality filter — use dL_max not dE_max
# ---------------------------------------------------------------------------

def quality_filter(
    df: pd.DataFrame,
    dL_max_threshold: float = 1e-6,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter by angular momentum conservation: dL_max ≤ threshold.

    WHY dL_max AND NOT dE_max:
    - Parabolic orbits have E₀ ≈ 0, so ΔE/|E₀| explodes numerically
      even when integration is perfect. Using dE_max destroys 97% of the
      parabolic class, leaving only 4 samples after filtering.
    - dL_max is conserved to machine precision for ALL orbit types by the
      Velocity Verlet integrator (central force symmetry).
    """
    n_before = len(df)
    df_out = df[df["dL_max"] <= dL_max_threshold].reset_index(drop=True)
    if verbose:
        dropped = n_before - len(df_out)
        if dropped:
            print(f"  Quality filter (dL_max ≤ {dL_max_threshold:.0e}): "
                  f"dropped {dropped}, kept {len(df_out)}")
    return df_out


# ---------------------------------------------------------------------------
# Mass ratio generalisation split
# ---------------------------------------------------------------------------

def train_test_generalisation_split(
    df: pd.DataFrame,
    q_train_max: float = 0.15,
    q_ood_min: float   = 0.35,
    test_frac: float   = 0.20,
    seed: int          = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Three-way split for mass ratio generalisation experiment.

    Training pool : q ≤ q_train_max  (asymmetric mass, star-planet regime)
    In-dist test  : held-out 20% of training pool
    OOD test      : q ≥ q_ood_min   (near-equal mass, binary star regime)
    """
    pool = df[df["q"] <= q_train_max].copy()
    ood  = df[df["q"] >= q_ood_min].copy()
    pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test     = int(len(pool) * test_frac)
    df_test_id = pool.iloc[:n_test]
    df_train   = pool.iloc[n_test:]
    return df_train, df_test_id, ood


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(
    csv_path: str | Path,
    apply_quality_filter: bool = True,
    dL_max_threshold: float    = 1e-6,
    drop_nan_features: bool    = True,
    verbose: bool              = True,
) -> pd.DataFrame:
    """
    Load metadata.csv, apply dL_max quality filter, add MEGNO_clean.
    """
    df = pd.read_csv(csv_path)

    if apply_quality_filter:
        df = quality_filter(df, dL_max_threshold=dL_max_threshold,
                            verbose=verbose)

    df = add_megno_clean(df)

    if drop_nan_features:
        key_cols = PHYSICS_NORM + ["e", "orbit_class"]
        before = len(df)
        df = df.dropna(subset=key_cols).reset_index(drop=True)
        if verbose and len(df) < before:
            print(f"  NaN filter: dropped {before - len(df)} rows")

    if verbose:
        print(f"  Loaded {len(df)} samples  |  "
              + "  ".join(f"{k}:{v}" for k, v in
                          df["orbit_name"].value_counts().items()))
    return df


# ---------------------------------------------------------------------------
# Trajectory window loader
# ---------------------------------------------------------------------------

def load_trajectory_window(
    npz_path: str | Path, fraction: float = 0.20
) -> np.ndarray:
    """Load first `fraction` of trajectory from .npz file. Returns (K, 2)."""
    data = np.load(npz_path)
    traj = data["traj"]
    K    = max(1, int(traj.shape[0] * fraction))
    return traj[:K]


# ---------------------------------------------------------------------------
# Feature descriptions (for paper)
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "epsilon":        "Specific energy E₀/μ  [mass-normalised]",
    "h":              "Specific angular momentum L₀/μ",
    "mu":             "Reduced mass μ = m₁m₂/(m₁+m₂)",
    "q":              "Mass ratio m₁/m₂",
    "r0":             "Initial separation",
    "E0":             "Total mechanical energy",
    "L0":             "Angular momentum",
    "dE_max":         "Max |ΔE/KE₀| — energy drift (proxy for eccentricity)",
    "dE_slope":       "Linear slope of ΔE(t) — secular drift",
    "dL_max":         "Max |ΔL/L₀| — momentum drift (quality metric)",
    "MEGNO_clean":    "<Y>(T) clipped [0,10] — chaos indicator (→2 regular)",
    "e_inst_std":     "Std of instantaneous e(t) — eccentricity noise signal",
    "traj_r_mean":    "Mean radial distance along trajectory",
    "traj_r_std":     "Std of radial distance",
    "traj_r_min_num": "Minimum r observed numerically",
    "traj_r_max_num": "Maximum r observed numerically",
    "traj_vr_max":    "Maximum radial velocity component",
    "traj_vt_mean":   "Mean tangential velocity component",
}