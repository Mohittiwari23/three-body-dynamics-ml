"""
ml_predictor/features.py
==============

Physics-informed feature engineering pipeline for three-body instability
prediction using early-time gravitational dynamics.

──────────────────────────────────────────────────────────────────────────────
DESIGN PHILOSOPHY
──────────────────────────────────────────────────────────────────────────────

The dataset encodes trajectories of chaotic three-body gravitational systems.
The goal is to predict long-term outcomes using *restricted early-time data*.

Feature information is partitioned into three causally distinct tiers:

Tier 1 — Initial Conditions (t = 0)
    Pure system state before any dynamical evolution. Encodes geometry,
    energy, angular momentum, and mass hierarchy. Causally valid unconditionally.

Tier 2 — Early Window Dynamics (w5 / w20)
    Computed strictly within early trajectory windows. Encodes incipient
    instability signatures: energy/angular-momentum drift, eccentricity
    fluctuations, closest approaches. Window suffix (_w5, _w20) is
    mandatory — conflating windows violates experimental isolation.

Tier 3 — Engineered Features
    Nonlinear combinations of Tier 1 + Tier 2. Encode known dynamical
    stability laws (Hill criterion, escape velocity proximity, encounter
    strength scaling). Each engineered feature is window-scoped when it
    depends on dynamical quantities — critical for ablation validity.

──────────────────────────────────────────────────────────────────────────────
FEATURE GROUP EXPERIMENTS (A–F)
──────────────────────────────────────────────────────────────────────────────

  A  →  IC only                              (pure geometry/invariants)
  B  →  IC + w5 + w5_engineered             (early-window prediction)
  C  →  IC + w20 + w20_engineered           (medium-window prediction)
  D  →  w5 + w5_engineered only             (dynamics alone, short window)
  E  →  w20 + w20_engineered only           (dynamics alone, medium window)
  F  →  IC + w5 + w20 + w20_engineered      (full model, all information)

Design rationale:
  - Groups A/B/C isolate information gain from adding dynamics to IC.
  - Groups D/E test whether IC is necessary or if dynamics alone suffice.
  - Groups B vs C quantify the predictability gain from 5% → 20% window.
  - Group F is the production model; uses w20 engineered (stronger signal)
    rather than duplicating all engineered features for both windows.

──────────────────────────────────────────────────────────────────────────────
LEAKAGE POLICY (ENFORCED)
──────────────────────────────────────────────────────────────────────────────

NEVER include:
  - outcome labels (outcome, outcome_class, outcome_class4)
  - MEGNO / MEGNO_clean (requires full trajectory)
  - post-hoc computed quantities
  - dataset metadata (traj_file, idx)
  - window-B features in a window-A experiment group

Engineered features are window-scoped: w5 engineered uses ONLY w5 dynamics,
w20 engineered uses ONLY w20 dynamics. Cross-window contamination is
explicitly blocked at the group-selection level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# =============================================================================
# LEAKAGE CONTROL LISTS
# =============================================================================

LABEL_COLS: list[str] = [
    "outcome",
    "outcome_class",
    "outcome_class4",
]

META_COLS: list[str] = [
    "idx",
    "traj_file",
    "regime",
]

LEAKY_COLS: list[str] = [
    "MEGNO",
    "MEGNO_clean",
]

# =============================================================================
# TIER 1: INITIAL CONDITION FEATURES  (t = 0, no dynamical evolution)
# =============================================================================

IC_FEATURES: list[str] = [
    # Global invariants — conserved by Newtonian gravity, computed at t=0
    "epsilon_total",     # total specific energy; sets overall binding regime
    "h_total",           # total angular momentum magnitude

    # Mass hierarchy — determines Hill stability threshold scaling
    "q12",               # m1/m2 inner binary mass ratio
    "q13",               # m1/m3
    "q23",               # m2/m3

    # Spatial configuration — initial orbital geometry
    "r12_init",          # inner binary separation
    "r3_sep",            # outer body separation from barycentre

    # Kinematic state — outer body velocity relative to local circular speed
    "v3_frac",           # |v3| / v_circular; < 1 → sub-circular, > √2 → unbound
    "v3_angle",          # direction of v3 ∈ [0, 2π); replaced by sin/cos in ENG
]

# =============================================================================
# TIER 2: EARLY WINDOW DYNAMICS (w5 = first 5% of inner orbital period)
# =============================================================================

W5_FEATURES: list[str] = [
    "dE_max_w5",         # max fractional energy violation in [0, T_w5]
    "dL_max_w5",         # max angular momentum drift in [0, T_w5]

    "e12_std_w5",        # eccentricity fluctuation of inner binary
    "e13_std_w5",        # eccentricity fluctuation of outer pair (1-3)
    "e23_std_w5",        # eccentricity fluctuation of outer pair (2-3)

    "r_min_12_w5",       # minimum separation of inner binary in window
    "r_min_13_w5",       # minimum separation 1-3 in window
    "r_min_23_w5",       # minimum separation 2-3 in window
]

# =============================================================================
# TIER 2: EARLY WINDOW DYNAMICS (w20 = first 20% of inner orbital period)
# =============================================================================

W20_FEATURES: list[str] = [
    "dE_max_w20",
    "dL_max_w20",

    "e12_std_w20",
    "e13_std_w20",
    "e23_std_w20",

    "r_min_12_w20",
    "r_min_13_w20",
    "r_min_23_w20",
]

# =============================================================================
# TIER 3: ENGINEERED FEATURES (window-scoped)
# =============================================================================

# Structural features derived from IC only (window-independent)
_IC_ENGINEERED: list[str] = [
    "hill_ratio",           # Mardling-Aarseth stability proxy
    "r_separation_ratio",   # compactness of the triple
    "energy_partition",     # binding at outer orbit scale
    "v3_margin",            # signed distance from escape velocity
    "v3_sin",               # circular encoding of v3_angle
    "v3_cos",               # circular encoding of v3_angle
]

# Dynamical features engineered from w5 dynamics
_W5_ENGINEERED: list[str] = [
    "close_encounter_strength_w5",  # orbital perturbation per unit closest approach
    "dL_dE_coupling_w5",            # simultaneous conservation law violation
]

# Dynamical features engineered from w20 dynamics
_W20_ENGINEERED: list[str] = [
    "close_encounter_strength_w20",
    "dL_dE_coupling_w20",
]

# Convenience aggregates for external use (visualiser, evaluator)
IC_ENGINEERED_FEATURES: list[str] = _IC_ENGINEERED
W5_ENGINEERED_FEATURES: list[str] = _W5_ENGINEERED
W20_ENGINEERED_FEATURES: list[str] = _W20_ENGINEERED
ALL_ENGINEERED_FEATURES: list[str] = _IC_ENGINEERED + _W5_ENGINEERED + _W20_ENGINEERED
DYN_FEATURES: list[str] = W5_FEATURES + W20_FEATURES


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def _engineer_ic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Tier 3 features that depend only on initial conditions.

    These are window-independent: valid for any feature group that includes IC.

    hill_ratio
        Proxy for Mardling-Aarseth (2001) hierarchical stability criterion.
        r_out/r_in × (1 / 3(1 + q12))^(1/3).
        This is the ratio of the outer semi-major axis to the Hill sphere
        radius of the inner binary. Values ≲ 1 are unconditionally unstable.

    r_separation_ratio
        r12_init / r3_sep — system compactness.
        When → 1, the outer body is at the same spatial scale as the inner
        binary, making strong three-body coupling geometrically inevitable.

    energy_partition
        |epsilon_total| × r3_sep — effective binding energy at the outer orbit
        scale. Small values → outer body loosely bound → low energy exchange
        required for ejection.

    v3_margin
        √2 − v3_frac — signed distance from escape velocity.
        Positive → bound; zero → marginally bound; negative → already escaping.

    v3_sin, v3_cos
        Circular encoding of v3_angle ∈ [0, 2π).
        Tree models cannot bridge the 0/2π wrap discontinuity in raw angle.
        Physical note: v3_cos ≈ −1 (retrograde) is empirically more stable
        due to reduced resonance overlap with the inner binary.
    """
    df["hill_ratio"] = (df["r3_sep"] / df["r12_init"]) * (
        1.0 / (3.0 * (1.0 + df["q12"])) ** (1.0 / 3.0)
    )
    df["r_separation_ratio"] = df["r12_init"] / df["r3_sep"]
    df["energy_partition"]   = np.abs(df["epsilon_total"]) * df["r3_sep"]
    df["v3_margin"]          = np.sqrt(2.0) - df["v3_frac"]
    df["v3_sin"]             = np.sin(df["v3_angle"])
    df["v3_cos"]             = np.cos(df["v3_angle"])
    return df


def _engineer_window_features(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    Compute Tier 3 features that depend on a specific early window.

    Parameters
    ----------
    df     : DataFrame containing raw dynamic features for `window`.
    window : "w5" or "w20" — must match column suffixes in df.

    close_encounter_strength_{window}
        max(e_ij_std) / min(r_min_outer) — instability strength per unit
        closest approach. High value → strong chaotic perturbation in window.
        This is the most physically direct chaos precursor available from
        early-time data.

    dL_dE_coupling_{window}
        dL_max × dE_max — product of angular momentum and energy drift.
        Simultaneous violation of both conservation laws is a stronger
        chaos signal than either alone. A near-circular orbit can have
        moderate dE with near-zero dL; simultaneous violation means the
        orbit geometry is being rapidly distorted.
    """
    assert window in ("w5", "w20"), f"window must be 'w5' or 'w20', got {window!r}"

    dE_col = f"dE_max_{window}"
    dL_col = f"dL_max_{window}"

    if dE_col not in df.columns:
        return df  # window not present in this dataset — skip silently

    rmin = np.minimum(
        df.get(f"r_min_13_{window}", pd.Series(np.inf, index=df.index)),
        df.get(f"r_min_23_{window}", pd.Series(np.inf, index=df.index)),
    )

    emax = np.maximum(
        df.get(f"e12_std_{window}", pd.Series(0.0, index=df.index)),
        np.maximum(
            df.get(f"e13_std_{window}", pd.Series(0.0, index=df.index)),
            df.get(f"e23_std_{window}", pd.Series(0.0, index=df.index)),
        ),
    )

    df[f"close_encounter_strength_{window}"] = emax / (rmin + 1e-6)
    df[f"dL_dE_coupling_{window}"]           = df[dL_col] * df[dE_col]

    return df


def engineer_features(df: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    """
    Compute all Tier 3 engineered features for the requested windows.

    Parameters
    ----------
    df      : DataFrame containing IC + requested window dynamic features.
    windows : Subset of ["w5", "w20"] to engineer. Only windows whose raw
              features are present will be processed; others are skipped.

    Returns
    -------
    df : Copy with engineered columns appended. Original columns unchanged.
    """
    df = df.copy()
    df = _engineer_ic_features(df)
    for w in windows:
        df = _engineer_window_features(df, w)
    return df


# =============================================================================
# FEATURE GROUP SELECTION
# =============================================================================

# Maps group label → (base_feature_lists, windows_to_engineer)
# base_feature_lists : lists of raw feature names to include
# windows_to_engineer: which window engineered features to add
#
# IC_ENGINEERED is always added when IC is included (no window dependency).
# v3_angle is always excluded — replaced by v3_sin/v3_cos in engineered groups,
# kept out of non-engineered groups to avoid the wrap discontinuity issue while
# maintaining strict experimental separation.

_GROUP_SPEC: dict[str, dict] = {
    "A": {
        "bases":   [IC_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC only — pure geometry and invariants",
    },
    "B": {
        "bases":   [IC_FEATURES, W5_FEATURES],
        "windows": ["w5"],
        "ic_eng":  True,
        "desc":    "IC + w5 dynamics + w5 & IC engineered",
    },
    "C": {
        "bases":   [IC_FEATURES, W20_FEATURES],
        "windows": ["w20"],
        "ic_eng":  True,
        "desc":    "IC + w20 dynamics + w20 & IC engineered",
    },
    "D": {
        "bases":   [W5_FEATURES],
        "windows": ["w5"],
        "ic_eng":  False,
        "desc":    "w5 dynamics + w5 engineered only",
    },
    "E": {
        "bases":   [W20_FEATURES],
        "windows": ["w20"],
        "ic_eng":  False,
        "desc":    "w20 dynamics + w20 engineered only",
    },
    "F": {
        "bases":   [IC_FEATURES, W5_FEATURES, W20_FEATURES],
        "windows": ["w20"],   # w20 engineered is the stronger signal; w5 raw is included
        "ic_eng":  True,
        "desc":    "Full model: IC + w5 + w20 + w20 & IC engineered",
    },
}


def get_feature_matrix(df: pd.DataFrame, feature_group: str = "C") -> pd.DataFrame:
    """
    Construct a leakage-safe, window-consistent feature matrix.

    Parameters
    ----------
    df            : Raw dataset (IC + window dynamic columns + labels + meta).
    feature_group : One of A–F. See module docstring for definitions.

    Returns
    -------
    DataFrame — model-ready, numeric only, no label/meta/leaky columns.

    Notes
    -----
    - v3_angle is always excluded. Groups with engineered features include
      v3_sin / v3_cos instead. Groups without engineering omit angle entirely
      to keep experimental groups clean (v3_frac still provides speed info).
    - Engineered features are computed on a copy; original df is not mutated.
    - Missing columns are silently dropped with a warning — this handles
      datasets that only have one window computed.
    """
    feature_group = feature_group.upper()
    if feature_group not in _GROUP_SPEC:
        raise ValueError(
            f"Invalid feature_group {feature_group!r}. Must be one of: "
            + ", ".join(sorted(_GROUP_SPEC))
        )

    spec = _GROUP_SPEC[feature_group]

    # Assemble base feature names
    base_cols: list[str] = []
    for feat_list in spec["bases"]:
        base_cols.extend(feat_list)

    # Compute engineered features on a copy
    windows = spec["windows"]
    has_engineering = spec["ic_eng"] or len(windows) > 0

    if has_engineering:
        eng_windows = windows  # IC eng is always computed alongside window eng
        df_work = engineer_features(df, eng_windows)

        eng_cols: list[str] = []
        if spec["ic_eng"]:
            eng_cols.extend(_IC_ENGINEERED)
        for w in windows:
            eng_cols.extend(
                _W5_ENGINEERED if w == "w5" else _W20_ENGINEERED
            )
    else:
        df_work = df
        eng_cols = []

    desired_cols = base_cols + eng_cols

    # Strict exclusion set
    exclude = set(LABEL_COLS + META_COLS + LEAKY_COLS + ["v3_angle"])

    available: list[str] = []
    missing: list[str] = []
    for c in desired_cols:
        if c in exclude:
            continue
        if c in df_work.columns:
            available.append(c)
        else:
            missing.append(c)

    if missing:
        import warnings
        warnings.warn(
            f"[features] Group {feature_group}: {len(missing)} column(s) not found "
            f"in dataset and will be skipped: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            UserWarning,
            stacklevel=2,
        )

    # Deduplicate while preserving order
    seen: set[str] = set()
    final_cols: list[str] = []
    for c in available:
        if c not in seen:
            seen.add(c)
            final_cols.append(c)

    return df_work[final_cols].copy()


def describe_feature_groups() -> None:
    """Print a summary of all feature group definitions."""
    print("\n── Feature Group Definitions ──────────────────────────────────")
    for grp, spec in _GROUP_SPEC.items():
        n_base = sum(len(fl) for fl in spec["bases"])
        n_eng = (len(_IC_ENGINEERED) if spec["ic_eng"] else 0) + sum(
            len(_W5_ENGINEERED if w == "w5" else _W20_ENGINEERED) for w in spec["windows"]
        )
        print(f"  {grp}  {spec['desc']}")
        print(f"     raw={n_base}  engineered={n_eng}  total≈{n_base + n_eng}")
    print("─" * 63)


# =============================================================================
# LABEL EXTRACTION
# =============================================================================

def get_labels(df: pd.DataFrame) -> pd.Series:
    """
    Extract the classification target.

    Returns
    -------
    Series[int] : 0 = stable, 1 = unstable, 2 = chaotic.
    """
    return df["outcome_class"].astype(int)


# =============================================================================
# FEATURE AUDIT
# =============================================================================

def describe_features(
    df: pd.DataFrame,
    feature_group: str = "F",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate a structured feature audit for a given feature group.

    Acts as a data integrity checkpoint: detects missing window computations,
    pathological distributions, tier imbalance, and unintended leakage.

    Parameters
    ----------
    df            : Raw dataset.
    feature_group : Feature configuration (A–F).
    verbose       : Print formatted summary.

    Returns
    -------
    DataFrame — one row per feature with statistics and tier label.
    """
    feat_df = get_feature_matrix(df, feature_group=feature_group)

    tier_map: dict[str, str] = {}
    for f in IC_FEATURES:
        tier_map[f] = "IC"
    for f in W5_FEATURES:
        tier_map[f] = "W5"
    for f in W20_FEATURES:
        tier_map[f] = "W20"
    for f in _IC_ENGINEERED:
        tier_map[f] = "ENG-IC"
    for f in _W5_ENGINEERED:
        tier_map[f] = "ENG-W5"
    for f in _W20_ENGINEERED:
        tier_map[f] = "ENG-W20"

    rows = []
    for col in feat_df.columns:
        s = feat_df[col]
        rows.append({
            "feature":       col,
            "tier":          tier_map.get(col, "UNKNOWN"),
            "missing_%":     round(100 * s.isna().mean(), 3),
            "min":           float(s.min()),
            "median":        float(s.median()),
            "max":           float(s.max()),
            "std":           float(s.std()),
            "near_constant": bool(s.std() < 1e-8),
        })

    summary = pd.DataFrame(rows)

    if verbose:
        print(f"\n── Feature Audit: Group {feature_group} ────────────────────")
        print(f"  Total features : {len(summary)}")

        tier_counts = summary["tier"].value_counts()
        for tier, cnt in tier_counts.items():
            print(f"  {tier:<10}: {cnt} features")

        bad = summary[summary["missing_%"] > 20]
        if len(bad) > 0:
            print(f"\n  ⚠  High missingness (>20%):")
            for _, row in bad.iterrows():
                print(f"     {row['feature']:<35} {row['missing_%']:.1f}%")

        flat = summary[summary["near_constant"]]
        if len(flat) > 0:
            print(f"\n  ⚠  Near-constant (no predictive signal):")
            for _, row in flat.iterrows():
                print(f"     {row['feature']}")

        print("─" * 55)

    return summary