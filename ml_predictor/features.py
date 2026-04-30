"""
ml_predictor/features.py
==============

Physics-informed feature engineering pipeline for three-body instability
prediction using early-time gravitational dynamics.

──────────────────────────────────────────────────────────────────────────────
DESIGN PHILOSOPHY
──────────────────────────────────────────────────────────────────────────────

Feature information is partitioned into three causally distinct tiers:

Tier 1 — Initial Conditions (t = 0)
    Pure system state before any dynamical evolution. Encodes geometry,
    energy, angular momentum, mass hierarchy, and orbital shape.
    Causally valid unconditionally.

Tier 2 — Early Window Dynamics (w5 / w20)
    Computed strictly within early trajectory windows. Encodes incipient
    instability signatures: energy drift, eccentricity fluctuations, closest
    approaches. Window suffix (_w5, _w20) is mandatory — conflating windows
    violates experimental isolation.

    dL_max_{w5,w20} are excluded. Max angular momentum drift is unreliable
    under discrete trajectory sampling: a single coarse integration step
    produces a spuriously large spike uncorrelated with true orbital drift.
    dE_max is retained — fractional energy violation is a global integrator
    diagnostic, less sensitive to individual step size.

Tier 3 — Engineered Features
    Nonlinear combinations of Tier 1 + Tier 2. Encode known dynamical
    stability laws (Mardling-Aarseth criterion, encounter strength scaling).
    Added only in Group F to keep ablation groups A–E clean.

──────────────────────────────────────────────────────────────────────────────
FEATURE GROUP EXPERIMENTS (A–F)
──────────────────────────────────────────────────────────────────────────────

  A   →  IC only                              (10 features)
  A+  →  IC + IC-eng                          (14 features)
  B   →  IC + w5                              (17 features)
  B+  →  IC + IC-eng + w5 + w5-eng           (22 features)
  C   →  IC + w20                             (17 features)
  C+  →  IC + IC-eng + w20 + w20-eng         (22 features)
  D   →  w5 only                              (7 features)
  E   →  w20 only                             (7 features)
  F   →  IC + IC-eng + w5 + w20 + w20-eng    (29 features)

──────────────────────────────────────────────────────────────────────────────
LEAKAGE POLICY (ENFORCED)
──────────────────────────────────────────────────────────────────────────────

NEVER include:
  - outcome labels (outcome, outcome_class, outcome_class4)
  - MEGNO / MEGNO_clean (requires full trajectory integration)
  - E0_total, L0_total (absolute-scale duplicates of epsilon_total / h_total)
  - individual mass columns m1, m2, m3 (mass ratios encode all physically
    distinct information; absolute individual masses are degenerate with G
    in N-body units). M_total is included as an IC feature — it sets the
    overall gravitational scale and enters the Mardling-Aarseth criterion.
  - dataset metadata (traj_file, idx)
  - window-B features in a window-A experiment group
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
    "m1", "m2", "m3",     # individual masses — use ratios + M_total instead
    "E0_total", "L0_total",
]

LEAKY_COLS: list[str] = [
    "MEGNO",
    "MEGNO_clean",
]

# =============================================================================
# TIER 1: INITIAL CONDITION FEATURES  (t = 0)
# Count: 10
# =============================================================================

IC_FEATURES: list[str] = [
    "epsilon_total",  # total specific energy
    "h_total",        # total specific angular momentum magnitude
    "q12",            # m1/m2 inner binary mass ratio
    "q13",            # m1/m3
    "q23",            # m2/m3
    "M_total",        # total system mass — sets gravitational scale and
                      # enters the Mardling-Aarseth criterion directly
    "r12_init",       # inner binary separation
    "r3_sep",         # outer body separation from system barycentre
    "e_inner",        # initial inner binary eccentricity [0, 1)
    "v3_frac",        # |v3| / v_circular
]

# =============================================================================
# TIER 2: EARLY WINDOW DYNAMICS (w5 = first 5% of inner orbital period)
# Count: 7
# =============================================================================

W5_FEATURES: list[str] = [
    "dE_max_w5",
    "e12_std_w5",
    "e13_std_w5",
    "e23_std_w5",
    "r_min_12_w5",
    "r_min_13_w5",
    "r_min_23_w5",
]

# =============================================================================
# TIER 2: EARLY WINDOW DYNAMICS (w20 = first 20% of inner orbital period)
# Count: 7
# =============================================================================

W20_FEATURES: list[str] = [
    "dE_max_w20",
    "e12_std_w20",
    "e13_std_w20",
    "e23_std_w20",
    "r_min_12_w20",
    "r_min_13_w20",
    "r_min_23_w20",
]

# =============================================================================
# TIER 3: ENGINEERED FEATURES
# =============================================================================

# Count: 4
_IC_ENGINEERED: list[str] = [
    "hill_ratio",            # Mardling-Aarseth stability proxy
    "energy_partition",      # binding energy at outer orbit scale
    "hierarchy_log",         # log separation ratio — hierarchy strength
    "tidal_compactness_log", # log tidal forcing scale — outer-body compactness
]

# Count: 1
_W5_ENGINEERED: list[str] = [
    "close_encounter_strength_w5",
]

# Count: 1
_W20_ENGINEERED: list[str] = [
    "close_encounter_strength_w20",
]

# Convenience aggregates for external use
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
    Compute Tier 3 IC-engineered features. Called only for groups with ic_eng=True.

    Requires columns present in the raw dataset: r3_sep, r12_init,
    M_total, m3, m1, m2, epsilon_total.

    hill_ratio
        Mardling-Aarseth (2001) hierarchical stability proxy.

        Stability requires:
            (a_out / a_in) > C * (M_total / m3)^(1/3) / (1 - e_out)

        Computed (ignoring e_out, not available as a direct IC feature):
            hill_ratio = (r3_sep / r12_init) * (M_total / m3)^(-1/3)

        M_total and m3 are raw dataset columns. Values ~1 indicate the
        outer body is near the Hill sphere of the inner binary.

    energy_partition
        |epsilon_total| * r3_sep — binding energy at the outer orbit scale.
        Small values indicate a loosely bound outer body, requiring little
        energy exchange for ejection.

    hierarchy_log
        log(r3_sep / r12_init) — log-scale separation ratio.

        The raw ratio r3_sep / r12_init enters both the Mardling-Aarseth
        criterion and the encounter time scale. The log transform compresses
        the right-skewed distribution of outer/inner separations (which
        spans several orders of magnitude across hierarchical and compact
        regimes) into a form that tree-based models can threshold linearly.
        Negative values indicate r3_sep < r12_init, a physically pathological
        configuration almost always chaotic.

    tidal_compactness_log
        log(m3) - log(m1 + m2) - 3*log(r3_sep) + 3*log(r12_init)

        Encodes the log-scale tidal forcing amplitude of the third body on
        the inner binary. The tidal acceleration on the inner binary from
        body 3 scales as F_tidal ~ m3 / r3_sep^3, while the inner binary's
        self-gravity scales as ~ (m1+m2) / r12_init^3. The ratio
        F_tidal / F_self ~ (m3 / (m1+m2)) * (r12_init / r3_sep)^3
        measures how strongly the outer body perturbs the inner orbit at
        t = 0. High values signal strong tidal forcing even before any
        dynamical evolution — a leading-order chaos predictor for non-
        hierarchical configurations.
    """
    eps = 1e-12

    M_total_over_m3 = df["M_total"] / df["m3"]

    df["hill_ratio"] = (df["r3_sep"] / df["r12_init"]) * (
        M_total_over_m3 ** (-1.0 / 3.0)
    )
    df["energy_partition"] = np.abs(df["epsilon_total"]) * df["r3_sep"]

    df["hierarchy_log"] = np.log(
        (df["r3_sep"] + eps) / (df["r12_init"] + eps)
    )

    df["tidal_compactness_log"] = (
        np.log(df["m3"] + eps)
        - np.log(df["m1"] + df["m2"] + eps)
        - 3.0 * np.log(df["r3_sep"] + eps)
        + 3.0 * np.log(df["r12_init"] + eps)
    )

    return df


def _engineer_window_features(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    Compute Tier 3 features for a specific early window.

    close_encounter_strength_{window}
        max(e_ij_std) / min(r_min_outer) — eccentricity perturbation
        amplitude per unit closest approach distance.

        The perturbation to the inner binary eccentricity during a close
        encounter scales as delta_e ~ (m3 / r_min^2) * dt_enc. Dividing
        the observed eccentricity fluctuation by r_min normalizes by
        encounter depth. High values indicate strong chaotic forcing.
    """
    assert window in ("w5", "w20"), f"window must be 'w5' or 'w20', got {window!r}"

    if f"dE_max_{window}" not in df.columns:
        return df

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

    return df


def engineer_features(df: pd.DataFrame, windows: list[str]) -> pd.DataFrame:
    """
    Compute all Tier 3 engineered features for the requested windows.

    Parameters
    ----------
    df      : DataFrame containing IC + requested window dynamic features.
              Must include m1, m2, m3 (for hill_ratio, tidal_compactness_log)
              even though individual masses are excluded from the output
              matrix via META_COLS but required at compute time.
    windows : Subset of ["w5", "w20"] to engineer.

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

# Feature group counts:
#
#   A   IC only                                   10 + 0 + 0 = 10
#   A+  IC + IC_eng                               10 + 4 + 0 = 14
#   B   IC + w5                                   17 + 0 + 0 = 17
#   B+  IC + IC_eng + w5 + w5_eng                 17 + 4 + 1 = 22
#   C   IC + w20                                  17 + 0 + 0 = 17
#   C+  IC + IC_eng + w20 + w20_eng               17 + 4 + 1 = 22
#   D   w5 only                                    7 + 0 + 0 =  7
#   E   w20 only                                   7 + 0 + 0 =  7
#   F   IC + IC_eng + w5 + w20 + w20_eng          24 + 4 + 1 = 29
#
# NOTE: Group F computes to 29 features (10 IC + 4 IC_eng + 7 W5 + 7 W20
# + 1 W20_eng). IC_eng grew from 2 to 4 with the addition of hierarchy_log
# and tidal_compactness_log.

_GROUP_SPEC: dict[str, dict] = {
    "A": {
        "bases":   [IC_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC only — raw baseline (10 features)",
    },
    "A+": {
        "bases":   [IC_FEATURES],
        "windows": [],
        "ic_eng":  True,
        "desc":    "IC + IC-engineered — physics invariants (14 features)",
    },
    "B": {
        "bases":   [IC_FEATURES, W5_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC + w5 — window gain, no physics eng. (17 features)",
    },
    "B+": {
        "bases":   [IC_FEATURES, W5_FEATURES],
        "windows": ["w5"],
        "ic_eng":  True,
        "desc":    "IC + IC-eng + w5 + w5-eng — full 5% model (22 features)",
    },
    "C": {
        "bases":   [IC_FEATURES, W20_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC + w20 — window gain, no physics eng. (17 features)",
    },
    "C+": {
        "bases":   [IC_FEATURES, W20_FEATURES],
        "windows": ["w20"],
        "ic_eng":  True,
        "desc":    "IC + IC-eng + w20 + w20-eng — full 20% model (22 features)",
    },
    "D": {
        "bases":   [W5_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "w5 dynamics only — pure dynamics test (7 features)",
    },
    "E": {
        "bases":   [W20_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "w20 dynamics only — pure dynamics test (7 features)",
    },
    "F": {
        "bases":   [IC_FEATURES, W5_FEATURES, W20_FEATURES],
        "windows": ["w20"],
        "ic_eng":  True,
        "desc":    "IC + IC-eng + w5 + w20 + w20-eng — full model (29 features)",
    },
}


def get_feature_matrix(df: pd.DataFrame, feature_group: str = "C") -> pd.DataFrame:
    """
    Construct a leakage-safe, window-consistent feature matrix.

    Parameters
    ----------
    df            : Raw dataset. Must include m1, m2, m3 so that hill_ratio
                    and tidal_compactness_log can be computed for groups with
                    ic_eng=True (individual masses are excluded from the output
                    matrix via META_COLS but required at compute time).
    feature_group : One of A–F.

    Returns
    -------
    DataFrame — model-ready, numeric only, no label/meta/leaky columns.

    Notes
    -----
    - v3_angle is always excluded (wrap discontinuity; not encoded).
    - Engineered features are present in groups A+, B+, C+, and F.
    - Missing columns trigger a UserWarning and are skipped gracefully.
    """
    feature_group = feature_group.upper()
    if feature_group not in _GROUP_SPEC:
        raise ValueError(
            f"Invalid feature_group {feature_group!r}. Must be one of: "
            + ", ".join(sorted(_GROUP_SPEC))
        )

    spec = _GROUP_SPEC[feature_group]

    base_cols: list[str] = []
    for feat_list in spec["bases"]:
        base_cols.extend(feat_list)

    windows = spec["windows"]
    has_engineering = spec["ic_eng"] or len(windows) > 0

    df_work = df.copy()

    if has_engineering:
        df_work = engineer_features(df_work, windows)

        eng_cols: list[str] = []
        if spec["ic_eng"]:
            eng_cols.extend(_IC_ENGINEERED)
        for w in windows:
            eng_cols.extend(_W5_ENGINEERED if w == "w5" else _W20_ENGINEERED)
    else:
        eng_cols = []

    desired_cols = base_cols + eng_cols

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
            len(_W5_ENGINEERED if w == "w5" else _W20_ENGINEERED)
            for w in spec["windows"]
        )
        print(f"  {grp}  {spec['desc']}")
        print(f"     raw={n_base}  engineered={n_eng}  total={n_base + n_eng}")
    print("─" * 63)


# =============================================================================
# LABEL EXTRACTION
# =============================================================================

def get_labels(df: pd.DataFrame) -> pd.Series:
    """
    Returns
    -------
    Series[int] : 0 = stable, 1 = unstable, 2 = chaotic.
    """
    return df["outcome_class"].astype(int)


# features.py
def get_feature_group_names() -> list[str]:
    """Return all registered feature group identifiers."""
    return list(_GROUP_SPEC.keys())


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
    for feat in IC_FEATURES:
        tier_map[feat] = "IC"
    for feat in W5_FEATURES:
        tier_map[feat] = "W5"
    for feat in W20_FEATURES:
        tier_map[feat] = "W20"
    for feat in _IC_ENGINEERED:
        tier_map[feat] = "ENG-IC"
    for feat in _W5_ENGINEERED:
        tier_map[feat] = "ENG-W5"
    for feat in _W20_ENGINEERED:
        tier_map[feat] = "ENG-W20"

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