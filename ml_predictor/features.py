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

Tier 2 — Early Window Dynamics (w5 / w10 / w15 / w20 / w25 / w30)
    Computed strictly within early trajectory windows, where the suffix
    denotes the fraction of the inner orbital period observed:
      w5  = first 5%
      w10 = first 10%
      w15 = first 15%
      w20 = first 20%
      w25 = first 25%
      w30 = first 30%
    Window suffix (_wN) is mandatory — conflating windows violates
    experimental isolation.

    dL_max_{wN} are excluded across all windows. Max angular momentum
    drift is unreliable under discrete trajectory sampling: a single coarse
    integration step produces a spuriously large spike uncorrelated with
    true orbital drift. dE_max is retained — fractional energy violation
    is a global integrator diagnostic, less sensitive to individual step size.

Tier 3 — Engineered Features
    Nonlinear combinations of Tier 1 + Tier 2. Encode known dynamical
    stability laws (Mardling-Aarseth criterion, encounter strength scaling).
    Added only in groups with the + suffix and in group D.

──────────────────────────────────────────────────────────────────────────────
FEATURE GROUP EXPERIMENTS
──────────────────────────────────────────────────────────────────────────────

Baselines (2 groups)
  A   →  IC only                                          (10 features)
  A+  →  IC + IC_eng                                      (14 features)

Core (12 groups)
  B5  →  IC + w5                                          (17 features)
  B10 →  IC + w10                                         (17 features)
  B15 →  IC + w15                                         (17 features)
  B20 →  IC + w20                                         (17 features)
  B25 →  IC + w25                                         (17 features)
  B30 →  IC + w30                                         (17 features)
  B5+ →  IC + IC_eng + w5  + w5_eng                       (22 features)
  B10+→  IC + IC_eng + w10 + w10_eng                      (22 features)
  B15+→  IC + IC_eng + w15 + w15_eng                      (22 features)
  B20+→  IC + IC_eng + w20 + w20_eng                      (22 features)
  B25+→  IC + IC_eng + w25 + w25_eng                      (22 features)
  B30+→  IC + IC_eng + w30 + w30_eng                      (22 features)

Window-only (6 groups)
  C5  →  w5  only                                         (7 features)
  C10 →  w10 only                                         (7 features)
  C15 →  w15 only                                         (7 features)
  C20 →  w20 only                                         (7 features)
  C25 →  w25 only                                         (7 features)
  C30 →  w30 only                                         (7 features)

Upper bound (1 group)
  D   →  IC + IC_eng + w5 + w10 + w15 + w20 + w25 + w30
          + w30_eng                                        (57 features)
        = 10 IC + 4 IC_eng + 6×7 window features + 1 w30_eng
        Role: upper bound / kitchen-sink benchmark.
        Use: establishes performance ceiling, not for deployment.

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
  - dL_max_{wN} for any window (unreliable under discrete sampling)
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

# dL_max columns are excluded across all windows — unreliable under discrete
# trajectory sampling (single coarse step produces spurious spike).
_DL_MAX_COLS: list[str] = [
    f"dL_max_w{s}" for s in [5, 10, 15, 20, 25, 30]
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
# TIER 2: EARLY WINDOW DYNAMICS
# Each window: 7 features (dL_max excluded; dE_max + 3 e_std + 3 r_min)
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

W10_FEATURES: list[str] = [
    "dE_max_w10",
    "e12_std_w10",
    "e13_std_w10",
    "e23_std_w10",
    "r_min_12_w10",
    "r_min_13_w10",
    "r_min_23_w10",
]

W15_FEATURES: list[str] = [
    "dE_max_w15",
    "e12_std_w15",
    "e13_std_w15",
    "e23_std_w15",
    "r_min_12_w15",
    "r_min_13_w15",
    "r_min_23_w15",
]

W20_FEATURES: list[str] = [
    "dE_max_w20",
    "e12_std_w20",
    "e13_std_w20",
    "e23_std_w20",
    "r_min_12_w20",
    "r_min_13_w20",
    "r_min_23_w20",
]

W25_FEATURES: list[str] = [
    "dE_max_w25",
    "e12_std_w25",
    "e13_std_w25",
    "e23_std_w25",
    "r_min_12_w25",
    "r_min_13_w25",
    "r_min_23_w25",
]

W30_FEATURES: list[str] = [
    "dE_max_w30",
    "e12_std_w30",
    "e13_std_w30",
    "e23_std_w30",
    "r_min_12_w30",
    "r_min_13_w30",
    "r_min_23_w30",
]

# Lookup: window label → feature list
_WINDOW_FEATURES: dict[str, list[str]] = {
    "w5":  W5_FEATURES,
    "w10": W10_FEATURES,
    "w15": W15_FEATURES,
    "w20": W20_FEATURES,
    "w25": W25_FEATURES,
    "w30": W30_FEATURES,
}

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

# Count: 1 per window
_W5_ENGINEERED:  list[str] = ["close_encounter_strength_w5"]
_W10_ENGINEERED: list[str] = ["close_encounter_strength_w10"]
_W15_ENGINEERED: list[str] = ["close_encounter_strength_w15"]
_W20_ENGINEERED: list[str] = ["close_encounter_strength_w20"]
_W25_ENGINEERED: list[str] = ["close_encounter_strength_w25"]
_W30_ENGINEERED: list[str] = ["close_encounter_strength_w30"]

# Lookup: window label → engineered feature list
_WINDOW_ENGINEERED: dict[str, list[str]] = {
    "w5":  _W5_ENGINEERED,
    "w10": _W10_ENGINEERED,
    "w15": _W15_ENGINEERED,
    "w20": _W20_ENGINEERED,
    "w25": _W25_ENGINEERED,
    "w30": _W30_ENGINEERED,
}

# Convenience aggregates for external use
IC_ENGINEERED_FEATURES:  list[str] = _IC_ENGINEERED
ALL_ENGINEERED_FEATURES: list[str] = (
    _IC_ENGINEERED
    + _W5_ENGINEERED + _W10_ENGINEERED + _W15_ENGINEERED
    + _W20_ENGINEERED + _W25_ENGINEERED + _W30_ENGINEERED
)
DYN_FEATURES: list[str] = (
    W5_FEATURES + W10_FEATURES + W15_FEATURES
    + W20_FEATURES + W25_FEATURES + W30_FEATURES
)


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
    Compute Tier 3 close_encounter_strength for a specific early window.

    close_encounter_strength_{window}
        max(e_ij_std) / min(r_min_outer) — eccentricity perturbation
        amplitude per unit closest approach distance.

        The perturbation to the inner binary eccentricity during a close
        encounter scales as delta_e ~ (m3 / r_min^2) * dt_enc. Dividing
        the observed eccentricity fluctuation by r_min normalizes by
        encounter depth. High values indicate strong chaotic forcing.

        Uses outer-body encounters only (r_min_13, r_min_23) for the
        denominator — the inner binary separation r_min_12 does not
        measure the same physical process (third-body tidal injection).
    """
    valid_windows = ("w5", "w10", "w15", "w20", "w25", "w30")
    assert window in valid_windows, f"window must be one of {valid_windows}, got {window!r}"

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
    windows : Subset of ["w5", "w10", "w15", "w20", "w25", "w30"] to engineer.

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
#   A    IC only                                          10
#   A+   IC + IC_eng                                     14
#
#   B5   IC + w5                                         17
#   B10  IC + w10                                        17
#   B15  IC + w15                                        17
#   B20  IC + w20                                        17
#   B25  IC + w25                                        17
#   B30  IC + w30                                        17
#   B5+  IC + IC_eng + w5  + w5_eng                     22
#   B10+ IC + IC_eng + w10 + w10_eng                    22
#   B15+ IC + IC_eng + w15 + w15_eng                    22
#   B20+ IC + IC_eng + w20 + w20_eng                    22
#   B25+ IC + IC_eng + w25 + w25_eng                    22
#   B30+ IC + IC_eng + w30 + w30_eng                    22
#
#   C5   w5  only                                         7
#   C10  w10 only                                         7
#   C15  w15 only                                         7
#   C20  w20 only                                         7
#   C25  w25 only                                         7
#   C30  w30 only                                         7
#
#   D    IC + IC_eng + w5 + w10 + w15 + w20 + w25 + w30
#         + w30_eng                                       57
#        (10 IC + 4 IC_eng + 6×7 windows + 1 w30_eng)

_GROUP_SPEC: dict[str, dict] = {

    # ── Baselines ──────────────────────────────────────────────────────────────
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

    # ── Core: IC + single window, no engineering ───────────────────────────────
    "B5": {
        "bases":   [IC_FEATURES, W5_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC + w5 (17 features)",
    },
    "B10": {
        "bases":   [IC_FEATURES, W10_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC + w10 (17 features)",
    },
    "B15": {
        "bases":   [IC_FEATURES, W15_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC + w15 (17 features)",
    },
    "B20": {
        "bases":   [IC_FEATURES, W20_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC + w20 (17 features)",
    },
    "B25": {
        "bases":   [IC_FEATURES, W25_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC + w25 (17 features)",
    },
    "B30": {
        "bases":   [IC_FEATURES, W30_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "IC + w30 (17 features)",
    },

    # ── Core: IC + IC_eng + single window + window_eng ────────────────────────
    "B5+": {
        "bases":   [IC_FEATURES, W5_FEATURES],
        "windows": ["w5"],
        "ic_eng":  True,
        "desc":    "IC + IC_eng + w5 + w5_eng (22 features)",
    },
    "B10+": {
        "bases":   [IC_FEATURES, W10_FEATURES],
        "windows": ["w10"],
        "ic_eng":  True,
        "desc":    "IC + IC_eng + w10 + w10_eng (22 features)",
    },
    "B15+": {
        "bases":   [IC_FEATURES, W15_FEATURES],
        "windows": ["w15"],
        "ic_eng":  True,
        "desc":    "IC + IC_eng + w15 + w15_eng (22 features)",
    },
    "B20+": {
        "bases":   [IC_FEATURES, W20_FEATURES],
        "windows": ["w20"],
        "ic_eng":  True,
        "desc":    "IC + IC_eng + w20 + w20_eng (22 features)",
    },
    "B25+": {
        "bases":   [IC_FEATURES, W25_FEATURES],
        "windows": ["w25"],
        "ic_eng":  True,
        "desc":    "IC + IC_eng + w25 + w25_eng (22 features)",
    },
    "B30+": {
        "bases":   [IC_FEATURES, W30_FEATURES],
        "windows": ["w30"],
        "ic_eng":  True,
        "desc":    "IC + IC_eng + w30 + w30_eng (22 features)",
    },

    # ── Window-only ────────────────────────────────────────────────────────────
    "C5": {
        "bases":   [W5_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "w5 dynamics only (7 features)",
    },
    "C10": {
        "bases":   [W10_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "w10 dynamics only (7 features)",
    },
    "C15": {
        "bases":   [W15_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "w15 dynamics only (7 features)",
    },
    "C20": {
        "bases":   [W20_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "w20 dynamics only (7 features)",
    },
    "C25": {
        "bases":   [W25_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "w25 dynamics only (7 features)",
    },
    "C30": {
        "bases":   [W30_FEATURES],
        "windows": [],
        "ic_eng":  False,
        "desc":    "w30 dynamics only (7 features)",
    },

    # ── Upper bound ────────────────────────────────────────────────────────────
    "D": {
        "bases":   [
            IC_FEATURES,
            W5_FEATURES, W10_FEATURES, W15_FEATURES,
            W20_FEATURES, W25_FEATURES, W30_FEATURES,
        ],
        "windows": ["w30"],   # only w30_eng included; IC_eng via ic_eng=True
        "ic_eng":  True,
        "desc":    "IC + IC_eng + w5–w30 + w30_eng — upper bound (57 features)",
    },
}


def get_feature_matrix(df: pd.DataFrame, feature_group: str = "D") -> pd.DataFrame:
    """
    Construct a leakage-safe, window-consistent feature matrix.

    Parameters
    ----------
    df            : Raw dataset. Must include m1, m2, m3 so that hill_ratio
                    and tidal_compactness_log can be computed for groups with
                    ic_eng=True (individual masses are excluded from the output
                    matrix via META_COLS but required at compute time).
    feature_group : One of A, A+, B5–B30, B5+–B30+, C5–C30, D.

    Returns
    -------
    DataFrame — model-ready, numeric only, no label/meta/leaky columns.

    Notes
    -----
    - v3_angle is always excluded (wrap discontinuity; not encoded).
    - dL_max_{wN} are always excluded (unreliable under discrete sampling).
    - Engineered features are present in groups A+, B5+–B30+, and D.
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

    windows    = spec["windows"]
    has_engineering = spec["ic_eng"] or len(windows) > 0

    df_work = df.copy()

    if has_engineering:
        df_work = engineer_features(df_work, windows)

        eng_cols: list[str] = []
        if spec["ic_eng"]:
            eng_cols.extend(_IC_ENGINEERED)
        for w in windows:
            eng_cols.extend(_WINDOW_ENGINEERED[w])
    else:
        eng_cols = []

    desired_cols = base_cols + eng_cols

    exclude = set(
        LABEL_COLS + META_COLS + LEAKY_COLS
        + _DL_MAX_COLS
        + ["v3_angle"]
    )

    available: list[str] = []
    missing:   list[str] = []
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

    seen:       set[str]  = set()
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
        n_eng  = (len(_IC_ENGINEERED) if spec["ic_eng"] else 0) + sum(
            len(_WINDOW_ENGINEERED[w]) for w in spec["windows"]
        )
        print(f"  {grp:<5}  {spec['desc']}")
        print(f"         raw={n_base}  engineered={n_eng}  total={n_base + n_eng}")
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


def get_feature_group_names() -> list[str]:
    """Return all registered feature group identifiers."""
    return list(_GROUP_SPEC.keys())


# =============================================================================
# FEATURE AUDIT
# =============================================================================

def describe_features(
    df:            pd.DataFrame,
    feature_group: str  = "D",
    verbose:       bool = True,
) -> pd.DataFrame:
    """
    Generate a structured feature audit for a given feature group.

    Parameters
    ----------
    df            : Raw dataset.
    feature_group : Feature configuration.
    verbose       : Print formatted summary.

    Returns
    -------
    DataFrame — one row per feature with statistics and tier label.
    """
    feat_df = get_feature_matrix(df, feature_group=feature_group)

    tier_map: dict[str, str] = {}
    for feat in IC_FEATURES:
        tier_map[feat] = "IC"
    for window, feats in _WINDOW_FEATURES.items():
        for feat in feats:
            tier_map[feat] = window.upper()
    for feat in _IC_ENGINEERED:
        tier_map[feat] = "ENG-IC"
    for window, feats in _WINDOW_ENGINEERED.items():
        for feat in feats:
            tier_map[feat] = f"ENG-{window.upper()}"

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
            print(f"  {tier:<12}: {cnt} features")

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


# =============================================================================
# GROUP D VERIFICATION  (Item 1 — attestation)
# =============================================================================

def verify_group_d_count(df: pd.DataFrame, verbose: bool = True) -> int:
    """
    Verify that feature group D resolves to exactly 57 features on a given
    DataFrame. Raises AssertionError if the count is wrong.

    This is an attestation utility for Item 1 of the ML Engineer Action List.
    It must be called against a DataFrame that contains all expected columns
    (e.g. the synthetic dataset from run_baseline._make_synthetic_dataset, or
    the real metadata CSV). Do NOT call this at import time.

    Expected composition
    --------------------
    10  IC features      (epsilon_total … v3_frac)
     4  IC_eng features  (hill_ratio, energy_partition, hierarchy_log,
                          tidal_compactness_log)
    42  window features  (7 per window × 6 windows: w5–w30)
     1  w30_eng feature  (close_encounter_strength_w30)
    ── = 57

    Parameters
    ----------
    df      : DataFrame with all IC + window columns present.
    verbose : Print the per-tier breakdown when True.

    Returns
    -------
    n_features : int — confirmed feature count (57).
    """
    feat_df = get_feature_matrix(df, feature_group="D")
    n = len(feat_df.columns)

    if verbose:
        print("\n── Group D Feature Count Verification ──────────────────────")
        print(f"  Features returned by get_feature_matrix(df, 'D'): {n}")
        print(f"  Expected breakdown:")
        print(f"    IC features   : {len(IC_FEATURES)}")
        print(f"    IC_eng        : {len(_IC_ENGINEERED)}")
        print(f"    Window (6×7)  : {6 * 7}")
        print(f"    w30_eng       : {len(_W30_ENGINEERED)}")
        print(f"    Total         : {len(IC_FEATURES) + len(_IC_ENGINEERED) + 6*7 + len(_W30_ENGINEERED)}")
        print(f"  Actual columns  : {n}")
        if n == 57:
            print("  ✓  Group D confirmed: 57 features")
        else:
            print(f"  ✗  MISMATCH — check for missing columns in input DataFrame")
            print(f"     Columns present: {list(feat_df.columns)}")
        print("─" * 55)

    assert n == 57, (
        f"Group D feature count mismatch: expected 57, got {n}. "
        f"Ensure all IC, window (w5–w30), and engineered columns are present "
        f"in the input DataFrame. Missing columns are silently skipped by "
        f"get_feature_matrix — check for absent window columns in your CSV."
    )

    return n