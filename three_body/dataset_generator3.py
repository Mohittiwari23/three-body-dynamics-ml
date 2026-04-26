"""
three_body/dataset_generator3.py
==================================
Generates the Phase 3 labelled three-body dataset.

Design principles
-----------------
1. ALL systems are BOUND at t=0 (epsilon_total < 0, v3_frac < sqrt(2)=1.414).
   Ejections from unbound ICs are trivially predictable from E0 > 0 alone.
   Only gravitational-slingshot ejections produce non-trivial classification.

2. MEGNO is stored in metadata for labelling but excluded from ML features.
   Chaotic label is defined as MEGNO > 3 over the FULL trajectory.
   Including MEGNO as a feature gives the model the answer key (circular).

3. Features at TWO window sizes (5% and 20%) enable the key experiment:
   IC-only -> +5% window -> +20% window accuracy.
   This quantifies prediction improvement per unit of simulation cost.

4. Quality gate: dL_max_w20 <= 0.02  AND  dE_max_w20 <= 0.1 (bound systems only).
   dE_max gate catches samples where |E0| ~= 0 makes normalised drift blow up.

5. q_ij = min(mi, mj) / max(mi, mj) <= 1  uniformly enforced at row-write time.
   Previous version used m1/m2 which exceeds 1 when m1 > m2 (scatter regime
   gave q12 up to 1.43, q13 up to 311, q23 up to 527).

6. dE_slope is DROPPED from all feature tiers.
   Symplectic integrators conserve a modified Hamiltonian — energy oscillates
   around E0 with near-zero secular slope for ALL outcomes regardless of fate.
   Correlation with outcome_class ~= -0.05 (noise, not informative).

7. traj_file column is NOT written to the CSV.
   Trajectory files saved to disk; path reconstructable as traj_{idx:05d}.npz.

Regime fractions
----------------
  hierarchical (30%) : outer body far from binary, Hill stability, mostly stable
  asymmetric   (25%) : one dominant mass, mass-ratio diversity, moderate chaos
  compact_equal(25%) : compact near-equal masses, gravitational slingshot ejections
  scatter      (20%) : near-equal fast outer body, edge of stability

  Hierarchical raised from 25% to 30%, asymmetric reduced from 30% to 25%
  to counteract the asymmetric regime's ~60% chaotic rate which was causing
  the chaotic class to dominate at 49% vs the 33% target.

ML feature groups (importable by experiment scripts)
------------------------------------------------------
  IC_NORM  : epsilon_total, h_total, q12, q13, q23, M_total,
             r12_init, r3_sep, v3_frac, e_inner
  IC_RAW   : E0_total, L0_total, M_total, r12_init, r3_sep, v3_frac, e_inner

  DYN_W5   : dE_max_w5, r_min_12_w5, r_min_13_w5, r_min_23_w5,
             e12_std_w5, e13_std_w5, e23_std_w5
  DYN_W20  : same with _w20 suffix

  TIER0 = IC_NORM
  TIER1 = IC_NORM + DYN_W5
  TIER2 = IC_NORM + DYN_W20

Metadata columns (NOT ML features)
-------------------------------------
  MEGNO, MEGNO_clean : label-defining (chaotic <-> MEGNO > 3)
  dL_max_w5, dL_max_w20 : quality filter metrics
  regime, m1, m2, m3, M_total, r12_init, r3_sep, v3_angle, e_inner : raw geometry
  E0_total, L0_total : raw (unnormalised) energy and angular momentum

Label schema
------------
  outcome_class  0=stable  1=unstable(ejection+collision)  2=chaotic
  outcome_class4 0=stable  1=ejection  2=collision  3=chaotic
  outcome        string label
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from three_body.integrator3 import ThreeBodySystem, run_simulation_3body
from three_body.labeller import label_result

G = 1.0
N_SAMPLES     = 5000
COMPUTE_MEGNO = True
SEED          = 42
OUTPUT_DIR    = Path(__file__).resolve().parent.parent / "data" / "dataset3"


# ---------------------------------------------------------------------------
# ML feature group definitions  (import these in experiment scripts)
# ---------------------------------------------------------------------------

IC_NORM = [
    "epsilon_total",  # E0/M_total  -- normalised specific energy
    "h_total",        # |L0|/M_total -- normalised specific angular momentum
    "q12",            # min(m1,m2)/max(m1,m2) <= 1 -- inner binary mass ratio
    "q13",            # min(m1,m3)/max(m1,m3) <= 1 -- outer body mass ratio
    "q23",            # min(m2,m3)/max(m2,m3) <= 1 -- outer body mass ratio
    "M_total",        # total mass (scale factor)
    "r12_init",       # initial inner binary separation (periapsis)
    "r3_sep",         # initial outer body distance
    "v3_frac",        # v3/v_circ(r3) -- outer body velocity fraction < sqrt(2)
    "e_inner",        # inner binary eccentricity in [0, 1)
]

IC_RAW = [
    "E0_total",   # raw total energy (ablation baseline)
    "L0_total",   # raw total angular momentum (ablation baseline)
    "M_total", "r12_init", "r3_sep", "v3_frac", "e_inner",
]

# dE_slope excluded: near-zero for all outcomes due to symplectic conservation
DYN_W5 = [
    "dE_max_w5",
    "r_min_12_w5", "r_min_13_w5", "r_min_23_w5",
    "e12_std_w5",  "e13_std_w5",  "e23_std_w5",
]

DYN_W20 = [
    "dE_max_w20",
    "r_min_12_w20", "r_min_13_w20", "r_min_23_w20",
    "e12_std_w20",  "e13_std_w20",  "e23_std_w20",
]

TIER0 = IC_NORM
TIER1 = IC_NORM + DYN_W5
TIER2 = IC_NORM + DYN_W20


# ---------------------------------------------------------------------------
# CoM frame placement
# ---------------------------------------------------------------------------

def _com_3body(rng, m1, m2, m3, r12, r3_sep, v3_angle, v3_mag, e_inner=0.0):
    """
    Place three bodies in the centre-of-mass frame.
    r12 = periapsis of inner binary.
    v_peri = v_circ(r12) * sqrt(1 + e_inner)  [exact Kepler formula]
    """
    M12 = m1 + m2; M = M12 + m3
    r1 = np.array([ r12*m2/M12, 0.0])
    r2 = np.array([-r12*m1/M12, 0.0])
    r3 = np.array([r3_sep*np.cos(v3_angle), r3_sep*np.sin(v3_angle)])
    com = (m1*r1 + m2*r2 + m3*r3) / M
    r1 -= com; r2 -= com; r3 -= com

    v_inner = np.sqrt(G*M12/r12) * np.sqrt(1.0 + e_inner)
    v1_lab  = np.array([0.0,  v_inner*m2/M12])
    v2_lab  = np.array([0.0, -v_inner*m1/M12])
    v3_lab  = v3_mag * np.array([-np.sin(v3_angle), np.cos(v3_angle)])
    vcom    = (m1*v1_lab + m2*v2_lab + m3*v3_lab) / M
    v1_lab -= vcom; v2_lab -= vcom; v3_lab -= vcom
    return r1, r2, r3, v1_lab, v2_lab, v3_lab


# ---------------------------------------------------------------------------
# Regime samplers  (all v3_frac < sqrt(2), all q_ij <= 1 enforced at write)
# ---------------------------------------------------------------------------

def _sample_hierarchical(rng):
    """r3/r12 in [3,15]. Hill stability regime -> mostly stable."""
    M   = float(np.exp(rng.uniform(np.log(5),   np.log(200))))
    q12 = float(np.exp(rng.uniform(np.log(0.1), np.log(1.0))))
    m2  = M/(1+q12); m1 = q12*m2
    m3  = M * float(rng.uniform(0.001, 0.05))
    r12    = float(np.exp(rng.uniform(np.log(0.5), np.log(3.0))))
    r3_sep = r12 * float(rng.uniform(3.0, 15.0))
    v_circ = np.sqrt(G*(M+m3)/r3_sep)
    v3_frac = float(rng.uniform(0.4, 1.10))
    e_inner = float(rng.uniform(0.0, 0.5))
    return m1, m2, m3, r12, r3_sep, v3_frac*v_circ, float(rng.uniform(0, 2*np.pi)), e_inner


def _sample_asymmetric(rng):
    """One dominant mass (q12 in [0.01,0.3]). Moderate chaos."""
    M   = float(np.exp(rng.uniform(np.log(10),   np.log(500))))
    q12 = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
    m2  = M/(1+q12); m1 = q12*m2
    m3  = M * float(rng.uniform(0.005, 0.1))
    r12    = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0))))
    r3_sep = r12 * float(rng.uniform(1.5, 6.0))
    v_circ = np.sqrt(G*(M+m3)/r3_sep)
    v3_frac = float(rng.uniform(0.5, 1.30))
    e_inner = float(rng.uniform(0.0, 0.6))
    return m1, m2, m3, r12, r3_sep, v3_frac*v_circ, float(rng.uniform(0, 2*np.pi)), e_inner


def _sample_compact_equal(rng):
    """r3/r12 in [0.5,2.0]. Strong encounters -> slingshot ejections."""
    M   = float(np.exp(rng.uniform(np.log(3), np.log(30))))
    q12 = float(rng.uniform(0.5, 1.0))
    m2  = M/(1+q12)/2; m1 = q12*m2; m3 = M-m1-m2
    r12    = float(rng.uniform(0.5, 4.0))
    r3_sep = r12 * float(rng.uniform(0.5, 2.0))
    v_circ = np.sqrt(G*M/r3_sep)
    v3_frac = float(rng.uniform(0.6, 1.30))
    e_inner = float(rng.uniform(0.0, 0.6))
    return m1, m2, m3, r12, r3_sep, v3_frac*v_circ, float(rng.uniform(0, 2*np.pi)), e_inner


def _sample_scatter(rng):
    """Near-equal masses, fast outer body. Edge of stability."""
    M  = float(np.exp(rng.uniform(np.log(3), np.log(50))))
    m1 = M * float(rng.uniform(0.30, 0.45))
    m2 = M * float(rng.uniform(0.30, 0.45))
    m3 = M - m1 - m2
    r12    = float(rng.uniform(0.5, 3.0))
    r3_sep = r12 * float(rng.uniform(0.8, 2.5))
    v_circ = np.sqrt(G*M/r3_sep)
    v3_frac = float(rng.uniform(0.8, 1.35))
    e_inner = float(rng.uniform(0.0, 0.5))
    return m1, m2, m3, r12, r3_sep, v3_frac*v_circ, float(rng.uniform(0, 2*np.pi)), e_inner


SAMPLERS = {
    "hierarchical":  _sample_hierarchical,
    "asymmetric":    _sample_asymmetric,
    "compact_equal": _sample_compact_equal,
    "scatter":       _sample_scatter,
}

REGIME_FRACTIONS = {
    "hierarchical":  0.30,
    "asymmetric":    0.25,
    "compact_equal": 0.25,
    "scatter":       0.20,
}


def _integration_params(m1, m2, m3, r12, r3_sep):
    M_total = m1+m2+m3; M12 = m1+m2
    min_sep = max(min(r12*0.3, r3_sep*0.3), 0.01)
    dt      = min(np.sqrt(min_sep**3/(G*M_total))/20.0, 0.05)
    T_inner = 2.0*np.pi*np.sqrt(r12**3/(G*M12))
    n_steps = int(max(15.0*T_inner, 50.0)/dt)
    return dt, max(5000, min(n_steps, 20000))


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(
    n_samples:     int  = N_SAMPLES,
    compute_megno: bool = COMPUTE_MEGNO,
    seed:          int  = SEED,
    output_dir:    Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Generate n_samples labelled three-body simulations.
    Saves metadata CSV and trajectory .npz files (path NOT in CSV).
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    counts = {r: int(n_samples*f) for r, f in REGIME_FRACTIONS.items()}
    counts["scatter"] = n_samples - sum(v for k,v in counts.items() if k != "scatter")
    regimes = []
    for r, cnt in counts.items(): regimes += [r]*cnt
    regimes = np.array(regimes); rng.shuffle(regimes)

    OUTCOME_MAP3 = {"stable": 0, "ejection": 1, "collision": 1, "chaotic": 2}
    rows = []
    outcome_counts = {"stable": 0, "unstable": 0, "chaotic": 0}
    rejected = {"nan": 0, "dL": 0, "dE": 0, "exception": 0}
    t0 = time.time()

    print(f"Generating {n_samples} three-body simulations -> {output_dir}")
    print(f"MEGNO: {'ON' if compute_megno else 'OFF'}  (full trajectory, labelling only)")
    print(f"Regime mix: " + "  ".join(f"{r}:{c}" for r, c in counts.items()))
    print(f"\n{'idx':>5}  {'regime':>15}  {'outcome':>10}  {'MEGNO':>7}  {'ms':>6}")
    print("-"*55)

    for idx, regime in enumerate(regimes):
        m1, m2, m3, r12, r3_sep, v3_mag, v3_angle, e_inner = SAMPLERS[regime](rng)
        r1, r2, r3, v1, v2, v3 = _com_3body(
            rng, m1, m2, m3, r12, r3_sep, v3_angle, v3_mag, e_inner
        )
        dt, n_steps = _integration_params(m1, m2, m3, r12, r3_sep)

        sys3 = ThreeBodySystem(
            G=G, m1=m1, m2=m2, m3=m3,
            r1_0=r1, r2_0=r2, r3_0=r3,
            v1_0=v1, v2_0=v2, v3_0=v3,
            dt=dt, n_steps=n_steps, label=f"s{idx:05d}",
        )

        t_sim = time.time()
        try:
            result        = run_simulation_3body(sys3, compute_megno=compute_megno)
            result, feats = label_result(result)
        except Exception:
            rejected["exception"] += 1; continue

        if (np.isnan(result.E_hist).any() or np.isnan(result.L_hist).any()
                or np.isnan(result.traj).any()):
            rejected["nan"] += 1; continue

        dL = feats["dL_max_w20"]
        dE = feats["dE_max_w20"]
        if dL > 0.02:
            rejected["dL"] += 1; continue
        if dE > 0.1 and result.E0 < 0:
            rejected["dE"] += 1; continue

        ms = int((time.time()-t_sim)*1000)
        bucket = "unstable" if result.outcome in ("ejection","collision") else result.outcome
        outcome_counts[bucket] += 1

        meg   = result.MEGNO_final
        meg_c = float(np.clip(meg, 0, 10)) if not np.isnan(meg) else 2.0
        M_total = m1 + m2 + m3

        # q_ij = min/max convention: always <= 1, consistent across all regimes
        q12 = min(m1, m2) / max(m1, m2)
        q13 = min(m1, m3) / max(m1, m3)
        q23 = min(m2, m3) / max(m2, m3)

        # Save trajectory (no path stored in CSV; reconstruct as traj_{idx:05d}.npz)
        np.savez_compressed(
            traj_dir / f"traj_{idx:05d}.npz",
            traj   = result.traj.astype(np.float32),
            E_hist = result.E_hist.astype(np.float32),
            L_hist = result.L_hist.astype(np.float32),
            time   = result.time.astype(np.float32),
        )

        rows.append({
            # Identifiers
            "idx": idx, "regime": regime,
            # Raw geometry (metadata)
            "m1": m1, "m2": m2, "m3": m3, "M_total": M_total,
            "r12_init": r12, "r3_sep": r3_sep,
            "v3_angle": v3_angle, "e_inner": e_inner,
            # IC features NORMALISED (TIER0)
            "q12": q12, "q13": q13, "q23": q23,
            "v3_frac":       v3_mag / np.sqrt(G*M_total/r3_sep),
            "epsilon_total": result.E0 / M_total,
            "h_total":       result.L0 / M_total,
            # IC features RAW (ablation only)
            "E0_total": result.E0,
            "L0_total": result.L0,
            # 5% window features (TIER1) -- dE_slope excluded (zero signal)
            "dE_max_w5":   feats["dE_max_w5"],
            "r_min_12_w5": feats["r_min_12_w5"],
            "r_min_13_w5": feats["r_min_13_w5"],
            "r_min_23_w5": feats["r_min_23_w5"],
            "e12_std_w5":  feats["e12_std_w5"],
            "e13_std_w5":  feats["e13_std_w5"],
            "e23_std_w5":  feats["e23_std_w5"],
            # 20% window features (TIER2) -- dE_slope excluded
            "dE_max_w20":   feats["dE_max_w20"],
            "r_min_12_w20": feats["r_min_12_w20"],
            "r_min_13_w20": feats["r_min_13_w20"],
            "r_min_23_w20": feats["r_min_23_w20"],
            "e12_std_w20":  feats["e12_std_w20"],
            "e13_std_w20":  feats["e13_std_w20"],
            "e23_std_w20":  feats["e23_std_w20"],
            # Metadata only (NOT ML features)
            "MEGNO":       meg,
            "MEGNO_clean": meg_c,
            "dL_max_w5":   feats["dL_max_w5"],
            "dL_max_w20":  feats["dL_max_w20"],
            # Labels
            "outcome":        result.outcome,
            "outcome_class":  OUTCOME_MAP3[result.outcome],
            "outcome_class4": result.outcome_class,
        })

        if idx % 250 == 0 or idx < 5:
            print(f"{idx:5d}  {regime:>15}  {result.outcome:>10}  "
                  f"{meg_c:7.3f}  {ms:6d}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "metadata3.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    total   = len(df)
    if total:
        print(f"\nDone in {elapsed:.1f}s  ({elapsed/total*1000:.0f} ms/sample)")
    print(f"Saved {total} rows -> {csv_path}")
    print(f"Rejected: {rejected}")
    print("\nOutcome distribution:")
    for name, cnt in outcome_counts.items():
        pct = 100*cnt/total if total else 0
        print(f"  {name:>12}: {cnt:5d}  ({pct:.1f}%)")
    print(f"\nML feature tiers:")
    print(f"  TIER0  IC_NORM only:          {len(TIER0)} features")
    print(f"  TIER1  IC_NORM + 5% window:   {len(TIER1)} features")
    print(f"  TIER2  IC_NORM + 20% window:  {len(TIER2)} features")
    return df


if __name__ == "__main__":
    generate()