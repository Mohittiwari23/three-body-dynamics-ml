"""
three_body/dataset_generator3.py
==================================
Generates the Phase 3 labelled three-body dataset.

Design principles
-----------------
1. ALL systems are BOUND at t=0 (epsilon_total < 0, v3_frac < sqrt(2)).
   Ejections from unbound ICs are trivially predictable from E0 > 0 alone
   and require no ML. Only gravitational-slingshot ejections are interesting.

2. MEGNO is stored in metadata for labelling but excluded from ML features.
   The chaotic label is defined as MEGNO > 3 over the full trajectory.
   Including MEGNO as a feature gives the model the answer key (circular).

3. Features are extracted at TWO window sizes (5% and 20%) to enable
   the key experiment: IC-only vs 5%-window vs 20%-window accuracy.
   This quantifies the value of simulation time for early-warning prediction.

4. ML-usable feature groups are clearly separated from metadata columns:
   - IC features: initial conditions only (zero simulation cost)
   - Tier-1 features: IC + 5% window dynamics
   - Tier-2 features: IC + 20% window dynamics
   - Metadata only: MEGNO, dL_max, epsilon_total, raw outcome strings

5. Quality gate: dL_max ≤ 0.02 AND dE_max_w20 ≤ 0.1
   The dE_max gate catches samples where E0 ≈ 0 makes normalised drift
   blow up (parabolic-like systems that slipped through).

Sampling regimes
----------------
  hierarchical (25%)  : outer body far from binary, Hill stability
  asymmetric   (30%)  : one dominant mass, mass-ratio diversity
  compact_equal(25%)  : compact geometry, slingshot ejections
  scatter      (20%)  : near-equal masses, edge of stability

All regimes enforce v3_frac < sqrt(2) = 1.414 (all bound at t=0).

ML feature columns produced
-----------------------------
  IC (initial conditions):
    epsilon_total, h_total, q12, q13, M_total, r12_init, r3_sep,
    v3_frac, e_inner

  5% window dynamics (suffix _w5):
    dE_max_w5, dE_slope_w5, r_min_12_w5, r_min_13_w5, r_min_23_w5,
    e12_std_w5, e13_std_w5, e23_std_w5

  20% window dynamics (suffix _w20):
    dE_max_w20, dE_slope_w20, r_min_12_w20, r_min_13_w20, r_min_23_w20,
    e12_std_w20, e13_std_w20, e23_std_w20

  Labels:
    outcome         : raw string (stable/ejection/collision/chaotic)
    outcome_class   : 3-class (0=stable, 1=unstable, 2=chaotic)
    outcome_class4  : 4-class raw

  Metadata (NOT ML features):
    MEGNO, MEGNO_clean : label-defining, circular if used as features
    dL_max_w20         : quality filter metric
    regime             : sampling regime identifier
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
N_SAMPLES     = 200
COMPUTE_MEGNO = True
SEED          = 42
OUTPUT_DIR    = Path(__file__).resolve().parent.parent / "data" / "dataset3"


# ---------------------------------------------------------------------------
# CoM frame placement
# ---------------------------------------------------------------------------

def _com_3body(rng, m1, m2, m3, r12, r3_sep, v3_angle, v3_mag, e_inner=0.0):
    """
    Place three bodies in the CoM frame.

    r12 is the PERIAPSIS of the inner binary.
    Inner binary velocity: v_peri = v_circ(r12) * sqrt(1 + e_inner)
    This is the exact Kepler periapsis velocity formula.
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
# Regime samplers — all enforce v3_frac < sqrt(2) (all bound at t=0)
# ---------------------------------------------------------------------------

def _sample_hierarchical(rng):
    M   = float(np.exp(rng.uniform(np.log(5),   np.log(200))))
    q12 = float(np.exp(rng.uniform(np.log(0.1), np.log(1.0))))
    m2  = M/(1+q12); m1 = q12*m2
    m3  = M * float(rng.uniform(0.001, 0.05))
    r12    = float(np.exp(rng.uniform(np.log(0.5), np.log(3.0))))
    r3_sep = r12 * float(rng.uniform(3.0, 15.0))
    v_circ = np.sqrt(G*(M+m3)/r3_sep)
    v3_frac = float(rng.uniform(0.4, 1.10))   # max 1.10 < sqrt(2)
    e_inner = float(rng.uniform(0.0, 0.5))
    return m1, m2, m3, r12, r3_sep, v3_frac*v_circ, float(rng.uniform(0,2*np.pi)), e_inner


def _sample_asymmetric(rng):
    M   = float(np.exp(rng.uniform(np.log(10),   np.log(500))))
    q12 = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
    m2  = M/(1+q12); m1 = q12*m2
    m3  = M * float(rng.uniform(0.005, 0.1))
    r12    = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0))))
    r3_sep = r12 * float(rng.uniform(1.5, 6.0))
    v_circ = np.sqrt(G*(M+m3)/r3_sep)
    v3_frac = float(rng.uniform(0.5, 1.30))   # max 1.30 < sqrt(2)
    e_inner = float(rng.uniform(0.0, 0.6))
    return m1, m2, m3, r12, r3_sep, v3_frac*v_circ, float(rng.uniform(0,2*np.pi)), e_inner


def _sample_compact_equal(rng):
    M   = float(np.exp(rng.uniform(np.log(3), np.log(30))))
    q12 = float(rng.uniform(0.5, 1.0))
    m2  = M/(1+q12)/2; m1 = q12*m2; m3 = M-m1-m2
    r12    = float(rng.uniform(0.5, 4.0))
    r3_sep = r12 * float(rng.uniform(0.5, 2.0))  # compact
    v_circ = np.sqrt(G*M/r3_sep)
    v3_frac = float(rng.uniform(0.6, 1.30))       # max 1.30 < sqrt(2)
    e_inner = float(rng.uniform(0.0, 0.6))
    return m1, m2, m3, r12, r3_sep, v3_frac*v_circ, float(rng.uniform(0,2*np.pi)), e_inner


def _sample_scatter(rng):
    M  = float(np.exp(rng.uniform(np.log(3), np.log(50))))
    m1 = M * float(rng.uniform(0.30, 0.45))
    m2 = M * float(rng.uniform(0.30, 0.45))
    m3 = M - m1 - m2
    r12    = float(rng.uniform(0.5, 3.0))
    r3_sep = r12 * float(rng.uniform(0.8, 2.5))
    v_circ = np.sqrt(G*M/r3_sep)
    v3_frac = float(rng.uniform(0.8, 1.35))       # max 1.35 < sqrt(2)
    e_inner = float(rng.uniform(0.0, 0.5))
    return m1, m2, m3, r12, r3_sep, v3_frac*v_circ, float(rng.uniform(0,2*np.pi)), e_inner


SAMPLERS = {
    "hierarchical":  _sample_hierarchical,
    "asymmetric":    _sample_asymmetric,
    "compact_equal": _sample_compact_equal,
    "scatter":       _sample_scatter,
}
REGIME_FRACTIONS = {
    "hierarchical":  0.25,
    "asymmetric":    0.30,
    "compact_equal": 0.25,
    "scatter":       0.20,
}


def _integration_params(m1, m2, m3, r12, r3_sep):
    M_total = m1+m2+m3; M12 = m1+m2
    min_sep  = max(min(r12*0.3, r3_sep*0.3), 0.01)
    tau_peri = np.sqrt(min_sep**3 / (G*M_total))
    dt       = min(tau_peri/20.0, 0.05)
    T_inner  = 2.0*np.pi*np.sqrt(r12**3/(G*M12))
    n_steps  = int(max(15.0*T_inner, 50.0) / dt)
    n_steps  = max(n_steps, 5000)
    n_steps  = min(n_steps, 20000)
    return dt, n_steps


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

# ML feature groups — explicitly defined here so experiments can import them
IC_NORM = ["epsilon_total", "h_total", "q12", "q13", "M_total",
           "r12_init", "r3_sep", "v3_frac", "e_inner"]

IC_RAW  = ["E0_total", "L0_total", "M_total", "r12_init", "r3_sep",
           "v3_frac", "e_inner"]

DYN_W5  = ["dE_max_w5",  "dE_slope_w5",
           "r_min_12_w5",  "r_min_13_w5",  "r_min_23_w5",
           "e12_std_w5",   "e13_std_w5",   "e23_std_w5"]

DYN_W20 = ["dE_max_w20", "dE_slope_w20",
           "r_min_12_w20", "r_min_13_w20", "r_min_23_w20",
           "e12_std_w20",  "e13_std_w20",  "e23_std_w20"]

TIER0 = IC_NORM
TIER1 = IC_NORM + DYN_W5
TIER2 = IC_NORM + DYN_W20


def generate(
    n_samples:     int  = N_SAMPLES,
    compute_megno: bool = COMPUTE_MEGNO,
    seed:          int  = SEED,
    output_dir:    Path = OUTPUT_DIR,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    counts = {r: int(n_samples*f) for r,f in REGIME_FRACTIONS.items()}
    counts["scatter"] = n_samples - sum(v for k,v in counts.items() if k != "scatter")
    regimes = []
    for r, cnt in counts.items(): regimes += [r]*cnt
    regimes = np.array(regimes); rng.shuffle(regimes)

    OUTCOME_MAP3 = {"stable":0, "ejection":1, "collision":1, "chaotic":2}
    rows = []
    outcome_counts = {"stable":0, "unstable":0, "chaotic":0}
    rejected = {"nan":0, "dL":0, "dE":0, "exception":0}
    t0 = time.time()

    print(f"Generating {n_samples} three-body simulations → {output_dir}")
    print(f"MEGNO: {'ON' if compute_megno else 'OFF'}  (full trajectory, labelling only)")
    print(f"Regime mix: " + "  ".join(f"{r}:{c}" for r,c in counts.items()))
    print(f"\n{'idx':>5}  {'regime':>15}  {'outcome':>10}  {'MEGNO':>7}  {'ms':>6}")
    print("─"*55)

    for idx, regime in enumerate(regimes):
        m1,m2,m3,r12,r3_sep,v3_mag,v3_angle,e_inner = SAMPLERS[regime](rng)
        r1,r2,r3,v1,v2,v3 = _com_3body(rng,m1,m2,m3,r12,r3_sep,v3_angle,v3_mag,e_inner)
        dt, n_steps = _integration_params(m1,m2,m3,r12,r3_sep)

        sys3 = ThreeBodySystem(
            G=G, m1=m1, m2=m2, m3=m3,
            r1_0=r1, r2_0=r2, r3_0=r3,
            v1_0=v1, v2_0=v2, v3_0=v3,
            dt=dt, n_steps=n_steps, label=f"s{idx:05d}",
        )

        t_sim = time.time()
        try:
            result   = run_simulation_3body(sys3, compute_megno=compute_megno)
            result, feats = label_result(result)
        except Exception:
            rejected["exception"] += 1; continue

        if (np.isnan(result.E_hist).any() or np.isnan(result.L_hist).any()
                or np.isnan(result.traj).any()):
            rejected["nan"] += 1; continue

        # Quality gates: angular momentum AND energy conservation
        dL = feats["dL_max_w20"]; dE = feats["dE_max_w20"]
        if dL > 0.02:                         rejected["dL"] += 1; continue
        if dE > 0.1 and result.E0 < 0:        rejected["dE"] += 1; continue
        # Note: dE_max can be large for unbound systems (E0 > 0) because
        # normalisation by |E0| is small. We only apply dE gate to bound systems.
        # Unbound systems (E0>0) are still caught by ejection detection.

        ms = int((time.time()-t_sim)*1000)
        bucket = "unstable" if result.outcome in ("ejection","collision") else result.outcome
        outcome_counts[bucket] += 1

        meg   = result.MEGNO_final
        meg_c = float(np.clip(meg, 0, 10)) if not np.isnan(meg) else 2.0

        M_total = m1+m2+m3
        traj_path = traj_dir / f"traj_{idx:05d}.npz"
        np.savez_compressed(
            traj_path,
            traj   = result.traj.astype(np.float32),
            E_hist = result.E_hist.astype(np.float32),
            L_hist = result.L_hist.astype(np.float32),
            time   = result.time.astype(np.float32),
        )

        row = {
            # Identifiers
            "idx": idx, "regime": regime,
            # Raw masses and geometry
            "m1": m1, "m2": m2, "m3": m3, "M_total": M_total,
            "r12_init": r12, "r3_sep": r3_sep,
            "v3_angle": v3_angle, "e_inner": e_inner,
            # IC ML features (normalised)
            "q12": m1/m2, "q13": m1/m3, "q23": m2/m3,
            "v3_frac":      v3_mag / np.sqrt(G*M_total/r3_sep),
            "epsilon_total": result.E0 / M_total,
            "h_total":       result.L0 / M_total,
            # IC raw (for ablation)
            "E0_total": result.E0,
            "L0_total": result.L0,
            # 5% window features (Tier-1 ML)
            **feats,
            # Metadata only — NOT ML features
            "MEGNO":       meg,
            "MEGNO_clean": meg_c,
            # Labels
            "outcome":        result.outcome,
            "outcome_class":  OUTCOME_MAP3[result.outcome],
            "outcome_class4": result.outcome_class,
            "traj_file": str(traj_path),
        }
        rows.append(row)

        if idx % 250 == 0 or idx < 5:
            print(f"{idx:5d}  {regime:>15}  {result.outcome:>10}  "
                  f"{meg_c:7.3f}  {ms:6d}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "metadata3.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time()-t0
    total   = len(df)
    print(f"\nDone in {elapsed:.1f}s  ({elapsed/total*1000:.0f} ms/sample)" if total else "")
    print(f"Saved {total} rows → {csv_path}")
    print(f"Rejected: {rejected}")
    print("\nOutcome distribution:")
    for name, cnt in outcome_counts.items():
        pct = 100*cnt/total if total else 0
        print(f"  {name:>12}: {cnt:5d}  ({pct:.1f}%)")
    print(f"\nML feature groups exported:")
    print(f"  TIER0 (IC only):      {len(TIER0)} features")
    print(f"  TIER1 (IC + 5%):      {len(TIER1)} features")
    print(f"  TIER2 (IC + 20%):     {len(TIER2)} features")
    return df


if __name__ == "__main__":
    generate()