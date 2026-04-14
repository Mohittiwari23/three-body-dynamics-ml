"""
three_body/dataset_generator3.py
==================================
Generates labelled three-body dataset for Phase 3 ML experiments.

Sampling strategy
-----------------
Systems are drawn across three architectural regimes:

  Hierarchical (40%) : m3 << m1 ~ m2
    Third body orbits a tight inner binary. Hill stability regime.
    Expected mix: stable ~ 60%, ejection ~ 30%, collision ~ 10%

  Asymmetric (35%)   : m1 >> m2, m3
    One dominant mass with two test particles. Chaotic regime.
    Expected mix: stable ~ 50%, ejection ~ 35%, chaotic ~ 15%

  Near-equal (25%)   : m1 ~ m2 ~ m3
    Classic chaotic three-body problem. Figure-8 lives here.
    Expected mix: stable ~ 20%, ejection ~ 50%, chaotic ~ 30%

Feature schema
--------------
  Initial conditions (normalised):
    q12, q13, M_total, r12_init, r3_sep, v3_frac

  Conservation diagnostics (early window):
    dE_max, dE_slope, dL_max

  Chaos indicator:
    MEGNO_clean  (clipped to [0, 10])

  Pair eccentricity variation (early window):
    e12_std, e13_std, e23_std

  Closest approaches (early window):
    r_min_12, r_min_13, r_min_23

  Label:
    outcome_class  (0=stable, 1=ejection, 2=collision, 3=chaotic)
    outcome        (string)
"""

from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from three_body.integrator3 import (
    ThreeBodySystem, run_simulation_3body, total_energy_3, total_Lz_3
)
from three_body.labeller import label_result, OUTCOME_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SAMPLES      = 200
N_STEPS        = 3000      # total integration steps
WINDOW_FRAC    = 0.20      # fraction used for feature extraction
COMPUTE_MEGNO  = True
SEED           = 42
G              = 1.0
R_COLL         = 0.05      # collision radius

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset3_body"


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _com_3body(m1, m2, m3, r12, r3_sep, v1, v2, v3_angle, v3_mag):
    """
    Place three bodies in the CoM frame.
    Bodies 1&2 form an inner binary at separation r12 on the x-axis.
    Body 3 starts at distance r3_sep from the CoM of 1&2, at angle v3_angle.
    """
    M12 = m1 + m2
    M   = M12 + m3

    # Inner binary positions (CoM of 1+2 at origin)
    r1 = np.array([ r12 * m2/M12, 0.0])
    r2 = np.array([-r12 * m1/M12, 0.0])

    # Body 3 position
    r3 = np.array([r3_sep * np.cos(v3_angle), r3_sep * np.sin(v3_angle)])

    # Shift to 3-body CoM
    com = (m1*r1 + m2*r2 + m3*r3) / M
    r1 -= com; r2 -= com; r3 -= com

    # --- ADD eccentricity ---
    e_inner = np.random.uniform(0.0, 0.8)

    v_circ_12 = np.sqrt(G * M12 / r12)
    v_factor = np.sqrt((1 + e_inner) / (1 - e_inner + 1e-8))

    v1_lab = np.array([0.0, v_circ_12 * v_factor * m2 / M12])
    v2_lab = np.array([0.0, -v_circ_12 * v_factor * m1 / M12])

    # Body 3 velocity: tangential to position vector, scaled
    v3_dir = np.array([-np.sin(v3_angle), np.cos(v3_angle)])
    v3_lab = v3_mag * v3_dir

    # Shift to 3-body CoM velocity frame
    v_com = (m1*v1_lab + m2*v2_lab + m3*v3_lab) / M
    v1_lab -= v_com; v2_lab -= v_com; v3_lab -= v_com

    return r1, r2, r3, v1_lab, v2_lab, v3_lab


def _sample_hierarchical(rng):
    """Inner binary + distant third body. Hill-stable regime."""
    M_total = float(np.exp(rng.uniform(np.log(5), np.log(200))))
    q12     = float(np.exp(rng.uniform(np.log(0.1), np.log(1.0))))
    m2      = M_total / (1 + q12)
    m1      = q12 * m2
    m3_frac = float(rng.uniform(0.001, 0.05))
    m3      = m3_frac * M_total / (1 - m3_frac)
    M_total += m3

    r12 = float(np.exp(rng.uniform(np.log(0.5), np.log(3.0))))
    r3_sep = r12 * float(rng.uniform(3.0, 15.0))

    v_circ_outer = np.sqrt(G * M_total / r3_sep)
    v3_frac = float(rng.uniform(0.4, 1.3))
    v3_mag  = v3_frac * v_circ_outer * rng.uniform(0.9, 1.2)

    v3_angle = float(rng.uniform(0, 2*np.pi))
    return m1, m2, m3, r12, r3_sep, v3_mag, v3_angle


def _sample_asymmetric(rng):
    """One dominant body, two lighter companions."""
    M_total = float(np.exp(rng.uniform(np.log(10), np.log(500))))
    q12     = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
    m2      = M_total / (1 + q12)
    m1      = q12 * m2
    m3      = M_total * float(rng.uniform(0.005, 0.1))
    M_total += m3

    r12    = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0))))
    r3_sep = r12 * float(rng.uniform(1.5, 8.0))

    v_circ = np.sqrt(G * M_total / r3_sep)
    v3_mag = v_circ * float(rng.uniform(0.5, 2.0)) * rng.uniform(0.9, 1.2)
    v3_angle = float(rng.uniform(0, 2*np.pi))
    return m1, m2, m3, r12, r3_sep, v3_mag, v3_angle


def _sample_near_equal(rng):
    """Three bodies of comparable mass — classic chaotic regime."""
    M_total = float(np.exp(rng.uniform(np.log(3), np.log(30))))
    q12     = float(rng.uniform(0.5, 1.0))
    m2      = M_total / (1 + q12) / 2
    m1      = q12 * m2
    m3      = M_total - m1 - m2

    r12    = float(rng.uniform(0.5, 4.0))
    r3_sep = r12 * float(rng.uniform(1.2, 5.0))

    v_circ = np.sqrt(G * M_total / r3_sep)
    v3_mag = v_circ * float(rng.uniform(0.8, 2.2)) * rng.uniform(0.9, 1.2)
    v3_angle = float(rng.uniform(0, 2*np.pi))
    return m1, m2, m3, r12, r3_sep, v3_mag, v3_angle

def _sample_scatter(rng):
    """Strong interaction / slingshot regime"""

    M_total = float(np.exp(rng.uniform(np.log(3), np.log(50))))
    m1 = M_total * rng.uniform(0.3, 0.5)
    m2 = M_total * rng.uniform(0.3, 0.5)
    m3 = M_total - m1 - m2

    r12 = float(rng.uniform(0.5, 3.0))

    # CRITICAL: very close approach
    r3_sep = r12 * float(rng.uniform(0.8, 2.5))

    v_circ = np.sqrt(G * M_total / r3_sep)

    # FAST → slingshot
    v3_mag = v_circ * float(rng.uniform(1.2, 3.0))

    v3_angle = float(rng.uniform(0, 2*np.pi))

    return m1, m2, m3, r12, r3_sep, v3_mag, v3_angle
# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(
    n_samples: int = N_SAMPLES,
    n_steps:   int = N_STEPS,
    window_fraction: float = WINDOW_FRAC,
    compute_megno: bool = COMPUTE_MEGNO,
    seed: int = SEED,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    # Regime assignment
    regime_counts = {
        "hierarchical": int(n_samples * 0.20),
        "asymmetric": int(n_samples * 0.25),
        "near_equal": int(n_samples * 0.25),
        "scatter": n_samples - int(n_samples * 0.20) - int(n_samples * 0.25) - int(n_samples * 0.25),
    }
    regimes = []
    for r, cnt in regime_counts.items():
        regimes += [r] * cnt
    regimes = np.array(regimes)
    rng.shuffle(regimes)

    samplers = {
        "hierarchical": _sample_hierarchical,
        "asymmetric": _sample_asymmetric,
        "near_equal": _sample_near_equal,
        "scatter": _sample_scatter,
    }

    rows = []
    t0   = time.time()
    print(f"Generating {n_samples} three-body simulations → {output_dir}")
    print(f"MEGNO: {'ON' if compute_megno else 'OFF'}  window: {window_fraction:.0%}")
    print(f"{'idx':>5}  {'regime':>14}  {'outcome':>10}  {'dE_max':>8}  {'MEGNO':>7}  {'time':>6}")
    print("─"*60)

    outcome_counts = {k: 0 for k in ["stable","unstable","chaotic"]}

    for idx, regime in enumerate(regimes):
        sampler = samplers[regime]
        m1, m2, m3, r12, r3_sep, v3_mag, v3_angle = sampler(rng)

        r1, r2, r3, v1, v2, v3 = _com_3body(
            m1, m2, m3, r12, r3_sep, None, None, v3_angle, v3_mag
        )

        # Choose dt based on minimum approach timescale
        min_sep = min(r12 * 0.5, r3_sep * 0.5, 0.5)
        tau = np.sqrt(min_sep**3 / (G * (m1+m2+m3)))
        dt = min(tau / 20.0, 0.05)

        # --- NEW: physics-aware integration length ---
        M12 = m1 + m2
        T_inner = 2 * np.pi * np.sqrt(r12 ** 3 / (G * M12))

        min_time = 10.0  # minimum physical time
        n_steps_dyn = max(3000, int(20 * T_inner / dt), int(min_time / dt))
        n_steps_dyn = min(n_steps_dyn, 20000)

        sys3 = ThreeBodySystem(
            G=G, m1=m1, m2=m2, m3=m3,
            r1_0=r1, r2_0=r2, r3_0=r3,
            v1_0=v1, v2_0=v2, v3_0=v3,
            dt=dt, n_steps=n_steps_dyn,
            label=f"sample_{idx:05d}",
        )

        t_sim = time.time()
        try:
            result = run_simulation_3body(sys3, compute_megno=compute_megno)
            result = label_result(result, window_fraction=window_fraction)
            if result.outcome == "ejection" and result.ejection_step is not None:
                cut = min(result.ejection_step + 100, len(result.traj))
                result.traj = result.traj[:cut]
                result.E_hist = result.E_hist[:cut]
                result.L_hist = result.L_hist[:cut]
                result.time = result.time[:cut]
        except Exception as e:
            # Rare: bodies collide in step 0 or other numerical failure
            print(f"[ERROR] sample {idx}: {e}")
            result = None

        # Reject failed simulations
        if result is None:
            continue

        # Reject NaN / corrupted simulations
        if (np.isnan(result.E_hist).any() or np.isnan(result.L_hist).any() or np.isnan(result.traj).any()):
            continue

        # Quality filter (Phase 2 lesson — angular momentum conservation)
        if result.dL_max > 1e-2:
            continue

        sim_time = time.time() - t_sim
        if result.outcome in ["ejection", "collision"]:
            outcome_counts["unstable"] += 1
        else:
            outcome_counts[result.outcome] += 1

        # MEGNO cleaning
        meg = result.MEGNO_final
        meg_clean = float(np.clip(meg, 0, 10)) if not np.isnan(meg) else 2.0

        # Save trajectory (compressed)
        traj_path = traj_dir / f"traj_{idx:05d}.npz"
        np.savez_compressed(
            traj_path,
            traj   = result.traj.astype(np.float32),
            E_hist = result.E_hist.astype(np.float32),
            L_hist = result.L_hist.astype(np.float32),
            time   = result.time.astype(np.float32),
        )

        # Initial condition features
        M_total = m1 + m2 + m3
        q12 = m1/m2; q13 = m1/m3; q23 = m2/m3
        v_circ_12 = np.sqrt(G*(m1+m2)/r12)
        epsilon_total = result.E0 / (m1+m2+m3)  # specific energy (rough)
        h_total       = result.L0 / (m1+m2+m3)  # specific L (rough)

        # --- 3-class mapping ---
        outcome_map_3 = {
            "stable": 0,
            "ejection": 1,
            "collision": 1,
            "chaotic": 2
        }

        rows.append({
            "idx": idx,
            "regime": regime,

            # Masses
            "m1": m1, "m2": m2, "m3": m3,
            "M_total": M_total,
            "q12": q12, "q13": q13, "q23": q23,

            # Geometry
            "r12_init": r12,
            "r3_sep": r3_sep,
            "v3_frac": v3_mag / np.sqrt(G * M_total / r3_sep),
            "v3_angle": v3_angle,

            # Physics (normalized)
            "epsilon_total": epsilon_total,
            "h_total": h_total,

            # Diagnostics
            "dE_max": result.dE_max,
            "dE_slope": result.dE_slope,
            "dL_max": result.dL_max,

            # Chaos
            "MEGNO": result.MEGNO_final,
            "MEGNO_clean": meg_clean,

            # Pair features
            "e12_std": result.e12_std,
            "e13_std": result.e13_std,
            "e23_std": result.e23_std,

            "r_min_12": result.r_min_12,
            "r_min_13": result.r_min_13,
            "r_min_23": result.r_min_23,

            # --- LABELS ---
            "outcome": result.outcome,  # raw
            "outcome_class": outcome_map_3[result.outcome],  # 3-class
            "outcome_class4": result.outcome_class,  # original

            # File
            "traj_file": str(traj_path),
        })
        if idx % 200 == 0 or idx < 5:
            meg_s = f"{meg_clean:7.3f}" if not np.isnan(meg) else "    N/A"
            print(f"{idx:5d}  {regime:>14}  {result.outcome:>10}  "
                  f"{result.dE_max:8.2e}  {meg_s}  {sim_time:.2f}s")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "gen_data-200.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    if len(df) > 0:
        print(f"\nDone in {elapsed:.1f}s  ({elapsed / len(df) * 1000:.1f} ms/sample)")
    else:
        print(f"\nDone in {elapsed:.1f}s  (no valid samples)")
    print(f"Saved: {csv_path}  ({len(df)} rows)")
    print("\nOutcome distribution:")
    for name, cnt in outcome_counts.items():
        pct = 100*cnt/len(df) if len(df)>0 else 0
        print(f"  {name:>12}: {cnt:5d}  ({pct:.1f}%)")
    return df


if __name__ == "__main__":
    generate()
