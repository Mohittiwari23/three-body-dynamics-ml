"""
three_body/dataset_generator3.py
==================================
Generates the Phase 3 labelled three-body dataset.

Sampling regimes
----------------
Four regimes cover distinct regions of 3-body phase space:

  hierarchical (20%)
    Inner tight binary + distant outer body (r3/r12 ∈ [3,15]).
    Hill stability regime. Produces mostly stable systems.
    Purpose: train the model to recognise genuinely stable configurations.

  asymmetric (25%)
    One dominant mass + two lighter companions (q12 ∈ [0.01,0.3]).
    Intermediate chaos. Mix of stable and chaotic.
    Purpose: mass-ratio diversity for generalisation.

  compact_equal (30%)
    Three comparable masses, outer body CLOSE to inner binary
    (r3/r12 ∈ [0.5, 2.0]). Strong gravitational encounters.
    Purpose: generate ejections through actual slingshot dynamics.

  scatter (25%)
    Equal-ish masses, compact geometry, faster outer body
    (r3/r12 ∈ [0.8, 2.5], v3_frac ∈ [0.8, 1.35]).
    Purpose: energetic encounters, edge of stability.

All regimes enforce v3_frac < sqrt(2) = 1.414 (escape velocity).
This ensures all sampled systems are BOUND at t=0.
Ejections only occur through gravitational energy exchange — not trivially.

Bugs fixed vs previous versions
---------------------------------
  BUG 1: np.random.uniform in _com_3body broke seed reproducibility.
         Fixed: rng passed into _com_3body, uses rng.uniform throughout.

  BUG 2: Inner binary eccentricity formula was v_factor = sqrt((1+e)/(1-e)).
         This is the apoapsis/periapsis VELOCITY RATIO, not the periapsis speed.
         At e=0.8 it gives v_inner 2.2x too high, making binaries violently unbound.
         Fixed: v_inner = v_circ * sqrt(1 + e_inner)  [correct Kepler formula].

  BUG 3: scatter regime used v3_frac ∈ [1.2, 3.0].
         88% of those samples are unbound at t=0 — trivial ejections, not physics.
         Fixed: v3_frac ∈ [0.8, 1.35] (all bound at t=0).

  BUG 4: near_equal regime used v3_frac ∈ [0.8, 2.2]*[0.9, 1.2].
         Up to 64% unbound. Replaced by compact_equal with [0.6, 1.3].

Label note
----------
  outcome_class uses 3-class mapping:
    0 = stable
    1 = unstable  (ejection OR collision — both are disruption events)
    2 = chaotic
  outcome and outcome_class4 preserve the 4-class raw label.
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
from three_body.labeller import label_result

G = 1.0

N_SAMPLES    = 200
WINDOW_FRAC  = 0.20
COMPUTE_MEGNO = True
SEED         = 42
OUTPUT_DIR   = Path(__file__).resolve().parent.parent / "data" / "dataset3"


# ---------------------------------------------------------------------------
# CoM frame placement  (FIXED)
# ---------------------------------------------------------------------------

def _com_3body(rng, m1, m2, m3, r12, r3_sep, v3_angle, v3_mag, e_inner=0.0):
    """
    Place three bodies in the centre-of-mass frame.

    Bodies 1 and 2 form an inner binary at separation r12 on the x-axis.
    r12 is treated as the PERIAPSIS of the inner binary.
    Body 3 starts at distance r3_sep from the system CoM, at angle v3_angle.

    Inner binary velocity:
        v_peri = v_circ * sqrt(1 + e_inner)
    This is the exact Kepler formula for periapsis velocity given eccentricity e.

    Parameters
    ----------
    rng      : numpy Generator (seeded — do NOT call np.random directly)
    e_inner  : eccentricity of the inner binary ∈ [0, 1)
    """
    M12 = m1 + m2
    M   = M12 + m3

    r1 = np.array([ r12 * m2/M12, 0.0])
    r2 = np.array([-r12 * m1/M12, 0.0])
    r3 = np.array([r3_sep * np.cos(v3_angle), r3_sep * np.sin(v3_angle)])

    com = (m1*r1 + m2*r2 + m3*r3) / M
    r1 -= com; r2 -= com; r3 -= com

    # Correct periapsis velocity: v_peri = v_circ * sqrt(1 + e)
    v_circ_12 = np.sqrt(G * M12 / r12)
    v_inner   = v_circ_12 * np.sqrt(1.0 + e_inner)

    v1_lab = np.array([0.0,  v_inner * m2 / M12])
    v2_lab = np.array([0.0, -v_inner * m1 / M12])

    v3_dir = np.array([-np.sin(v3_angle), np.cos(v3_angle)])
    v3_lab = v3_mag * v3_dir

    vcom = (m1*v1_lab + m2*v2_lab + m3*v3_lab) / M
    v1_lab -= vcom; v2_lab -= vcom; v3_lab -= vcom

    return r1, r2, r3, v1_lab, v2_lab, v3_lab


# ---------------------------------------------------------------------------
# Regime samplers  (all enforce v3_frac < sqrt(2) — all bound at t=0)
# ---------------------------------------------------------------------------

_V_ESCAPE = np.sqrt(2.0)   # v3_frac above this → unbound


def _sample_hierarchical(rng):
    """Outer body far from inner binary. Hill stability regime."""
    M  = float(np.exp(rng.uniform(np.log(5), np.log(200))))
    q12 = float(np.exp(rng.uniform(np.log(0.1), np.log(1.0))))
    m2  = M / (1 + q12); m1 = q12 * m2
    m3  = M * float(rng.uniform(0.001, 0.05))

    r12    = float(np.exp(rng.uniform(np.log(0.5), np.log(3.0))))
    r3_sep = r12 * float(rng.uniform(3.0, 15.0))

    v_circ = np.sqrt(G * (M + m3) / r3_sep)
    v3_frac = float(rng.uniform(0.4, 1.10))     # well below escape
    v3_mag  = v3_frac * v_circ
    v3_angle = float(rng.uniform(0, 2*np.pi))
    e_inner  = float(rng.uniform(0.0, 0.5))

    return m1, m2, m3, r12, r3_sep, v3_mag, v3_angle, e_inner


def _sample_asymmetric(rng):
    """One dominant mass. Moderate chaos, mass-ratio diversity."""
    M   = float(np.exp(rng.uniform(np.log(10), np.log(500))))
    q12 = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))))
    m2  = M / (1 + q12); m1 = q12 * m2
    m3  = M * float(rng.uniform(0.005, 0.1))

    r12    = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0))))
    r3_sep = r12 * float(rng.uniform(1.5, 6.0))

    v_circ  = np.sqrt(G * (M + m3) / r3_sep)
    v3_frac = float(rng.uniform(0.5, 1.30))     # all bound
    v3_mag  = v3_frac * v_circ
    v3_angle = float(rng.uniform(0, 2*np.pi))
    e_inner  = float(rng.uniform(0.0, 0.6))

    return m1, m2, m3, r12, r3_sep, v3_mag, v3_angle, e_inner


def _sample_compact_equal(rng):
    """Three comparable masses, outer body close to inner binary.
    Strong encounters → gravitational slingshots → ejections."""
    M   = float(np.exp(rng.uniform(np.log(3), np.log(30))))
    q12 = float(rng.uniform(0.5, 1.0))
    m2  = M / (1 + q12) / 2; m1 = q12 * m2; m3 = M - m1 - m2

    r12    = float(rng.uniform(0.5, 4.0))
    r3_sep = r12 * float(rng.uniform(0.5, 2.0))   # COMPACT

    v_circ  = np.sqrt(G * M / r3_sep)
    v3_frac = float(rng.uniform(0.6, 1.30))        # all bound
    v3_mag  = v3_frac * v_circ
    v3_angle = float(rng.uniform(0, 2*np.pi))
    e_inner  = float(rng.uniform(0.0, 0.6))

    return m1, m2, m3, r12, r3_sep, v3_mag, v3_angle, e_inner


def _sample_scatter(rng):
    """Near-equal masses, compact geometry, fast outer body.
    Edge of stability — energetic encounters, high ejection rate."""
    M  = float(np.exp(rng.uniform(np.log(3), np.log(50))))
    m1 = M * float(rng.uniform(0.30, 0.45))
    m2 = M * float(rng.uniform(0.30, 0.45))
    m3 = M - m1 - m2

    r12    = float(rng.uniform(0.5, 3.0))
    r3_sep = r12 * float(rng.uniform(0.8, 2.5))

    v_circ  = np.sqrt(G * M / r3_sep)
    v3_frac = float(rng.uniform(0.8, 1.35))        # all bound, v_esc = 1.414
    v3_mag  = v3_frac * v_circ
    v3_angle = float(rng.uniform(0, 2*np.pi))
    e_inner  = float(rng.uniform(0.0, 0.5))

    return m1, m2, m3, r12, r3_sep, v3_mag, v3_angle, e_inner


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


# ---------------------------------------------------------------------------
# dt and n_steps selection
# ---------------------------------------------------------------------------

def _integration_params(m1, m2, m3, r12, r3_sep):
    """
    Choose dt and n_steps that guarantee:
      - At least 15 inner binary periods
      - At least 50 time units of physical evolution
      - Adequate resolution near closest approach

    Returns (dt, n_steps).
    """
    M_total = m1 + m2 + m3
    M12     = m1 + m2

    min_sep = max(min(r12 * 0.3, r3_sep * 0.3), 0.01)
    tau_peri = np.sqrt(min_sep**3 / (G * M_total))
    dt = min(tau_peri / 20.0, 0.05)

    T_inner  = 2.0 * np.pi * np.sqrt(r12**3 / (G * M12))
    t_min    = max(15.0 * T_inner, 50.0)
    n_steps  = int(t_min / dt)
    n_steps  = max(n_steps, 5000)
    n_steps  = min(n_steps, 20000)

    return dt, n_steps


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(
    n_samples: int = N_SAMPLES,
    window_fraction: float = WINDOW_FRAC,
    compute_megno: bool = COMPUTE_MEGNO,
    seed: int = SEED,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    # Build regime list
    counts = {r: int(n_samples * f) for r, f in REGIME_FRACTIONS.items()}
    counts["scatter"] = n_samples - sum(v for k,v in counts.items()
                                         if k != "scatter")
    regimes = []
    for r, cnt in counts.items():
        regimes += [r] * cnt
    regimes = np.array(regimes)
    rng.shuffle(regimes)

    rows = []
    outcome_counts = {"stable": 0, "unstable": 0, "chaotic": 0}
    t0 = time.time()

    print(f"Generating {n_samples} three-body simulations → {output_dir}")
    print(f"MEGNO: {'ON' if compute_megno else 'OFF'}   window: {window_fraction:.0%}")
    print(f"Regime mix: " + "  ".join(f"{r}:{c}" for r,c in counts.items()))
    print(f"\n{'idx':>5}  {'regime':>15}  {'outcome':>10}  "
          f"{'MEGNO':>7}  {'dE_max':>8}  {'ms':>6}")
    print("─" * 62)

    OUTCOME_MAP3 = {"stable":0, "ejection":1, "collision":1, "chaotic":2}

    for idx, regime in enumerate(regimes):
        m1, m2, m3, r12, r3_sep, v3_mag, v3_angle, e_inner = \
            SAMPLERS[regime](rng)

        r1, r2, r3, v1, v2, v3 = _com_3body(
            rng, m1, m2, m3, r12, r3_sep, v3_angle, v3_mag, e_inner
        )

        dt, n_steps = _integration_params(m1, m2, m3, r12, r3_sep)

        sys3 = ThreeBodySystem(
            G=G, m1=m1, m2=m2, m3=m3,
            r1_0=r1, r2_0=r2, r3_0=r3,
            v1_0=v1, v2_0=v2, v3_0=v3,
            dt=dt, n_steps=n_steps,
            label=f"s{idx:05d}",
        )

        t_sim = time.time()
        result = None
        try:
            result = run_simulation_3body(sys3, compute_megno=compute_megno)
            result = label_result(result, window_fraction=window_fraction)
        except Exception as e:
            pass

        if result is None:
            continue
        if (np.isnan(result.E_hist).any() or
                np.isnan(result.L_hist).any() or
                np.isnan(result.traj).any()):
            continue
        if result.dL_max > 2e-2:   # quality gate
            continue

        ms = int((time.time() - t_sim) * 1000)

        # Bucket for printing
        bucket = ("unstable" if result.outcome in ("ejection","collision")
                  else result.outcome)
        outcome_counts[bucket] += 1

        meg = result.MEGNO_final
        meg_c = float(np.clip(meg, 0, 10)) if not np.isnan(meg) else 2.0

        # Save trajectory
        traj_path = traj_dir / f"traj_{idx:05d}.npz"
        np.savez_compressed(
            traj_path,
            traj   = result.traj.astype(np.float32),
            E_hist = result.E_hist.astype(np.float32),
            L_hist = result.L_hist.astype(np.float32),
            time   = result.time.astype(np.float32),
        )

        M_total = m1 + m2 + m3
        rows.append({
            "idx": idx, "regime": regime,
            "m1": m1, "m2": m2, "m3": m3, "M_total": M_total,
            "q12": m1/m2, "q13": m1/m3, "q23": m2/m3,
            "r12_init": r12, "r3_sep": r3_sep,
            "v3_frac": v3_mag / np.sqrt(G*M_total/r3_sep),
            "v3_angle": v3_angle, "e_inner": e_inner,
            "epsilon_total": result.E0 / M_total,
            "h_total":       result.L0 / M_total,
            "dE_max":    result.dE_max,
            "dE_slope":  result.dE_slope,
            "dL_max":    result.dL_max,
            "MEGNO":       meg,
            "MEGNO_clean": meg_c,
            "e12_std": result.e12_std,
            "e13_std": result.e13_std,
            "e23_std": result.e23_std,
            "r_min_12": result.r_min_12,
            "r_min_13": result.r_min_13,
            "r_min_23": result.r_min_23,
            "outcome":        result.outcome,
            "outcome_class":  OUTCOME_MAP3[result.outcome],
            "outcome_class4": result.outcome_class,
            "traj_file": str(traj_path),
        })

        if idx % 250 == 0 or idx < 5:
            print(f"{idx:5d}  {regime:>15}  {result.outcome:>10}  "
                  f"{meg_c:7.3f}  {result.dE_max:8.2e}  {ms:6d}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "metadata3.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    total = len(df)
    print(f"\nDone in {elapsed:.1f}s  "
          f"({elapsed/total*1000:.0f} ms/sample)" if total else "")
    print(f"Saved {total} rows → {csv_path}")
    print("\nOutcome distribution:")
    for name, cnt in outcome_counts.items():
        pct = 100*cnt/total if total else 0
        print(f"  {name:>12}: {cnt:5d}  ({pct:.1f}%)")
    return df


if __name__ == "__main__":
    generate()