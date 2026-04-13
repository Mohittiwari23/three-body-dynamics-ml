"""
ml/dataset_generator.py
========================
Generates the labelled two-body orbital dataset for ML experiments.

Design
------
All samples are drawn in the true CoM frame.  Eccentricity labels are
computed analytically from (E₀, L₀) — no simulation error enters the
label.  The simulation only produces trajectory and dynamical features.

Sampling strategy
-----------------
  alpha = v_perp / v_circ controls eccentricity:
    Circular    :  alpha ∈ [0.990, 1.010]
    Elliptical  :  alpha ∈ [0.050, 0.989] ∪ [1.011, √2·0.999]
    Parabolic   :  alpha ∈ [√2·0.9999, √2·1.0003]   (tight band, E→0)
    Hyperbolic  :  alpha ∈ [√2·1.001, 2.500]

  r0           :  log-uniform  [1.0,   20.0]
  M_total      :  log-uniform  [10,    1000]   (G=1 units)
  q = m1/m2    :  log-uniform  [0.001, 1.0]
  omega        :  uniform      [0,     2π)     (periapsis rotation angle)

Feature schema
--------------
  Physics (scale-invariant, rotation-invariant):
    E0, L0, mu, epsilon, h, e, p, a, r_min, r_max, T_orb, T_norm, r_ratio

  Raw inputs:
    m1, m2, M_total, q, r0, v_perp, alpha, omega

  Dynamical (novel contribution — from short integration):
    dE_max, dE_slope, dL_max, MEGNO, residual_max, residual_mean, e_inst_std

  Trajectory summaries:
    traj_r_mean, traj_r_std, traj_r_min_num, traj_r_max_num,
    traj_vr_max, traj_vt_mean

  Labels:
    orbit_class  (0=circular 1=elliptical 2=parabolic 3=hyperbolic)
    orbit_name
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

# Make orbital_simulator importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orbital_simulator.physics import (
    reduced_mass, orbital_invariants, orbital_period,
    turning_points, compute_orbit_residuals, energy_scale,
)

# ---------------------------------------------------------------------------
# Configuration — edit these to change dataset size / balance
# ---------------------------------------------------------------------------

N_SAMPLES    = 2000          # total simulations to run
N_STEPS      = 3000          # fixed trajectory length (steps per simulation)
COMPUTE_MEGNO = True         # False = ~3× faster but no MEGNO feature
SEED         = 42
G            = 1.0           # normalised units

FRACTIONS = {
    "circular":   0.10,
    "elliptical": 0.55,
    "parabolic":  0.10,
    "hyperbolic": 0.25,
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset"


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _sample_alpha(regime: str, rng: np.random.Generator) -> float:
    sqrt2 = np.sqrt(2.0)
    if regime == "circular":
        return float(rng.uniform(0.990, 1.010))
    if regime == "elliptical":
        if rng.random() < 0.5:
            return float(rng.uniform(0.050, 0.989))
        return float(rng.uniform(1.011, sqrt2 * 0.999))
    if regime == "parabolic":
        return float(rng.uniform(sqrt2 * 0.9999, sqrt2 * 1.0003))
    # hyperbolic
    return float(rng.uniform(sqrt2 * 1.001, 2.500))


def _com_ics(
    m1: float, m2: float, r0: float, v_perp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M  = m1 + m2
    r1 = np.array([ r0 * m2 / M,  0.0])
    r2 = np.array([-r0 * m1 / M,  0.0])
    v1 = np.array([0.0,  v_perp * m2 / M])
    v2 = np.array([0.0, -v_perp * m1 / M])
    return r1, r2, v1, v2


def _rotate(vec: np.ndarray, omega: float) -> np.ndarray:
    c, s = np.cos(omega), np.sin(omega)
    return np.array([c * vec[0] - s * vec[1],
                     s * vec[0] + c * vec[1]])


def _build_regime_list(n: int, fractions: dict, rng: np.random.Generator) -> list[str]:
    regimes = []
    for regime, frac in fractions.items():
        regimes += [regime] * int(n * frac)
    while len(regimes) < n:
        regimes.append("elliptical")
    arr = np.array(regimes)
    rng.shuffle(arr)
    return list(arr[:n])


# ---------------------------------------------------------------------------
# Core simulation — single pass, computes all features
# ---------------------------------------------------------------------------

def _simulate(
    m1: float, m2: float,
    r1_0: np.ndarray, r2_0: np.ndarray,
    v1_0: np.ndarray, v2_0: np.ndarray,
    dt: float, n_steps: int,
    compute_megno: bool = True,
    delta0: float = 1e-8,
) -> dict:
    """
    Velocity Verlet integration.  Computes all dynamical features in one pass.

    Returns a dict with trajectory arrays and scalar summary statistics.
    Velocity history is tracked inline to compute instantaneous eccentricity,
    radial velocity, and tangential velocity without a second pass.
    """
    r1 = r1_0.copy();  r2 = r2_0.copy()
    v1 = v1_0.copy();  v2 = v2_0.copy()
    mu = m1 * m2 / (m1 + m2)
    k  = G * m1 * m2

    traj   = np.empty((n_steps, 2))
    E_hist = np.empty(n_steps)
    L_hist = np.empty(n_steps)
    e_inst = np.empty(n_steps)
    vr_arr = np.empty(n_steps)
    vt_arr = np.empty(n_steps)

    # Shadow trajectory for MEGNO
    if compute_megno:
        r1s = r1.copy();  r1s[0] += delta0
        r2s = r2.copy()
        v1s = v1.copy();  v2s = v2.copy()
        W_meg  = 0.0
        w_prev = delta0
        t_meg  = 0.0

    def _accel(r_rel: np.ndarray, m_other: float) -> np.ndarray:
        d = np.linalg.norm(r_rel)
        return -G * m_other * r_rel / d**3

    for i in range(n_steps):
        r_rel = r1 - r2
        a1 =  _accel(r_rel, m2);  a2 = -_accel(r_rel, m1)
        r1 += v1 * dt + 0.5 * a1 * dt**2
        r2 += v2 * dt + 0.5 * a2 * dt**2
        rn = r1 - r2
        a1n =  _accel(rn, m2);  a2n = -_accel(rn, m1)
        v1 += 0.5 * (a1 + a1n) * dt
        v2 += 0.5 * (a2 + a2n) * dt

        rn_mag = np.linalg.norm(rn)
        vr = r1 - r2;  vv = v1 - v2
        KE = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)
        PE = -k / rn_mag
        E_i = KE + PE
        L_i = mu * (vr[0] * vv[1] - vr[1] * vv[0])

        traj[i]   = rn
        E_hist[i] = E_i
        L_hist[i] = L_i

        # Instantaneous eccentricity
        disc = 1.0 + 2.0 * E_i * L_i**2 / (mu * k**2)
        e_inst[i] = np.sqrt(max(0.0, disc))

        # Radial and tangential velocity components
        r_unit = rn / rn_mag if rn_mag > 0 else np.array([1.0, 0.0])
        t_unit = np.array([-r_unit[1], r_unit[0]])
        vr_arr[i] = abs(np.dot(vv, r_unit))
        vt_arr[i] = abs(np.dot(vv, t_unit))

        # MEGNO shadow advance
        if compute_megno:
            t_meg += dt
            rs = r1s - r2s
            as1 =  _accel(rs, m2);  as2 = -_accel(rs, m1)
            r1s += v1s * dt + 0.5 * as1 * dt**2
            r2s += v2s * dt + 0.5 * as2 * dt**2
            rsn = r1s - r2s
            as1n =  _accel(rsn, m2);  as2n = -_accel(rsn, m1)
            v1s += 0.5 * (as1 + as1n) * dt
            v2s += 0.5 * (as2 + as2n) * dt

            dpos = (r1s - r2s) - (r1 - r2)
            dvel = (v1s - v2s) - (v1 - v2)
            w    = np.sqrt(np.dot(dpos, dpos) + np.dot(dvel, dvel))
            if w > 0 and w_prev > 0:
                W_meg += t_meg * np.log(w / w_prev)
            w_prev = w
            if w > delta0 * 1e4:
                scale = delta0 / (w + 1e-300)
                r1s = r1 + dpos * scale;  r2s = r2.copy()
                v1s = v1 + dvel * scale;  v2s = v2.copy()
                w_prev = delta0

    megno = 2.0 * W_meg / (n_steps * dt) if compute_megno else np.nan

    return {
        "traj":    traj,
        "E_hist":  E_hist,
        "L_hist":  L_hist,
        "e_inst":  e_inst,
        "vr_arr":  vr_arr,
        "vt_arr":  vt_arr,
        "megno":   megno,
    }


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(
    n_samples: int = N_SAMPLES,
    n_steps:   int = N_STEPS,
    compute_megno: bool = COMPUTE_MEGNO,
    seed: int  = SEED,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Generate the full dataset and save to disk.

    Returns the metadata DataFrame.
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)

    regimes = _build_regime_list(n_samples, FRACTIONS, rng)
    rows    = []

    t0 = time.time()
    print(f"Generating {n_samples} simulations → {output_dir}")
    print(f"MEGNO: {'ON' if compute_megno else 'OFF'}   N_STEPS: {n_steps}")
    print(f"{'idx':>5}  {'regime':>12}  {'e':>8}  {'alpha':>6}  {'q':>7}  {'omega':>6}  "
          f"{'dE_max':>8}  {'MEGNO':>7}")
    print("─" * 72)

    for idx, regime in enumerate(regimes):

        # --- Sample parameters ---
        M_total = float(np.exp(rng.uniform(np.log(10),    np.log(1000))))
        q       = float(np.exp(rng.uniform(np.log(0.001), np.log(1.0))))
        m2      = M_total / (1.0 + q)
        m1      = q * m2
        r0      = float(np.exp(rng.uniform(np.log(1.0),   np.log(20.0))))
        omega   = float(rng.uniform(0.0, 2.0 * np.pi))
        alpha   = _sample_alpha(regime, rng)

        v_circ  = np.sqrt(G * M_total / r0)
        v_perp  = alpha * v_circ

        # --- CoM initial conditions ---
        r1_0, r2_0, v1_0, v2_0 = _com_ics(m1, m2, r0, v_perp)

        # --- Rotate by omega (randomises periapsis direction) ---
        r1_0 = _rotate(r1_0, omega);  r2_0 = _rotate(r2_0, omega)
        v1_0 = _rotate(v1_0, omega);  v2_0 = _rotate(v2_0, omega)

        # --- Analytical invariants (exact labels, no simulation error) ---
        mu0 = reduced_mass(m1, m2)
        k0  = G * m1 * m2
        KE0 = 0.5 * m1 * np.dot(v1_0, v1_0) + 0.5 * m2 * np.dot(v2_0, v2_0)
        PE0 = -k0 / np.linalg.norm(r1_0 - r2_0)
        E0  = KE0 + PE0
        r_rel0 = r1_0 - r2_0
        v_rel0 = v1_0 - v2_0
        L0  = mu0 * (r_rel0[0] * v_rel0[1] - r_rel0[1] * v_rel0[0])

        e, p   = orbital_invariants(E0, L0, m1, m2, G)
        r_min, r_max = turning_points(e, p)
        T_orb  = orbital_period(e, p, m1, m2, G)

        # Derived invariants
        a       = p / (1.0 - e**2) if e < 0.9999 else np.nan
        epsilon = E0 / mu0                         # specific energy
        h       = L0 / mu0                         # specific angular momentum
        T_norm  = T_orb / np.sqrt(r0**3 / (G * M_total)) if T_orb else np.nan
        r_ratio = r_min / r_max if (np.isfinite(r_max) and r_max > 0) else np.nan
        KE0_rel = 0.5 * mu0 * np.dot(v_rel0, v_rel0)
        E_sc    = energy_scale(E0, KE0_rel)

        # Orbit classification
        orbit_class = (0 if e < 1e-4 else
                       1 if e < 0.9999 else
                       2 if e < 1.005 else 3)
        orbit_name  = ["circular", "elliptical", "parabolic", "hyperbolic"][orbit_class]

        # --- Choose dt and run ---
        # dt sized to be <<  periapsis dynamical timescale
        tau_peri = np.sqrt(r_min**3 / (G * M_total))
        dt       = min(tau_peri / 30.0, 0.1)

        sim = _simulate(m1, m2, r1_0, r2_0, v1_0, v2_0,
                        dt, n_steps, compute_megno=compute_megno)

        traj   = sim["traj"]
        E_hist = sim["E_hist"]
        L_hist = sim["L_hist"]
        e_inst = sim["e_inst"]
        vr_arr = sim["vr_arr"]
        vt_arr = sim["vt_arr"]

        # --- Conservation diagnostics ---
        dE    = (E_hist - E0) / E_sc
        dL    = (L_hist - L0) / abs(L0)
        dE_max   = float(np.max(np.abs(dE)))
        dL_max   = float(np.max(np.abs(dL)))

        # Linear slope of dE(t) — secular drift indicator
        t_arr = np.arange(n_steps) * dt
        dE_slope = float(np.polyfit(t_arr, dE, 1)[0])

        # --- Orbit residuals (only meaningful for bound orbits) ---
        residuals = compute_orbit_residuals(traj, p, e)
        finite_res = residuals[np.isfinite(residuals)]
        if len(finite_res) > 0 and e < 0.9999:
            residual_max  = float(np.max(finite_res) / r_min)
            residual_mean = float(np.mean(finite_res) / r_min)
        else:
            residual_max  = np.nan
            residual_mean = np.nan

        # --- Trajectory summary stats ---
        r_norms = np.linalg.norm(traj, axis=1)
        e_inst_std   = float(np.std(e_inst))
        traj_r_mean  = float(np.mean(r_norms))
        traj_r_std   = float(np.std(r_norms))
        traj_r_min_n = float(np.min(r_norms))
        traj_r_max_n = float(np.max(r_norms))
        traj_vr_max  = float(np.max(vr_arr))
        traj_vt_mean = float(np.mean(vt_arr))

        # --- Save trajectory ---
        traj_path = traj_dir / f"traj_{idx:05d}.npz"
        np.savez_compressed(
            traj_path,
            traj   = traj.astype(np.float32),
            E_hist = E_hist.astype(np.float32),
            L_hist = L_hist.astype(np.float32),
            time   = (t_arr).astype(np.float32),
            dt     = np.float32(dt),
        )

        # --- Metadata row ---
        rows.append({
            # Index
            "idx":          idx,
            # Raw inputs
            "m1":           m1,
            "m2":           m2,
            "M_total":      M_total,
            "q":            q,
            "G":            G,
            "r0":           r0,
            "v_perp":       v_perp,
            "alpha":        alpha,
            "omega":        omega,
            "dt":           dt,
            "n_steps":      n_steps,
            # Analytical invariants (exact labels)
            "E0":           E0,
            "L0":           L0,
            "mu":           mu0,
            "epsilon":      epsilon,
            "h":            h,
            "e":            e,
            "p":            p,
            "a":            a,
            "r_min":        r_min,
            "r_max":        r_max if np.isfinite(r_max) else np.nan,
            "T_orb":        T_orb if T_orb else np.nan,
            "T_norm":       T_norm,
            "r_ratio":      r_ratio,
            # Classification label
            "orbit_class":  orbit_class,
            "orbit_name":   orbit_name,
            # Dynamical features
            "dE_max":       dE_max,
            "dE_slope":     dE_slope,
            "dL_max":       dL_max,
            "MEGNO":        sim["megno"],
            "residual_max": residual_max,
            "residual_mean":residual_mean,
            "e_inst_std":   e_inst_std,
            # Trajectory summaries
            "traj_r_mean":  traj_r_mean,
            "traj_r_std":   traj_r_std,
            "traj_r_min_num": traj_r_min_n,
            "traj_r_max_num": traj_r_max_n,
            "traj_vr_max":  traj_vr_max,
            "traj_vt_mean": traj_vt_mean,
            # Trajectory file
            "traj_file":    str(traj_path),
        })

        if idx % 100 == 0 or idx < 5:
            meg_str = f"{sim['megno']:7.3f}" if not np.isnan(sim["megno"]) else "    N/A"
            print(f"{idx:5d}  {orbit_name:>12}  {e:8.5f}  {alpha:6.3f}  "
                  f"{q:7.4f}  {omega:6.3f}  {dE_max:8.2e}  {meg_str}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  ({elapsed / n_samples * 1000:.1f} ms/sample)")
    print(f"Saved: {csv_path}")
    print(f"Saved: {traj_dir}/ ({n_samples} .npz files)")
    _print_dataset_summary(df)
    return df


def _print_dataset_summary(df: pd.DataFrame) -> None:
    print("\n── Dataset Summary ─────────────────────────────────")
    print("Class distribution:")
    for name, cnt in df["orbit_name"].value_counts().items():
        print(f"  {name:>12}: {cnt:4d}  ({100*cnt/len(df):.1f}%)")
    print("\nMass ratio (q) distribution:")
    for lo, hi in [(0, 0.01), (0.01, 0.1), (0.1, 0.4), (0.4, 1.0)]:
        cnt = ((df["q"] > lo) & (df["q"] <= hi)).sum()
        print(f"  q ∈ ({lo:.3f}, {hi:.3f}]: {cnt:4d}")
    print("\nConservation quality:")
    print(f"  Max  |ΔE/scale| : {df['dE_max'].max():.2e}")
    print(f"  Mean |ΔE/scale| : {df['dE_max'].mean():.2e}")
    print(f"  Max  |ΔL/L₀|   : {df['dL_max'].max():.2e}")
    if "MEGNO" in df.columns:
        finite_meg = df["MEGNO"].dropna()
        print(f"  MEGNO mean/std  : {finite_meg.mean():.3f} ± {finite_meg.std():.3f}")
    print("────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    generate()