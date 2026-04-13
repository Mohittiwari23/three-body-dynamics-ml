"""
orbital_simulator/analysis.py
================================
Post-simulation analysis tools.

These functions operate on SimulationResult objects and produce
quantitative metrics beyond what the integrator computes.

Current capabilities
--------------------
- Simulation summary report (console output)
- Convergence study: run same physics at multiple dt values
- Orbit residual statistics

Roadmap hooks
-------------
- Lyapunov exponent estimation (chaos detection)
- Poincaré section extraction (once 3-body added)
- Minimum approach distance (conjunction detection)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .physics import (
    OrbitalSystem,
    SimulationResult,
    orbital_period,
)
from .integrator import run_simulation


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_summary(result: SimulationResult) -> None:
    """
    Print a structured physics summary to the console.

    Covers: orbit classification, conserved quantities, turning points,
    integrator quality metrics, and orbit residual statistics.
    """
    res = result
    r   = res.system

    sep = "─" * 58
    print(f"\n{'═' * 58}")
    print(f"  ORBITAL SIMULATION SUMMARY  ·  case: {r.label.upper()}")
    print(f"{'═' * 58}")

    print(f"\n  ORBIT CLASSIFICATION")
    print(sep)
    print(f"  Type          :  {res.orbit_type}")
    print(f"  Eccentricity  :  e = {res.e:.8f}")
    print(f"  Semi-latus    :  p = {res.p:.6e}  [length]")
    print(f"  Periapsis     :  r_min = {res.r_min:.6e}  [length]")
    rmax_str = f"{res.r_max:.6e}" if np.isfinite(res.r_max) else "∞  (unbound orbit)"
    print(f"  Apoapsis      :  r_max = {rmax_str}  [length]")
    if res.T_orb:
        print(f"  Period        :  T = {res.T_orb:.6e}  [time]")
    else:
        print(f"  Period        :  N/A  (parabolic or hyperbolic)")

    print(f"\n  CONSERVED QUANTITIES  (initial values)")
    print(sep)
    print(f"  Total energy  :  E₀ = {res.E0:.6e}")
    print(f"  Angular mom.  :  L₀ = {res.L0:.6e}")

    print(f"\n  INTEGRATOR QUALITY  (Velocity Verlet)")
    print(sep)
    print(f"  dt            :  {r.dt:.3e}  [time]")
    print(f"  Steps         :  {len(res.traj):,}")
    print(f"  Max |ΔE/scale|:  {res.max_energy_error:.3e}   ← should be < 1e-6")
    print(f"  Max |ΔL/L₀|   :  {res.max_momentum_error:.3e}   ← should be < 1e-10")

    if res.max_orbit_residual is not None:
        rel_res = res.max_orbit_residual / res.r_min
        print(f"\n  ORBIT RESIDUAL  |r_num − r_analytic(θ)| / r_min")
        print(sep)
        print(f"  Peak residual :  {rel_res:.3e} × r_min")
        print(f"  Mean residual :  {np.nanmean(res.residuals) / res.r_min:.3e} × r_min")
        quality = (
            "EXCELLENT — numerical and analytical orbits agree"   if rel_res < 1e-4 else
            "GOOD — minor numerical drift, framework is valid"    if rel_res < 1e-2 else
            "FAIR — consider reducing dt or checking the case"    if rel_res < 0.1  else
            "POOR — significant drift, check dt and periapsis"
        )
        print(f"  Quality       :  {quality}")

    print(f"\n{'═' * 58}\n")


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

@dataclass
class ConvergencePoint:
    """Single data point in a timestep convergence study."""
    dt             : float
    steps          : int
    max_energy_err : float
    max_momentum_err: float
    max_residual   : Optional[float]


def convergence_study(
    system: OrbitalSystem,
    dt_values: list[float],
    verbose: bool = True
) -> list[ConvergencePoint]:
    """
    Run the same physical system at multiple timestep sizes.

    This is the standard numerical analysis tool for demonstrating that
    the integrator converges at the expected rate (2nd order for Verlet).
    For a research demo, showing ~4× improvement in energy error when
    halving dt is compelling evidence of a correct implementation.

    Parameters
    ----------
    system    : base OrbitalSystem  (dt is overridden by dt_values)
    dt_values : list of timestep sizes to test

    Returns
    -------
    List of ConvergencePoint, one per dt value.
    """
    from dataclasses import replace
    points = []

    if verbose:
        print(f"\nConvergence study for case: {system.label}")
        print(f"{'dt':>12}  {'steps':>8}  {'max|ΔE|':>12}  {'max|ΔL|':>12}  {'max|Δr|/r_min':>14}")
        print("─" * 66)

    for dt in dt_values:
        sys_copy = replace(system, dt=dt)
        res = run_simulation(sys_copy)

        max_res = (res.max_orbit_residual / res.r_min
                   if res.max_orbit_residual is not None else None)

        pt = ConvergencePoint(
            dt              = dt,
            steps           = len(res.traj),
            max_energy_err  = res.max_energy_error,
            max_momentum_err= res.max_momentum_error,
            max_residual    = max_res,
        )
        points.append(pt)

        if verbose:
            res_str = f"{max_res:.3e}" if max_res is not None else "   N/A"
            print(f"  {dt:>10.4e}  {pt.steps:>8,}  {pt.max_energy_err:>12.3e}"
                  f"  {pt.max_momentum_err:>12.3e}  {res_str:>14}")

    if verbose:
        print()
        _check_convergence_order(points)

    return points


def _check_convergence_order(points: list[ConvergencePoint]) -> None:
    """
    Estimate and report the empirical convergence order from energy error.

    Velocity Verlet is 2nd order: halving dt should reduce energy error by ~4×.
    """
    if len(points) < 2:
        return
    print("  Empirical convergence order (energy error):")
    for i in range(1, len(points)):
        p0, p1 = points[i-1], points[i]
        if p0.max_energy_err > 0 and p1.max_energy_err > 0:
            ratio    = p0.dt / p1.dt
            err_ratio= p0.max_energy_err / p1.max_energy_err
            order    = np.log(err_ratio) / np.log(ratio)
            print(f"    dt {p0.dt:.2e} → {p1.dt:.2e}  :  "
                  f"error ratio = {err_ratio:.2f}×  →  order ≈ {order:.2f}")
    print("    (expected order ≈ 2.0 for Velocity Verlet)\n")


# ---------------------------------------------------------------------------
# MEGNO — Mean Exponential Growth factor of Nearby Orbits
# ---------------------------------------------------------------------------

def compute_MEGNO(system: OrbitalSystem, delta0: float = 1e-8) -> float:
    """
    Compute the MEGNO chaos indicator for a two-body system.

    Definition (Cincotta & Simó 2000)
    ----------------------------------
        <Y>(T) = (2/T) ∫₀ᵀ t · d(ln|w(t)|)/dt  dt

    where w(t) is the phase-space separation between the main trajectory
    and a shadow trajectory with initial offset delta0.

    Convergence values
    ------------------
        <Y> → 2        : regular (quasi-periodic) orbit
        <Y> → λt/2     : chaotic orbit  (grows linearly, λ = Lyapunov exponent)

    For 2-body (always regular): validates the integrator is correct when <Y> ≈ 2.
    For 3-body (future use): distinguishes stable from chaotic configurations.

    Algorithm
    ---------
    Runs two trajectories simultaneously — main and shadow — using the same
    Velocity Verlet scheme as the main integrator.  The shadow is periodically
    renormalised to prevent floating point overflow while preserving direction.

    Parameters
    ----------
    system  : OrbitalSystem to analyse
    delta0  : initial phase-space separation magnitude (default 1e-8)

    Returns
    -------
    Y_mean  : <Y>(T) at end of simulation
    """
    G, m1, m2, dt = system.G, system.m1, system.m2, system.dt
    steps = int(system.t_max / dt)

    # Main trajectory
    r1 = system.r1_0.copy()
    r2 = system.r2_0.copy()
    v1 = system.v1_0.copy()
    v2 = system.v2_0.copy()

    # Shadow trajectory: offset main by delta0 in r1[0]
    r1s = r1.copy();  r1s[0] += delta0
    r2s = r2.copy()
    v1s = v1.copy()
    v2s = v2.copy()

    def _accel(r_rel: np.ndarray, m_other: float) -> np.ndarray:
        d = np.linalg.norm(r_rel)
        return -G * m_other * r_rel / d**3

    W      = 0.0      # accumulated MEGNO integral
    w_prev = delta0   # previous phase-space separation
    t      = 0.0

    for _ in range(steps):
        t += dt

        # --- Advance main trajectory ---
        rr = r1 - r2
        a1  =  _accel(rr, m2);   a2  = -_accel(rr, m1)
        r1 += v1 * dt + 0.5 * a1 * dt**2
        r2 += v2 * dt + 0.5 * a2 * dt**2
        rn  = r1 - r2
        a1n =  _accel(rn, m2);   a2n = -_accel(rn, m1)
        v1 += 0.5 * (a1 + a1n) * dt
        v2 += 0.5 * (a2 + a2n) * dt

        # --- Advance shadow trajectory ---
        rs  = r1s - r2s
        as1  =  _accel(rs, m2);  as2  = -_accel(rs, m1)
        r1s += v1s * dt + 0.5 * as1 * dt**2
        r2s += v2s * dt + 0.5 * as2 * dt**2
        rsn  = r1s - r2s
        as1n =  _accel(rsn, m2); as2n = -_accel(rsn, m1)
        v1s += 0.5 * (as1 + as1n) * dt
        v2s += 0.5 * (as2 + as2n) * dt

        # --- Phase-space separation magnitude ---
        dpos = (r1s - r2s) - (r1 - r2)
        dvel = (v1s - v2s) - (v1 - v2)
        w    = np.sqrt(np.dot(dpos, dpos) + np.dot(dvel, dvel))

        # --- Accumulate MEGNO: W += t * d(ln w) ---
        if w > 0.0 and w_prev > 0.0:
            W += t * np.log(w / w_prev)   # t * d(ln w) per step

        w_prev = w

        # --- Renormalise shadow to avoid overflow ---
        if w > delta0 * 1e4:
            scale = delta0 / (w + 1e-300)
            r1s = r1 + dpos * scale
            r2s = r2.copy()
            v1s = v1 + dvel * scale
            v2s = v2.copy()
            w_prev = delta0

    # <Y>(T) = (2/T) * W  where W = Σ t_i · ln(w_i / w_{i-1})
    return 2.0 * W / t