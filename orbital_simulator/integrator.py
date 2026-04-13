"""
orbital_simulator/integrator.py
================================
Numerical time integration for the two-body gravitational problem.

Integrator: Velocity Verlet (Störmer-Verlet)
--------------------------------------------
This is a *symplectic* integrator, meaning it exactly conserves a
modified (shadow) Hamiltonian that is close to the true one.  The
consequences for orbital mechanics are:

  - Energy error stays BOUNDED over long integrations (does not grow
    secularly like Runge-Kutta methods do for oscillatory problems).
  - Angular momentum is conserved to machine precision because the
    force is always central (radial), which is an exact symmetry that
    the Verlet scheme respects.

Algorithm per step
------------------
    1. Compute acceleration at current position   a_n
    2. Advance position:  r_{n+1} = r_n + v_n dt + ½ a_n dt²
    3. Compute acceleration at new position       a_{n+1}
    4. Advance velocity:  v_{n+1} = v_n + ½(a_n + a_{n+1}) dt

Why this matters for the roadmap
---------------------------------
For three-body / perturbed systems, the integrator will be extended to
accept an arbitrary force function f(r1, r2, ...) rather than pure
Newtonian gravity.  The structure here is designed for that extension:
`_gravitational_acceleration` is isolated and easily replaced.

Future upgrade path: RK4 comparison, symplectic integrator order study,
adaptive dt based on local force gradient near periapsis.
"""

from __future__ import annotations

import numpy as np
from typing import Callable

from .physics import (
    OrbitalSystem,
    SimulationResult,
    total_energy,
    angular_momentum_z,
    orbital_invariants,
    orbital_period,
    classify_orbit,
    turning_points,
    energy_scale,
    compute_orbit_residuals,
    reduced_mass,
)


# ---------------------------------------------------------------------------
# Force law — isolated for easy extension (perturbations, 3-body, etc.)
# ---------------------------------------------------------------------------

def _gravitational_acceleration(
    r_rel: np.ndarray,
    G: float,
    mass_attractor: float
) -> np.ndarray:
    """
    Gravitational acceleration on a body due to a point mass.

        a = -G M r̂ / r²  =  -G M r_vec / r³

    Parameters
    ----------
    r_rel          : position vector pointing FROM attractor TO body
    G              : gravitational constant
    mass_attractor : mass of the attracting body

    Notes
    -----
    Perturbation forces (J2 oblateness, solar pressure, drag) would be
    added as additional acceleration terms in a future `perturbed_acceleration`
    function, keeping this baseline function unchanged.
    """
    r_mag = np.linalg.norm(r_rel)
    return -G * mass_attractor * r_rel / r_mag**3


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(system: OrbitalSystem) -> SimulationResult:
    """
    Run the two-body simulation using Velocity Verlet integration.

    Parameters
    ----------
    system : OrbitalSystem
        Fully specified physical + numerical system parameters.

    Returns
    -------
    SimulationResult
        Complete time-history of the simulation with all physics metrics
        pre-computed.

    Design notes
    ------------
    - Physics state (r1, r2, v1, v2) is kept in the lab frame.
    - Conservation quantities are evaluated from the lab-frame state at
      every step — they are diagnostic outputs, not physics inputs.
    - The relative position traj[i] = r1 - r2 is stored separately
      because all orbital geometry is defined in the relative frame.
    """
    # Unpack system for local use (avoids repeated attribute lookups in loop)
    G  = system.G
    m1, m2 = system.m1, system.m2
    r1 = system.r1_0.copy()
    r2 = system.r2_0.copy()
    v1 = system.v1_0.copy()
    v2 = system.v2_0.copy()
    dt = system.dt

    # --- Compute initial conserved quantities ---
    E0 = total_energy(r1, r2, v1, v2, m1, m2, G)
    L0 = angular_momentum_z(r1, r2, v1, v2, m1, m2)
    e, p = orbital_invariants(E0, L0, m1, m2, G)
    r_min, r_max = turning_points(e, p)
    T_orb = orbital_period(e, p, m1, m2, G)
    orbit_type = classify_orbit(e)

    # If orbital period is known, simulate exactly N full revolutions
    # (set by the caller via system.t_max, which is pre-scaled in cases.py)
    steps = max(1, int(system.t_max / dt))

    # --- Pre-allocate output arrays (much faster than appending) ---
    traj   = np.empty((steps, 2), dtype=np.float64)
    E_hist = np.empty(steps,      dtype=np.float64)
    L_hist = np.empty(steps,      dtype=np.float64)

    # --- Integration loop ---
    for i in range(steps):
        # Step 1: Accelerations at current position
        r_rel = r1 - r2
        a1 =  _gravitational_acceleration(r_rel, G, m2)
        a2 = -_gravitational_acceleration(r_rel, G, m1)

        # Step 2: Update positions (Verlet position step)
        r1 = r1 + v1 * dt + 0.5 * a1 * dt**2
        r2 = r2 + v2 * dt + 0.5 * a2 * dt**2

        # Step 3: Accelerations at new position
        r_rel_new = r1 - r2
        a1n =  _gravitational_acceleration(r_rel_new, G, m2)
        a2n = -_gravitational_acceleration(r_rel_new, G, m1)

        # Step 4: Update velocities (Verlet velocity step)
        v1 = v1 + 0.5 * (a1 + a1n) * dt
        v2 = v2 + 0.5 * (a2 + a2n) * dt

        # Record state
        traj[i]   = r_rel_new
        E_hist[i] = total_energy(r1, r2, v1, v2, m1, m2, G)
        L_hist[i] = angular_momentum_z(r1, r2, v1, v2, m1, m2)

    # --- Post-process conservation metrics ---
    mu0     = reduced_mass(m1, m2)
    v_rel_0 = system.v1_0 - system.v2_0
    KE0     = 0.5 * mu0 * np.dot(v_rel_0, v_rel_0)
    E_sc    = energy_scale(E0, KE0)

    dE = (E_hist - E0) / E_sc
    dL = (L_hist - L0) / abs(L0)

    # --- Orbit residuals: numerical vs. analytical ---
    residuals = compute_orbit_residuals(traj, p, e)

    time_arr = np.linspace(0.0, system.t_max, steps)

    return SimulationResult(
        traj       = traj,
        E_hist     = E_hist,
        L_hist     = L_hist,
        time       = time_arr,
        E0         = E0,
        L0         = L0,
        e          = e,
        p          = p,
        r_min      = r_min,
        r_max      = r_max,
        T_orb      = T_orb,
        orbit_type = orbit_type,
        dE         = dE,
        dL         = dL,
        residuals  = residuals,
        system     = system,
    )
