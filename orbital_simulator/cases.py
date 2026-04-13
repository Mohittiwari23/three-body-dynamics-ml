"""
orbital_simulator/cases.py
============================
Predefined physical system configurations for the two-body simulator.

Design principle
----------------
Each case defines the *physics* (masses, G, initial conditions) separately
from the *numerical* parameters (dt, t_max).  This makes it trivial to run
the same physical system at different resolutions — e.g. for a convergence
study or timestep adequacy check — without modifying the physical definition.

Available cases
---------------
  "circular"    : e = 0,  stable circular orbit
  "elliptical"  : 0 < e < 1,  bound ellipse
  "parabolic"   : e = 1,  escape on parabolic trajectory
  "hyperbolic"  : e > 1,  flyby on hyperbolic trajectory
  "earth_moon"  : realistic Earth-Moon system  (SI units)
  "earth_sun"   : realistic Earth-Sun system   (SI units)

Future cases to add toward the roadmap
---------------------------------------
  "leo_satellite"   : Low Earth orbit satellite (first step toward collision)
  "geostationary"   : GEO orbit demonstration
  "binary_star"     : Equal-mass binary (strong radiation context)
  "three_body_stub" : Placeholder for first 3-body extension

Adding a new case
-----------------
Simply add a new entry in `CASE_REGISTRY` following the pattern below.
The build function will validate it automatically.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from .physics import OrbitalSystem, orbital_period, orbital_invariants, total_energy, angular_momentum_z


# ---------------------------------------------------------------------------
# Helper: centre-of-mass initial conditions
# ---------------------------------------------------------------------------

def _com_ics(
    m1: float, m2: float,
    r0: float, v_rel_perp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct initial conditions in the centre-of-mass frame.

    Places the two bodies along the x-axis separated by r0, with
    perpendicular relative velocity v_rel_perp (purely tangential).
    The CoM is at the origin and has zero velocity.

    The CoM condition guarantees that the total momentum is exactly zero
    and the CoM frame is inertial — a requirement for the Kepler problem.

    Parameters
    ----------
    m1, m2       : masses of the two bodies
    r0           : initial separation distance
    v_rel_perp   : magnitude of relative velocity (perpendicular to r)

    Returns
    -------
    r1_0, r2_0, v1_0, v2_0 : 2D position and velocity vectors
    """
    M  = m1 + m2
    r1 = np.array([ r0 * m2 / M, 0.0])
    r2 = np.array([-r0 * m1 / M, 0.0])
    v1 = np.array([0.0,  v_rel_perp * m2 / M])
    v2 = np.array([0.0, -v_rel_perp * m1 / M])
    return r1, r2, v1, v2


# ---------------------------------------------------------------------------
# Toy-unit cases (G=1)
# ---------------------------------------------------------------------------

def _toy_case(
    name: str,
    speed_factor: float,
    n_revolutions: int = 3
) -> OrbitalSystem:
    """
    Build a toy-unit (G=1) two-body case with controlled eccentricity.

    The speed factor is relative to circular velocity:
        speed_factor = 1.0        → circular orbit
        1.0 < sf < sqrt(2)        → elliptical
        sf = sqrt(2)              → parabolic escape
        sf > sqrt(2)              → hyperbolic

    Parameters
    ----------
    n_revolutions : number of orbital periods to simulate (bound orbits)
    """
    G, m1, m2, r0 = 1.0, 1.0, 100.0, 4.0
    M      = m1 + m2
    v_circ = np.sqrt(G * M / r0)
    v      = v_circ * speed_factor

    r1, r2, v1, v2 = _com_ics(m1, m2, r0, v)

    # Determine t_max: use n_revolutions × T for bound, hardcoded otherwise
    E0 = total_energy(r1, r2, v1, v2, m1, m2, G)
    L0 = angular_momentum_z(r1, r2, v1, v2, m1, m2)
    e, p = orbital_invariants(E0, L0, m1, m2, G)

    if e < 0.9999:
        from .physics import orbital_period
        T = orbital_period(e, p, m1, m2, G)
        t_max = T * n_revolutions if T else 80.0
    else:
        t_max = {"parabolic": 80.0, "hyperbolic": 25.0}.get(name, 60.0)

    return OrbitalSystem(
        G=G, m1=m1, m2=m2,
        r1_0=r1, r2_0=r2, v1_0=v1, v2_0=v2,
        dt=0.001, t_max=t_max,
        label=name
    )


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

def build_case(name: str, n_revolutions: int = 3) -> OrbitalSystem:
    """
    Build an OrbitalSystem by name.

    Parameters
    ----------
    name          : case identifier string (see module docstring for options)
    n_revolutions : how many full orbits to simulate (bound cases only)

    Returns
    -------
    OrbitalSystem ready to pass to run_simulation()
    """
    name = name.lower().strip()

    # --- Toy-unit cases ---
    if name == "circular":
        return _toy_case("circular",   speed_factor=1.0,            n_revolutions=n_revolutions)

    if name == "elliptical":
        return _toy_case("elliptical", speed_factor=1.35,           n_revolutions=n_revolutions)

    if name == "parabolic":
        return _toy_case("parabolic",  speed_factor=np.sqrt(2.0),   n_revolutions=n_revolutions)

    if name == "hyperbolic":
        return _toy_case("hyperbolic", speed_factor=np.sqrt(2.0) * 1.3, n_revolutions=n_revolutions)

    # --- Realistic SI cases ---
    if name == "earth_moon":
        G  = 6.6743e-11   # m³ kg⁻¹ s⁻²
        m1 = 5.972e24     # kg  (Earth)
        m2 = 7.342e22     # kg  (Moon)
        r0 = 3.844e8      # m   (mean Earth-Moon distance)
        v  = np.sqrt(G * (m1 + m2) / r0)   # exact circular speed
        r1, r2, v1, v2 = _com_ics(m1, m2, r0, v)
        # Compute period and scale by n_revolutions (same logic as toy cases)
        E0 = total_energy(r1, r2, v1, v2, m1, m2, G)
        L0 = angular_momentum_z(r1, r2, v1, v2, m1, m2)
        e, p = orbital_invariants(E0, L0, m1, m2, G)
        T = orbital_period(e, p, m1, m2, G)
        t_max = T * n_revolutions if T else 2.551e6
        return OrbitalSystem(
            G=G, m1=m1, m2=m2,
            r1_0=r1, r2_0=r2, v1_0=v1, v2_0=v2,
            dt=200.0, t_max=t_max,
            label="earth_moon"
        )

    if name == "earth_sun":
        G  = 6.6743e-11
        m1 = 5.972e24     # kg  (Earth)
        m2 = 1.989e30     # kg  (Sun)
        r0 = 1.471e11     # m   (perihelion distance)
        v  = 3.029e4      # m/s (Earth perihelion velocity)
        r1, r2, v1, v2 = _com_ics(m1, m2, r0, v)
        E0 = total_energy(r1, r2, v1, v2, m1, m2, G)
        L0 = angular_momentum_z(r1, r2, v1, v2, m1, m2)
        e, p = orbital_invariants(E0, L0, m1, m2, G)
        T = orbital_period(e, p, m1, m2, G)
        t_max = T * n_revolutions if T else 3.156e7
        return OrbitalSystem(
            G=G, m1=m1, m2=m2,
            r1_0=r1, r2_0=r2, v1_0=v1, v2_0=v2,
            dt=3600.0, t_max=t_max,
            label="earth_sun"
        )

    raise ValueError(
        f"Unknown case: {name!r}\n"
        f"Available cases: circular, elliptical, parabolic, hyperbolic, "
        f"earth_moon, earth_sun"
    )


AVAILABLE_CASES = [
    "circular", "elliptical", "parabolic", "hyperbolic",
    "earth_moon", "earth_sun"
]