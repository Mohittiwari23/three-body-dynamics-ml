"""
orbital_simulator/physics.py
==============================
Core Newtonian gravitational mechanics for the two-body problem.

All calculations are performed in SI units (or in the consistent
natural-unit system chosen by the user for toy cases).

Physical formulas implemented
------------------------------
Reduced mass          :  μ = m₁m₂ / (m₁+m₂)
Gravitational force   :  F = -G m₁m₂ r̂ / r²
Kinetic energy        :  KE = ½m₁v₁² + ½m₂v₂²
Potential energy      :  PE = -G m₁m₂ / r
Angular momentum (z)  :  L  = μ (r × v)_z  [relative frame]
Eccentricity          :  e  = sqrt(1 + 2EL² / μk²)  where k=Gm₁m₂
Semi-latus rectum     :  p  = L² / (μk)
Effective potential   :  U_eff(r) = L²/(2μr²) − Gm₁m₂/r
Orbital period        :  T  = 2π a^(3/2) / sqrt(G(m₁+m₂))
                          where a = p/(1−e²)  [bound orbits only]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OrbitalSystem:
    """
    Complete description of a two-body gravitational system.

    Separates *physical* parameters (masses, G) from *numerical* parameters
    (dt, t_max) so the same physical system can be run at different
    resolutions without touching the physics definition.

    Attributes
    ----------
    G   : gravitational constant  [m³ kg⁻¹ s⁻²]  (or natural units)
    m1  : mass of body 1
    m2  : mass of body 2
    r1_0 : initial position of body 1  [2-vector]
    r2_0 : initial position of body 2  [2-vector]
    v1_0 : initial velocity of body 1  [2-vector]
    v2_0 : initial velocity of body 2  [2-vector]
    dt   : integration timestep        [s]
    t_max: total simulation duration   [s]
    label: human-readable name for this case
    """
    G    : float
    m1   : float
    m2   : float
    r1_0 : np.ndarray
    r2_0 : np.ndarray
    v1_0 : np.ndarray
    v2_0 : np.ndarray
    dt   : float
    t_max: float
    label: str = "unnamed"

    @property
    def total_mass(self) -> float:
        return self.m1 + self.m2

    @property
    def reduced_mass(self) -> float:
        return self.m1 * self.m2 / (self.m1 + self.m2)


@dataclass
class SimulationResult:
    """
    Output container from a simulation run.

    Stores the full time history of the relative trajectory, conserved
    quantities, and derived physics metrics. Adding new outputs (e.g.
    Lyapunov exponents, 3D extension) should be done by extending this
    class rather than changing function signatures.

    Attributes
    ----------
    traj      : relative position r1 − r2 at each step   shape (N, 2)
    E_hist    : total mechanical energy at each step      shape (N,)
    L_hist    : z-component angular momentum at each step shape (N,)
    time      : time array                                shape (N,)
    E0        : initial total energy  (scalar)
    L0        : initial angular momentum  (scalar)
    e         : orbital eccentricity
    p         : semi-latus rectum
    r_min     : periapsis distance
    r_max     : apoapsis distance  (np.inf for unbound orbits)
    T_orb     : orbital period (None for unbound)
    orbit_type: string classification of orbit shape
    dE        : relative energy error (E − E₀) / E_scale  shape (N,)
    dL        : relative angular momentum error            shape (N,)
    residuals : |r_numerical − r_analytical(θ)|            shape (N,)
               (None if analytical comparison not performed)
    system    : back-reference to the OrbitalSystem used
    """
    traj      : np.ndarray
    E_hist    : np.ndarray
    L_hist    : np.ndarray
    time      : np.ndarray
    E0        : float
    L0        : float
    e         : float
    p         : float
    r_min     : float
    r_max     : float
    T_orb     : Optional[float]
    orbit_type: str
    dE        : np.ndarray
    dL        : np.ndarray
    residuals : Optional[np.ndarray]
    system    : OrbitalSystem

    # -----------------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------------

    @property
    def max_energy_error(self) -> float:
        """Maximum absolute relative energy drift."""
        return float(np.max(np.abs(self.dE)))

    @property
    def max_momentum_error(self) -> float:
        """Maximum absolute relative angular-momentum drift."""
        return float(np.max(np.abs(self.dL)))

    @property
    def max_orbit_residual(self) -> Optional[float]:
        """Peak orbit residual |r_num − r_analytic(θ)| in length units."""
        if self.residuals is None:
            return None
        return float(np.max(self.residuals))

    @property
    def is_bound(self) -> bool:
        return self.e < 1.0


# ---------------------------------------------------------------------------
# Scalar physics functions
# ---------------------------------------------------------------------------

def reduced_mass(m1: float, m2: float) -> float:
    """Two-body reduced mass  μ = m₁m₂ / (m₁+m₂)."""
    return m1 * m2 / (m1 + m2)


def total_energy(
    r1: np.ndarray, r2: np.ndarray,
    v1: np.ndarray, v2: np.ndarray,
    m1: float, m2: float, G: float
) -> float:
    """
    Total mechanical energy of the two-body system.

        E = ½m₁|v₁|² + ½m₂|v₂|² − Gm₁m₂/|r₁−r₂|

    Computed in the lab frame, which equals the CoM-frame energy
    (CoM kinetic energy is separately conserved and constant).
    """
    KE = 0.5 * m1 * np.dot(v1, v1) + 0.5 * m2 * np.dot(v2, v2)
    PE = -G * m1 * m2 / np.linalg.norm(r1 - r2)
    return float(KE + PE)


def angular_momentum_z(
    r1: np.ndarray, r2: np.ndarray,
    v1: np.ndarray, v2: np.ndarray,
    m1: float, m2: float
) -> float:
    """
    z-component of the orbital angular momentum.

    Computed in the reduced-mass frame:
        L = μ (r_rel × v_rel)_z = μ (x·ẏ − y·ẋ)

    This is an exact conserved quantity for 2D planar motion in a
    central force field.  Any drift in L during simulation is purely
    numerical and serves as an integrator quality metric.
    """
    mu    = reduced_mass(m1, m2)
    r_rel = r1 - r2
    v_rel = v1 - v2
    return float(mu * (r_rel[0] * v_rel[1] - r_rel[1] * v_rel[0]))


def effective_potential(
    r: np.ndarray | float,
    L: float, m1: float, m2: float, G: float
) -> np.ndarray | float:
    """
    Effective radial potential in the reduced-mass frame.

        U_eff(r) = L²/(2μr²) − Gm₁m₂/r

    The first term is the centrifugal barrier (repulsive).
    The second term is the gravitational well (attractive).

    Physical interpretation
    -----------------------
    The radial motion of the reduced particle is equivalent to 1D motion
    in this effective potential.  Turning points satisfy U_eff(r) = E,
    i.e. the kinetic energy of radial motion vanishes there.

    Parameters
    ----------
    r : radial distance (scalar or array)

    Returns
    -------
    U_eff at each r (same shape as input)
    """
    mu = reduced_mass(m1, m2)
    return L**2 / (2.0 * mu * r**2) - G * m1 * m2 / r


def orbital_invariants(
    E: float, L: float,
    m1: float, m2: float, G: float
) -> tuple[float, float]:
    """
    Derive orbital shape invariants from conserved quantities.

    Eccentricity
    ------------
        e = sqrt(1 + 2EL² / (μk²))
    where k = Gm₁m₂ is the gravitational coupling constant.

    This formula comes from solving the orbit equation analytically:
    inserting the conic section ansatz r(θ) = p/(1+e cosθ) into the
    equations of motion and matching to E and L.

    Semi-latus rectum
    -----------------
        p = L² / (μk)

    These two quantities completely determine the orbit shape:
        e < 1  →  ellipse (or circle at e=0)
        e = 1  →  parabola
        e > 1  →  hyperbola

    Parameters
    ----------
    E : total mechanical energy
    L : angular momentum

    Returns
    -------
    (e, p) : eccentricity, semi-latus rectum
    """
    mu = reduced_mass(m1, m2)
    k  = G * m1 * m2
    discriminant = 1.0 + (2.0 * E * L**2) / (mu * k**2)
    e = np.sqrt(max(0.0, discriminant))
    p = L**2 / (mu * k)
    return float(e), float(p)


def orbital_period(
    e: float, p: float,
    m1: float, m2: float, G: float
) -> Optional[float]:
    """
    Keplerian orbital period for bound orbits (e < 1).

        T = 2π a^(3/2) / sqrt(G(m₁+m₂))

    where a = p/(1−e²) is the semi-major axis.

    Returns None for parabolic or hyperbolic orbits (they do not repeat).
    Threshold e < 0.9999 guards against near-parabolic near-singularity.
    """
    if e >= 0.9999:
        return None
    a = p / (1.0 - e**2)
    T = 2.0 * np.pi * a**1.5 / np.sqrt(G * (m1 + m2))
    return float(T)


def classify_orbit(e: float) -> str:
    """
    Map eccentricity to a human-readable orbit classification.

    Thresholds
    ----------
    e < 1e-4   : numerically circular  (exact circle at e=0)
    1e-4 ≤ e < 1 : elliptical
    1 ≤ e < 1.005 : parabolic  (small numerical window around e=1)
    e ≥ 1.005  : hyperbolic
    """
    if e < 1e-4:
        return "Circular  (e ≈ 0)"
    elif e < 1.0:
        return "Elliptical  (0 < e < 1)"
    elif e < 1.005:
        return "Parabolic  (e ≈ 1)"
    else:
        return "Hyperbolic  (e > 1)"


def turning_points(e: float, p: float) -> tuple[float, float]:
    """
    Radial turning points from the conic section geometry.

        r_min = p / (1 + e)   [periapsis: closest approach]
        r_max = p / (1 - e)   [apoapsis:  farthest point, bound only]

    For unbound orbits (e ≥ 1), r_max = np.inf.
    These are the exact roots of U_eff(r) = E.
    """
    r_min = p / (1.0 + e)
    r_max = p / (1.0 - e) if e < 1.0 else np.inf
    return float(r_min), float(r_max)


def energy_scale(E0: float, KE0: float) -> float:
    """
    Choose a physically meaningful energy normalisation scale.

    For bound orbits:  scale = |E₀|  (total binding energy)
    For near-parabolic: |E₀| → 0, so fall back to KE₀ to avoid
    division by near-zero.  This is documented here so the choice is
    explicit and reproducible.
    """
    return abs(E0) if abs(E0) > 1e-10 * KE0 else KE0


def timestep_adequacy_warning(
    dt: float,
    r_min: float, m1: float, m2: float, G: float
) -> Optional[str]:
    """
    Check whether dt is small enough relative to the orbital timescale
    near periapsis.

    The local dynamical timescale at periapsis is
        τ_peri = r_min / v_peri  ≈  sqrt(r_min³ / (G M))

    We require dt < τ_peri / 20 for adequate sampling.
    Returns a warning string if this condition is violated, else None.

    This check is a first step toward adaptive timestepping.
    """
    tau_peri = np.sqrt(r_min**3 / (G * (m1 + m2)))
    if dt > tau_peri / 20.0:
        ratio = dt / tau_peri
        return (
            f"WARNING: dt = {dt:.3e} is {ratio:.1f}× the periapsis "
            f"timescale τ_peri = {tau_peri:.3e}. "
            f"Consider reducing dt to < {tau_peri/20:.3e} for accuracy "
            f"near closest approach."
        )
    return None


def analytical_orbit_xy(
    p: float, e: float, peri_angle: float = 0.0, n_points: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (x, y) points of the analytical conic orbit.

    Orbit equation:  r(θ) = p / (1 + e·cosθ)

    For bound orbits: full 2π sweep.
    For unbound:  θ swept between ±θ_∞ where θ_∞ = arccos(−1/e) is the
                  asymptotic direction.  A safety margin of 0.97 keeps the
                  plot away from the asymptote.

    Parameters
    ----------
    peri_angle : rotation of periapsis from +x axis [radians]
    n_points   : number of points on the curve
    """
    if e < 1.0:
        theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    else:
        # θ_∞ is the asymptotic angle where r → ∞
        theta_inf = np.arccos(np.clip(-1.0 / e, -1.0, 1.0))
        # 0.97 margin: display crop, not physics.  r grows very fast near θ_∞.
        theta = np.linspace(-theta_inf * 0.97, theta_inf * 0.97, n_points)

    denom = 1.0 + e * np.cos(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 1e-9, p / denom, np.nan)

    valid = np.isfinite(r) & (r > 0.0)
    th_v, r_v = theta[valid], r[valid]
    x = r_v * np.cos(th_v + peri_angle)
    y = r_v * np.sin(th_v + peri_angle)
    return x, y


def compute_orbit_residuals(
    traj: np.ndarray,
    p: float, e: float
) -> np.ndarray:
    """
    Quantitative comparison between numerical trajectory and analytical orbit.

    For each simulated position r_vec(t), extract the angle θ(t) and
    compute the analytical radius r_analytic(θ).  The residual is:

        Δr(t) = |r_numerical(t)| − r_analytic(θ(t))

    A near-zero residual (relative to r_min) is the strongest possible
    validation that:
      (1) the integrator conserves the orbit shape
      (2) the analytical invariants (e, p) are correct

    Parameters
    ----------
    traj : shape (N, 2) array of relative positions [r1 - r2]

    Returns
    -------
    residuals : shape (N,) absolute difference in radial distance [same units as traj]
    """
    r_num = np.linalg.norm(traj, axis=1)
    theta = np.arctan2(traj[:, 1], traj[:, 0])      # angle of relative position
    denom = 1.0 + e * np.cos(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_analytic = np.where(denom > 1e-9, p / denom, np.nan)
    residuals = np.abs(r_num - r_analytic)
    return residuals
