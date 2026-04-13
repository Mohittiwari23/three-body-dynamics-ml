"""
tests/test_physics.py
======================
Unit tests for the physics module.

These tests validate that the physical formulas are correct
independently of the integrator.  Passing these tests is the
minimum standard for calling the framework "physically credible."

Run with:   python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orbital_simulator.physics import (
    reduced_mass,
    total_energy,
    angular_momentum_z,
    orbital_invariants,
    effective_potential,
    orbital_period,
    classify_orbit,
    turning_points,
    analytical_orbit_xy,
    compute_orbit_residuals,
)


# ---------------------------------------------------------------------------
# Reduced mass
# ---------------------------------------------------------------------------

class TestReducedMass:
    def test_equal_masses(self):
        """Two equal masses → μ = m/2"""
        assert reduced_mass(2.0, 2.0) == pytest.approx(1.0)

    def test_one_dominant(self):
        """m2 >> m1 → μ ≈ m1  (planet around star)"""
        mu = reduced_mass(1.0, 1e6)
        assert mu == pytest.approx(1.0, rel=1e-5)

    def test_symmetry(self):
        """μ(m1, m2) == μ(m2, m1)"""
        assert reduced_mass(3.0, 7.0) == pytest.approx(reduced_mass(7.0, 3.0))


# ---------------------------------------------------------------------------
# Conserved quantities for known circular orbit
# ---------------------------------------------------------------------------

class TestCircularOrbit:
    """
    For a circular orbit:
        v_circ = sqrt(G M / r)
        E = -G m1 m2 / (2r)      (virial theorem)
        L = μ * r * v_circ
        e = 0
    """

    def setup_method(self):
        self.G = 1.0
        self.m1, self.m2 = 1.0, 100.0
        self.r0 = 4.0
        M = self.m1 + self.m2
        v = np.sqrt(self.G * M / self.r0)
        mu = reduced_mass(self.m1, self.m2)

        # CoM frame
        self.r1 = np.array([self.r0 * self.m2 / M, 0.0])
        self.r2 = np.array([-self.r0 * self.m1 / M, 0.0])
        self.v1 = np.array([0.0,  v * self.m2 / M])
        self.v2 = np.array([0.0, -v * self.m1 / M])

    def test_total_energy_circular(self):
        """E_circular = -G m1 m2 / (2 r0)"""
        E = total_energy(self.r1, self.r2, self.v1, self.v2,
                         self.m1, self.m2, self.G)
        E_expected = -self.G * self.m1 * self.m2 / (2.0 * self.r0)
        assert E == pytest.approx(E_expected, rel=1e-10)

    def test_angular_momentum_circular(self):
        """L = μ r v_circ"""
        L = angular_momentum_z(self.r1, self.r2, self.v1, self.v2,
                               self.m1, self.m2)
        mu = reduced_mass(self.m1, self.m2)
        v_circ = np.sqrt(self.G * (self.m1 + self.m2) / self.r0)
        L_expected = mu * self.r0 * v_circ
        assert L == pytest.approx(L_expected, rel=1e-10)

    def test_eccentricity_circular(self):
        """Circular orbit → e ≈ 0"""
        E = total_energy(self.r1, self.r2, self.v1, self.v2,
                         self.m1, self.m2, self.G)
        L = angular_momentum_z(self.r1, self.r2, self.v1, self.v2,
                               self.m1, self.m2)
        e, _ = orbital_invariants(E, L, self.m1, self.m2, self.G)
        assert e == pytest.approx(0.0, abs=1e-6)

    def test_orbit_classification_circular(self):
        assert classify_orbit(0.0) == "Circular  (e ≈ 0)"
        assert classify_orbit(5e-5) == "Circular  (e ≈ 0)"


# ---------------------------------------------------------------------------
# Orbit type classification
# ---------------------------------------------------------------------------

class TestOrbitClassification:
    def test_elliptical(self):
        assert classify_orbit(0.5) == "Elliptical  (0 < e < 1)"

    def test_parabolic(self):
        assert classify_orbit(1.0) == "Parabolic  (e ≈ 1)"
        assert classify_orbit(1.002) == "Parabolic  (e ≈ 1)"

    def test_hyperbolic(self):
        assert classify_orbit(1.5) == "Hyperbolic  (e > 1)"
        assert classify_orbit(2.0) == "Hyperbolic  (e > 1)"


# ---------------------------------------------------------------------------
# Turning points
# ---------------------------------------------------------------------------

class TestTurningPoints:
    def test_circular(self):
        """Circular: r_min == r_max == p"""
        p, e = 4.0, 0.0
        r_min, r_max = turning_points(e, p)
        assert r_min == pytest.approx(p, rel=1e-10)
        assert r_max == pytest.approx(p, rel=1e-10)

    def test_elliptical_sum(self):
        """r_min + r_max = 2a = 2p/(1-e²)"""
        p, e = 4.0, 0.6
        r_min, r_max = turning_points(e, p)
        a = p / (1 - e**2)
        assert (r_min + r_max) / 2.0 == pytest.approx(a, rel=1e-10)

    def test_unbound_rmax(self):
        """Hyperbolic: r_max = inf"""
        _, r_max = turning_points(1.5, 3.0)
        assert not np.isfinite(r_max)


# ---------------------------------------------------------------------------
# Effective potential
# ---------------------------------------------------------------------------

class TestEffectivePotential:
    def test_minimum_at_circular_radius(self):
        """
        The minimum of U_eff occurs at the circular orbit radius.
        dU_eff/dr = 0 → r_circ = L² / (μ G m1 m2)
        """
        G, m1, m2 = 1.0, 1.0, 100.0
        mu = reduced_mass(m1, m2)
        k  = G * m1 * m2
        r0 = 4.0
        v_circ = np.sqrt(G * (m1 + m2) / r0)
        L  = mu * r0 * v_circ
        r_circ = L**2 / (mu * k)

        # Numerically verify U_eff is at its minimum around r_circ
        dr   = r_circ * 1e-4
        U_c  = effective_potential(r_circ,      L, m1, m2, G)
        U_lo = effective_potential(r_circ - dr, L, m1, m2, G)
        U_hi = effective_potential(r_circ + dr, L, m1, m2, G)
        assert U_c < U_lo
        assert U_c < U_hi

    def test_centrifugal_dominates_small_r(self):
        """For r → 0, L²/2μr² term dominates → U_eff → +∞"""
        U = effective_potential(1e-6, L=1.0, m1=1.0, m2=1.0, G=1.0)
        assert U > 1e6


# ---------------------------------------------------------------------------
# Orbital period
# ---------------------------------------------------------------------------

class TestOrbitalPeriod:
    def test_earth_sun_period(self):
        """Earth-Sun period should be approximately 1 year = 3.156e7 s"""
        G  = 6.6743e-11
        m1 = 5.972e24
        m2 = 1.989e30
        # For a circular orbit at Earth's mean radius:
        r0 = 1.496e11
        mu = reduced_mass(m1, m2)
        k  = G * m1 * m2
        v  = np.sqrt(G * (m1 + m2) / r0)
        L  = mu * r0 * v
        E  = -G * m1 * m2 / (2 * r0)
        e, p = orbital_invariants(E, L, m1, m2, G)
        T = orbital_period(e, p, m1, m2, G)
        assert T == pytest.approx(3.156e7, rel=0.01)

    def test_parabolic_no_period(self):
        assert orbital_period(1.0, 1.0, 1.0, 1.0, 1.0) is None

    def test_hyperbolic_no_period(self):
        assert orbital_period(1.5, 1.0, 1.0, 1.0, 1.0) is None


# ---------------------------------------------------------------------------
# Analytical orbit
# ---------------------------------------------------------------------------

class TestAnalyticalOrbit:
    def test_circular_constant_radius(self):
        """r(θ) = p/(1 + 0·cosθ) = p  (constant for e=0)"""
        x, y = analytical_orbit_xy(p=4.0, e=0.0)
        r = np.sqrt(x**2 + y**2)
        assert np.all(np.abs(r - 4.0) < 1e-10)

    def test_hyperbolic_returns_finite_points(self):
        """Hyperbolic orbit should return only finite points"""
        x, y = analytical_orbit_xy(p=3.0, e=1.5)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Orbit residuals
# ---------------------------------------------------------------------------

class TestOrbitResiduals:
    def test_residual_on_exact_orbit(self):
        """
        Points that lie exactly on r(θ) = p/(1+e cosθ) should give
        near-zero residuals.
        """
        p, e = 3.0, 0.4
        x, y = analytical_orbit_xy(p, e)
        traj = np.stack([x, y], axis=1)
        residuals = compute_orbit_residuals(traj, p, e)
        finite = residuals[np.isfinite(residuals)]
        assert finite.max() < 1e-10
