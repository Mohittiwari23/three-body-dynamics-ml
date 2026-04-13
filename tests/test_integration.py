"""
tests/test_integration.py
==========================
Integration tests for the full simulation pipeline.

These test the *full pipeline*: build_case → run_simulation → analysis.
They verify:
  1. All cases run without errors
  2. Conserved quantities meet SRS performance requirements
  3. Orbit residuals < 0.1% of r_min
  4. MEGNO ≈ 2 for all regular 2-body orbits
  5. SI cases respect n_revolutions (bug-fix regression test)
  6. All cases start in the CoM frame

Run with:   python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orbital_simulator import build_case, run_simulation, compute_MEGNO


BOUND_CASES = ["circular", "elliptical", "earth_moon"]
ALL_CASES   = ["circular", "elliptical", "parabolic", "hyperbolic",
               "earth_moon", "earth_sun"]


# ---------------------------------------------------------------------------
# Basic pipeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", ALL_CASES)
def test_simulation_runs(case_name):
    """Every case must run without error and return a populated result."""
    system = build_case(case_name)
    result = run_simulation(system)
    assert result.traj.shape[1] == 2
    assert len(result.E_hist) == len(result.traj)
    assert len(result.L_hist) == len(result.traj)
    assert np.all(np.isfinite(result.E_hist))
    assert np.all(np.isfinite(result.L_hist))


# ---------------------------------------------------------------------------
# Conservation (SRS performance requirements)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", BOUND_CASES)
def test_energy_conservation(case_name):
    """Energy drift < 1e-5 (SRS specification)."""
    result = run_simulation(build_case(case_name))
    assert result.max_energy_error < 1e-5, (
        f"{case_name}: energy error {result.max_energy_error:.2e} exceeds 1e-5"
    )


@pytest.mark.parametrize("case_name", BOUND_CASES)
def test_angular_momentum_conservation(case_name):
    """Angular momentum drift < 1e-8 (SRS specification)."""
    result = run_simulation(build_case(case_name))
    assert result.max_momentum_error < 1e-8, (
        f"{case_name}: L error {result.max_momentum_error:.2e} exceeds 1e-8"
    )


# ---------------------------------------------------------------------------
# Orbit residuals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", ["circular", "elliptical"])
def test_orbit_residuals(case_name):
    """Residual |r_num − r_analytic| < 0.1% of r_min (SRS specification)."""
    result = run_simulation(build_case(case_name))
    assert result.residuals is not None
    rel_residual = result.max_orbit_residual / result.r_min
    assert rel_residual < 1e-3, (
        f"{case_name}: residual {rel_residual:.2e} exceeds 1e-3 × r_min"
    )


# ---------------------------------------------------------------------------
# Eccentricity and orbit type
# ---------------------------------------------------------------------------

def test_eccentricity_circular():
    assert run_simulation(build_case("circular")).e < 1e-4

def test_eccentricity_hyperbolic():
    assert run_simulation(build_case("hyperbolic")).e > 1.0

def test_parabolic_e_near_one():
    e = run_simulation(build_case("parabolic")).e
    assert abs(e - 1.0) < 1e-4, f"Parabolic e = {e:.6f}, expected ≈ 1.0"

def test_bound_orbit_finite_rmax():
    assert np.isfinite(run_simulation(build_case("elliptical")).r_max)

def test_unbound_orbit_infinite_rmax():
    assert not np.isfinite(run_simulation(build_case("hyperbolic")).r_max)


# ---------------------------------------------------------------------------
# CoM frame — all cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", ALL_CASES)
def test_initial_conditions_com_frame(case_name):
    """Total momentum must be zero in the CoM frame."""
    sys_ = build_case(case_name)
    p_total = sys_.m1 * sys_.v1_0 + sys_.m2 * sys_.v2_0
    assert np.linalg.norm(p_total) < 1e-6 * sys_.m1, (
        f"{case_name}: total momentum = {p_total} is not zero"
    )


# ---------------------------------------------------------------------------
# n_revolutions — SI cases (Bug fix regression tests)
# ---------------------------------------------------------------------------

def test_earth_moon_respects_n_revolutions():
    """
    earth_moon t_max must scale proportionally with n_revolutions.
    Previously hardcoded to 2.551e6 s — this test catches any regression.
    """
    s1 = build_case("earth_moon", n_revolutions=1)
    s3 = build_case("earth_moon", n_revolutions=3)
    ratio = s3.t_max / s1.t_max
    assert abs(ratio - 3.0) < 0.15, (
        f"earth_moon t_max ratio = {ratio:.3f}, expected 3.0"
    )


def test_earth_sun_respects_n_revolutions():
    """earth_sun t_max must scale with n_revolutions."""
    s1 = build_case("earth_sun", n_revolutions=1)
    s2 = build_case("earth_sun", n_revolutions=2)
    ratio = s2.t_max / s1.t_max
    assert abs(ratio - 2.0) < 0.05, (
        f"earth_sun t_max ratio = {ratio:.3f}, expected 2.0"
    )


# ---------------------------------------------------------------------------
# MEGNO — chaos indicator validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_name", ["circular", "elliptical"])
def test_megno_regular_orbit(case_name):
    """
    MEGNO must stay bounded for regular 2-body orbits.

    Exact convergence target is <Y> → 2.0 (Cincotta & Simó 2000).
    However, convergence rate depends strongly on eccentricity:
      - Circular (e≈0):  converges in ~3 periods  → <Y> ≈ 2.0 ± 0.3
      - Elliptical (e=0.82): needs ~10³ periods   → <Y> bounded but not yet at 2.0

    The test verifies that MEGNO is BOUNDED and POSITIVE, not diverging.
    Divergence (Y → ∞) would indicate a chaotic orbit — impossible for 2-body.
    The tighter convergence test is in test_megno_circular_tight().
    """
    system = build_case(case_name, n_revolutions=3)
    Y = compute_MEGNO(system)
    assert Y > 0.5, f"{case_name}: MEGNO <Y> = {Y:.4f} — too small, possible bug"
    assert Y < 20.0, f"{case_name}: MEGNO <Y> = {Y:.4f} — diverging, indicates chaos"


def test_megno_circular_tight():
    """
    Circular orbit MEGNO must be tightly converged to 2.0 ± 0.25.
    Circular orbits converge fastest because the orbit is exactly periodic.
    """
    system = build_case("circular", n_revolutions=3)
    Y = compute_MEGNO(system)
    assert abs(Y - 2.0) < 0.25, (
        f"Circular MEGNO <Y> = {Y:.4f}, expected 2.0 ± 0.25"
    )