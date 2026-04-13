"""
orbital_simulator
==================
A physics-driven two-body orbital dynamics simulator.

Phase 1 foundation of a research framework for orbital stability analysis.

Quick start
-----------
    from orbital_simulator import build_case, run_simulation, build_animation
    import matplotlib.pyplot as plt

    system = build_case("elliptical")
    result = run_simulation(system)
    ani    = build_animation(result)
    plt.show()
"""

from .cases      import build_case, AVAILABLE_CASES
from .physics    import (
    OrbitalSystem,
    SimulationResult,
    reduced_mass,
    total_energy,
    angular_momentum_z,
    effective_potential,
    orbital_invariants,
    orbital_period,
    classify_orbit,
    turning_points,
    energy_scale,
    analytical_orbit_xy,
    compute_orbit_residuals,
    timestep_adequacy_warning,
)
from .integrator import run_simulation
from .analysis   import print_summary, convergence_study, compute_MEGNO
from .visualize  import build_animation

__version__ = "0.1.0"
__author__  = "Orbital Simulator Project"

__all__ = [
    "build_case", "AVAILABLE_CASES",
    "OrbitalSystem", "SimulationResult",
    "reduced_mass", "total_energy", "angular_momentum_z",
    "effective_potential", "orbital_invariants", "orbital_period",
    "classify_orbit", "turning_points", "energy_scale",
    "analytical_orbit_xy", "compute_orbit_residuals",
    "timestep_adequacy_warning",
    "run_simulation",
    "print_summary", "convergence_study", "compute_MEGNO",
    "build_animation",
]