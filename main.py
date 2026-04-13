"""
main.py
=======
Command-line entry point for the orbital simulator.

Usage
-----
    python main.py                          # default: elliptical case
    python main.py --case circular
    python main.py --case earth_sun
    python main.py --case elliptical --revolutions 5
    python main.py --case elliptical --convergence
    python main.py --case elliptical --save orbit.mp4

Available cases: circular, elliptical, parabolic, hyperbolic, earth_moon, earth_sun
"""

import argparse
import sys
import matplotlib.pyplot as plt

from orbital_simulator import (
    build_case,
    run_simulation,
    build_animation,
    print_summary,
    convergence_study,
    AVAILABLE_CASES,
    timestep_adequacy_warning,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Newtonian two-body orbital dynamics simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--case", "-c",
        default="elliptical",
        choices=AVAILABLE_CASES,
        help="Orbital case to simulate (default: elliptical)"
    )
    parser.add_argument(
        "--revolutions", "-r",
        type=int, default=3,
        help="Number of orbital revolutions for bound orbits (default: 3)"
    )
    parser.add_argument(
        "--stride", "-s",
        type=int, default=30,
        help="Animation frame stride — higher = faster animation (default: 30)"
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Run a timestep convergence study before animating"
    )
    parser.add_argument(
        "--save",
        type=str, default=None,
        help="Save animation to this file path (.mp4 or .gif)"
    )
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Skip animation, only print summary and run analysis"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═' * 58}")
    print(f"  ORBITAL SIMULATOR  ·  Phase 1 Research Prototype")
    print(f"  Two-Body Newtonian Gravity  ·  Velocity Verlet")
    print(f"{'═' * 58}")
    print(f"  Case: {args.case.upper()}")

    # ---- Build physical system ----
    system = build_case(args.case, n_revolutions=args.revolutions)

    # ---- Timestep adequacy check ----
    # (need e, p first — do a quick initial-condition analysis)
    from orbital_simulator.physics import (
        total_energy, angular_momentum_z,
        orbital_invariants, turning_points
    )
    E0 = total_energy(system.r1_0, system.r2_0, system.v1_0, system.v2_0,
                      system.m1, system.m2, system.G)
    L0 = angular_momentum_z(system.r1_0, system.r2_0, system.v1_0, system.v2_0,
                            system.m1, system.m2)
    e_ic, p_ic = orbital_invariants(E0, L0, system.m1, system.m2, system.G)
    r_min_ic, _ = turning_points(e_ic, p_ic)

    warning = timestep_adequacy_warning(system.dt, r_min_ic,
                                        system.m1, system.m2, system.G)
    if warning:
        print(f"\n  {warning}")

    # ---- Optional convergence study ----
    if args.convergence:
        import numpy as np
        base_dt = system.dt
        dt_values = [base_dt * 4, base_dt * 2, base_dt, base_dt / 2, base_dt / 4]
        convergence_study(system, dt_values)

    # ---- Run simulation ----
    print(f"\n  Running simulation …  ({int(system.t_max / system.dt):,} steps)")
    result = run_simulation(system)
    print(f"  Complete.")

    # ---- Print summary ----
    print_summary(result)

    # ---- Animate ----
    if not args.no_animate:
        ani = build_animation(result, stride=args.stride, save_path=args.save)
        # Keep reference to prevent garbage collection
        plt.gcf()._ani_ref = ani
        plt.show()
    else:
        print("  Animation skipped (--no-animate).\n")


if __name__ == "__main__":
    main()
