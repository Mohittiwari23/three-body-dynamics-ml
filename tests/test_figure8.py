import sys
sys.path.insert(0, '.')

from three_body.integrator3 import validate_figure_eight
from three_body.integrator3 import figure_eight_system, run_simulation_3body
from three_body.visualize3 import animate_3body


if __name__ == "__main__":
    result = validate_figure_eight(dt=0.001, n_periods=10)

    print("\n=== FIGURE-8 VALIDATION ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Re-run simulation to get full trajectory (validate_figure_eight only returns summary)
    sys_ = figure_eight_system(dt=0.001, n_periods=10)
    sim_result = run_simulation_3body(sys_, compute_megno=True)

    print("\nLaunching visualization...")

    animate_3body(sim_result)

