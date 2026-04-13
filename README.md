# Orbital Dynamics Simulator
### A Physics-Driven Computational Framework for Orbital Stability Analysis

> **Current phase:** Phase 1 — Two-body Newtonian mechanics prototype  
> **Goal:** A research framework for orbital stability analysis and satellite conjunction prediction

---

## What This Project Is

This is not an animation project.

It is a computational physics framework being built as the foundation for orbital stability research — ultimately aimed at detecting potential collision risks in multi-body satellite systems. The current prototype demonstrates that the physical and numerical foundation is correct and extensible.

The simulator explicitly connects three things that orbital mechanics courses often treat separately:

| Conserved Quantity | Analytical Consequence | Numerical Signature |
|---|---|---|
| Total energy `E` | Determines orbit shape (bound vs unbound) | Energy error < 1e-6 per orbit |
| Angular momentum `L` | Determines orbit size via `p = L²/μk` | Momentum error < 1e-10 (machine precision) |
| Effective potential `U_eff(r)` | Predicts turning points r_min, r_max | Verified against analytical conic prediction |

---

## Quick Start

```bash
git clone <repo-url>
cd orbital_simulator
pip install -e .

# Run with default case (elliptical)
python main.py

# Choose a specific case
python main.py --case circular
python main.py --case earth_sun
python main.py --case hyperbolic

# Run a convergence study (demonstrates integrator is 2nd order)
python main.py --case elliptical --convergence

# Run all physics unit tests
pytest tests/ -v
```

---

## Project Structure

```
orbital_simulator/
│
├── orbital_simulator/          # Core Python package
│   ├── __init__.py             # Public API exports
│   ├── physics.py              # All physics formulas (pure functions)
│   ├── integrator.py           # Velocity Verlet time integration
│   ├── cases.py                # Physical system definitions
│   ├── analysis.py             # Post-simulation metrics, convergence study
│   └── visualize.py            # All plotting and animation
│
├── tests/
│   ├── test_physics.py         # Unit tests: formulas, conserved quantities
│   └── test_integration.py     # Pipeline tests: full sim, conservation checks
│
├── docs/
│   └── PHYSICS.md              # Derivations and formula reference
│
├── notebooks/                  # Jupyter notebooks (exploratory analysis)
│
├── main.py                     # Command-line entry point
├── setup.py                    # Installable package
├── requirements.txt
└── README.md
```

**Design principle:** Each module has exactly one responsibility.  
`physics.py` does not touch matplotlib. `visualize.py` does not touch physics formulas.  
This makes every component individually testable and replaceable.

---

## Physics Implementation

### The Two-Body Reduction

The full two-body problem (positions r₁, r₂ in the lab frame) reduces exactly to a one-body problem via the reduced mass μ = m₁m₂/(m₁+m₂). All orbital geometry is computed in the *relative frame* r = r₁ - r₂.

### Conserved Quantities

**Total energy:**
```
E = ½m₁|v₁|² + ½m₂|v₂|² − Gm₁m₂/|r₁−r₂|
```

**Angular momentum (z-component in 2D):**
```
L = μ(r × v)_z = μ(x ẏ − y ẋ)
```

These are exact integrals of motion. Any drift during simulation is *purely numerical* — it measures integrator quality, not physics.

### Orbital Invariants

From E and L alone, the complete orbit shape can be derived:

```
e = sqrt(1 + 2EL²/μk²)    where k = Gm₁m₂
p = L²/μk
```

These yield the conic orbit equation:
```
r(θ) = p / (1 + e·cosθ)
```

| e | Orbit type |
|---|---|
| e = 0 | Circle |
| 0 < e < 1 | Ellipse |
| e = 1 | Parabola (escape) |
| e > 1 | Hyperbola (flyby) |

### Effective Potential Analysis

```
U_eff(r) = L²/(2μr²) − Gm₁m₂/r
```

The centrifugal barrier `L²/2μr²` (repulsive) competes with the gravitational well `−Gm₁m₂/r` (attractive). The resulting potential:
- Has a minimum at the circular orbit radius
- Has turning points where `U_eff(r) = E` (the radial velocity vanishes)
- Is bounded below for bound orbits, unbounded for unbound orbits

This is the physical mechanism underlying orbital stability.

### Velocity Verlet Integrator

```
r_{n+1} = r_n + v_n dt + ½ a_n dt²
a_{n+1} = F(r_{n+1}) / m
v_{n+1} = v_n + ½(a_n + a_{n+1}) dt
```

Velocity Verlet is **symplectic**: it exactly conserves a modified Hamiltonian close to the true one. Energy error stays *bounded* over long integrations rather than growing secularly (as it would with standard Runge-Kutta). Angular momentum is conserved to machine precision because the gravitational force is always radial (exact central force symmetry).

---

## Visualisation Panels

The simulator produces a 5-panel animated figure:

| Panel | What it shows | Why it matters |
|---|---|---|
| **Orbit** | Numerical trajectory (blue) vs analytical conic (orange dashed) | Visual agreement between simulation and Kepler theory |
| **Effective Potential** | U_eff(r) with E-level, r_min, r_max marked; animated particle dot | Shows the physical mechanism — the particle bounces between turning points |
| **Energy Conservation** | (E − E₀)/scale vs time | Integrator quality metric — should stay < 1e-6 |
| **Angular Momentum** | (L − L₀)/L₀ vs time | Should be < 1e-10; central force symmetry |
| **Orbit Residual** ← *key* | \|r_numerical − r_analytical(θ)\| / r_min | **Quantitative** validation that numerical and analytical orbits agree |

The **orbit residual panel** is the most scientifically significant. It provides the quantitative bridge: if the residual is small (< 0.01% of r_min), the framework is demonstrably correct.

---

## Pre-defined Cases

| Case | Physics | Notes |
|---|---|---|
| `circular` | e ≈ 0 | Stable circular orbit |
| `elliptical` | 0 < e < 1 | Bound elliptical orbit |
| `parabolic` | e = 1 | Escape at exactly escape velocity |
| `hyperbolic` | e > 1 | Flyby / scattering trajectory |
| `earth_moon` | SI units | Realistic Earth-Moon system |
| `earth_sun` | SI units | Realistic Earth-Sun orbit |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Physics unit tests only
pytest tests/test_physics.py -v

# Integration pipeline tests only
pytest tests/test_integration.py -v

# With coverage report
pytest tests/ --cov=orbital_simulator --cov-report=term-missing
```

The test suite validates:
- All physics formulas against known analytical results
- Energy conservation < 1e-5 for all bound cases
- Angular momentum conservation < 1e-8 for all bound cases  
- Orbit residuals < 0.1% of r_min
- Earth-Sun period correct to 1% against known value

---

## Research Roadmap

This project is developed in phases. Each phase builds directly on the previous one.

```
Phase 1 (CURRENT)
─────────────────
2D Newtonian two-body problem
Velocity Verlet integration
Conserved quantities: E, L
Orbital invariants: e, p
Effective potential analysis
Quantitative orbit validation
Unit-tested physics core

        │
        ▼

Phase 2
───────
3D orbital dynamics
Keplerian orbital elements: (a, e, i, Ω, ω, M)
Conversion between Cartesian ↔ orbital elements
Visualization in 3D

        │
        ▼

Phase 3
───────
Orbital perturbations
J2 oblateness term (most important for LEO)
Atmospheric drag model
Solar radiation pressure
Verify perturbations cause expected precession rates

        │
        ▼

Phase 4
───────
N-body gravitational simulation
Constellation configurations (Starlink-like)
Conjunction detection: minimum approach distance
Close-approach prediction algorithm
Statistical ensemble of orbital histories

        │
        ▼

Phase 5
───────
Machine learning for stability prediction
Training data: simulated orbital histories from Phase 4
Features: (e, p, a, i, Ω, ω, perturbation params)
Target: stability class or time-to-conjunction
Model validation against numerical simulation
```

### Why this phase matters for the roadmap

Phase 1 is not just a demo. It establishes:

1. **The physical framework** — effective potential, orbital invariants, and the E-L-orbit connection are the same quantities that define stability in the N-body case.

2. **The validation methodology** — the orbit residual check (numerical vs analytical) is the template for how every future phase will be validated. Once we add perturbations, the residual measures the perturbation effect.

3. **The data structure** — `OrbitalSystem` and `SimulationResult` are designed to extend to 3D and N-body without rewriting the physics. Adding a 3rd body means adding a body to `OrbitalSystem` and a force term in `_gravitational_acceleration`.

4. **The test suite** — the conservation tests are the minimum standard. Every future phase will inherit these tests and add its own.

---

## Code Architecture Notes

### Why separate `physics.py` from `integrator.py`?

`physics.py` contains only pure mathematical functions (no state, no loops). These can be unit-tested in isolation, imported into notebooks, and reused in any future module. The integrator is a separate concern: it calls physics functions in a time loop. Mixing them (as most tutorials do) makes both harder to test and modify.

### Why `OrbitalSystem` and `SimulationResult` dataclasses?

As the project grows, every function would otherwise take 7–10 positional arguments. Dataclasses make the API stable: adding a `perturbation_force` field to `OrbitalSystem` doesn't break any existing function signatures.

### Why not use scipy.integrate.solve_ivp?

scipy's ODE solvers use adaptive RK methods that are not symplectic. For orbital mechanics, a non-symplectic integrator accumulates secular energy drift that mimics physical orbital decay — making it impossible to distinguish numerical error from real physics. The Velocity Verlet choice is deliberate and scientifically motivated.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.24 | Array operations, linear algebra |
| matplotlib | ≥ 3.7 | Visualisation and animation |
| pytest | ≥ 7.0 | Test suite (dev only) |

No heavy dependencies. The physics core has no dependencies beyond numpy.

---

## Contributing / Extending

To add a new physical case:
1. Add an entry in `cases.py → build_case()`
2. Add it to `AVAILABLE_CASES`
3. Add a test in `tests/test_integration.py`

To add a new analysis metric:
1. Add a function in `analysis.py`
2. Optionally add a field to `SimulationResult` in `physics.py`
3. Add a unit test in `tests/test_physics.py`

To add a perturbation force (Phase 3):
1. Add the force term in `integrator.py → _gravitational_acceleration()`
2. Add perturbation parameters to `OrbitalSystem`
3. Existing tests remain as the baseline regression suite
