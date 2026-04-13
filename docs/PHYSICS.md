# Physics Reference
## Orbital Dynamics Simulator — Formula Derivations

---

## 1. The Two-Body Problem and Reduction to One-Body

The Newtonian two-body problem describes two point masses m₁ and m₂
interacting through gravity:

```
m₁ r̈₁ = +G m₁m₂ (r₂−r₁) / |r₁−r₂|³
m₂ r̈₂ = +G m₁m₂ (r₁−r₂) / |r₁−r₂|³
```

Define the relative coordinate r = r₁ − r₂ and the CoM coordinate
R = (m₁r₁ + m₂r₂)/(m₁+m₂). These decouple:

```
R̈ = 0                   (CoM moves at constant velocity)
μ r̈ = −G m₁m₂ r̂ / r²   (relative coordinate obeys 1-body equation)
```

where μ = m₁m₂/(m₁+m₂) is the reduced mass.

**Consequence:** In the CoM frame (R=0), the full two-body problem is
mathematically identical to a particle of mass μ orbiting a fixed
center of force. All orbital geometry is computed from r.

---

## 2. Conserved Quantities

### 2.1 Total Mechanical Energy

```
E = ½m₁|v₁|² + ½m₂|v₂|² − G m₁m₂/|r₁−r₂|
```

In the CoM frame, this decomposes as:
```
E = ½M|V_CoM|² + ½μ|ṙ|² − G m₁m₂/r
  = E_CoM + E_rel
```

Since R̈ = 0, E_CoM is a constant. The *orbital energy* is:
```
E_rel = ½μ|ṙ|² − G m₁m₂/r = ½μ(ṙ² + r²θ̇²) − k/r
```
where k = G m₁m₂.

**Conservation:** E is conserved because gravity is a conservative force
(derivable from a potential). Any numerical drift in E measures integrator
quality exclusively.

### 2.2 Angular Momentum

For a central force (force directed purely along r̂), the torque τ = r × F = 0.
Therefore:

```
dL/dt = 0  →  L = μ(r × ṙ) = constant
```

In 2D: L = μ(x ẏ − y ẋ).

**Conservation:** L is an exact symmetry. In the Velocity Verlet scheme,
the force is always computed radially — this means L is conserved to
*machine precision* (not just timestep order). Any L drift is floating
point rounding noise.

---

## 3. Orbital Invariants from Conserved Quantities

### 3.1 The Orbit Equation

In polar coordinates (r, θ) centered on the CoM origin, the orbit
equation is derived by substituting u = 1/r and using L = μr²θ̇:

```
d²u/dθ² + u = μk/L²
```

This has the general solution:

```
u(θ) = μk/L² + A cos(θ − θ₀)
```

or equivalently:

```
r(θ) = p / (1 + e cos(θ − θ₀))
```

where:
```
p = L²/(μk)       (semi-latus rectum)
e = A L²/(μk)     (eccentricity, determined by initial conditions)
θ₀ = 0            (periapsis direction, set by choice of coordinates)
```

### 3.2 Eccentricity from Energy and Angular Momentum

Matching the orbit equation to the total energy:

```
E = ½μ(ṙ² + r²θ̇²) − k/r
```

At the turning points (ṙ = 0), and substituting r = p/(1 ± e):

```
E = −k/(2a) = −μk²(1−e²)/(2L²)
```

Solving for e:

```
e = sqrt(1 + 2EL²/(μk²))
```

This is the fundamental formula linking the conserved quantities (E, L)
to the geometric orbit parameter (e). It is the algebraic backbone of
the entire simulation.

**Physical interpretation:**
- E < 0 → 0 ≤ e < 1 → bound orbit (ellipse or circle)
- E = 0 → e = 1 → marginally bound (parabola)
- E > 0 → e > 1 → unbound (hyperbola)

---

## 4. Effective Potential

### 4.1 Definition

The radial equation of motion is:

```
μ r̈ = −∂U_eff/∂r = L²/(μr³) − k/r²
```

where the effective potential is:

```
U_eff(r) = L²/(2μr²) − k/r
```

The first term (centrifugal barrier, ~1/r²) is repulsive and dominates
at small r. The second term (gravitational well, ~1/r) is attractive and
dominates at large r. Their combination creates a potential well.

### 4.2 Minimum of U_eff (Circular Orbit Radius)

Setting dU_eff/dr = 0:

```
−L²/(μr³) + k/r² = 0
→  r_circ = L²/(μk)  =  p
```

The minimum of U_eff equals the semi-latus rectum p. At this radius,
all motion is tangential — this is the circular orbit.

### 4.3 Turning Points

The turning points satisfy U_eff(r) = E (radial kinetic energy = 0):

```
L²/(2μr²) − k/r = E
```

This is a quadratic in 1/r with solutions:

```
r_min = p / (1 + e)     (periapsis)
r_max = p / (1 − e)     (apoapsis, only for e < 1)
```

For e ≥ 1, the orbit is unbound: the body escapes to r → ∞.

### 4.4 Stability Criterion

The circular orbit at r = r_circ is a *stable* equilibrium of the
radial equation because d²U_eff/dr² > 0 there:

```
d²U_eff/dr² |_{r=r_circ} = 3L²/(μr⁴) − 2k/r³ |_{r_circ} = k/r_circ³ > 0
```

This means small radial perturbations result in bounded oscillations —
the orbit precesses but remains close to the unperturbed conic.

This stability result generalises: in the N-body case, orbital stability
is determined by whether the effective potential (including perturbation
forces) still has a well-defined minimum.

---

## 5. Velocity Verlet Integration

### 5.1 Algorithm

Given state (rₙ, vₙ) at time tₙ:

```
Step 1: aₙ = F(rₙ)/m
Step 2: r_{n+1} = rₙ + vₙ dt + ½ aₙ dt²
Step 3: a_{n+1} = F(r_{n+1})/m
Step 4: v_{n+1} = vₙ + ½(aₙ + a_{n+1}) dt
```

### 5.2 Why Symplectic?

The Velocity Verlet scheme is equivalent to a first-order *symplectic*
integrator applied to the Hamiltonian H = KE + PE. Symplectic integrators:

- Conserve the *phase space volume* (Liouville's theorem)
- Exactly conserve a *modified* Hamiltonian H̃ = H + O(dt²)
- Therefore keep |E − H̃| bounded for all time (no secular drift)

Compare to non-symplectic methods:
- Euler: O(dt) error, secular energy *growth*
- RK4: O(dt⁴) error per step, but *secular energy drift* over long times

For orbital mechanics, secular drift mimics physical dissipation
(orbital decay). Using RK4 would make it impossible to distinguish
numerical error from real orbital decay. Symplectic integrators are
the correct choice.

### 5.3 Convergence Order

Velocity Verlet is globally 2nd order: the energy error scales as O(dt²).
Halving dt reduces the energy error by ~4×. This is verified by the
`convergence_study()` function in `analysis.py`.

---

## 6. Orbit Residual — Validation Metric

### 6.1 Definition

For each simulation step i, extract the polar angle:

```
θᵢ = arctan2(yᵢ, xᵢ)
```

Compute the analytical prediction:

```
r_analytic(θᵢ) = p / (1 + e cos θᵢ)
```

The residual is:

```
Δrᵢ = |rᵢ_numerical − r_analytic(θᵢ)|
```

### 6.2 What it validates

A small residual (< 0.01% of r_min) simultaneously validates:
1. The integrator correctly solves Newton's equations
2. The analytical formulas for e and p are correct
3. The orbit invariants are genuinely conserved by the dynamics
4. The initial conditions are physically consistent

This is stronger than energy/momentum conservation alone, which only
validate individual quantities. The residual validates the *combined*
prediction of the entire analytical framework.

---

## 7. Connection to Future Phases

### 7.1 Perturbations (Phase 3)

Adding a perturbation force F_perturb:
```
r̈ = −GM r̂/r² + F_perturb/μ
```

E and L are no longer conserved. Instead, they evolve slowly:
```
dE/dt = ṙ · F_perturb
dL/dt = r × F_perturb
```

The orbit residual quantifies the perturbation effect: it measures
how far the orbit has drifted from the unperturbed Kepler prediction.

### 7.2 Stability and the Effective Potential (Phase 4)

In the N-body problem, each satellite experiences:
- Direct gravity from Earth
- Perturbations from other satellites
- J2 oblateness, drag, radiation pressure

Orbital stability is determined by whether the *total* effective
potential (gravity + perturbations) still confines the orbit. The
effective potential analysis established in Phase 1 is the framework
for this question.

### 7.3 Machine Learning Features (Phase 5)

The orbital invariants {e, p, a, i, Ω, ω} are the natural feature
space for ML stability prediction because they are:
- Slowly varying under perturbations (action-angle variables)
- Dimensionless or easily normalised
- Physically interpretable (directly relate to collision risk)

The Phase 1 framework generates clean training data with known
physical ground truth.
