# Physics Reference — Orbital Stability Framework

## Contents

1. The Two-Body Problem
2. Conserved Quantities
3. Orbital Invariants
4. Effective Potential
5. Velocity Verlet Integration
6. Orbit Residual — Two-Body Validation
7. MEGNO — Chaos Indicator
8. Phase 2 Feature Representation
9. The Three-Body Problem
10. Phase 3 Initial Condition Construction
11. Phase 3 Sampling Regimes
12. Phase 3 Outcome Labels
13. Phase 3 Feature Extraction
14. Bugs Fixed During Development
15. References

---

## 1. The Two-Body Problem

Two point masses m₁ and m₂:

```
m₁ r̈₁ = +G m₁m₂ (r₂−r₁) / |r₁−r₂|³
m₂ r̈₂ = +G m₁m₂ (r₁−r₂) / |r₁−r₂|³
```

Defining relative coordinate **r = r₁ − r₂** and CoM **R = (m₁r₁+m₂r₂)/M**:

```
R̈ = 0               CoM moves at constant velocity
μ r̈ = −G m₁m₂ r̂/r²  relative coord = 1-body equation
```

μ = m₁m₂/(m₁+m₂) is the reduced mass. In the CoM frame, the two-body
problem is exactly a single particle of mass μ in a central force field.
All orbital geometry is computed in the relative coordinate.

---

## 2. Conserved Quantities

### 2.1 Total Mechanical Energy

```
E = ½m₁|v₁|² + ½m₂|v₂|² − G m₁m₂/r
  = ½μ|v_rel|² − G m₁m₂/r          (CoM frame)
```

E is conserved (conservative force). Any numerical drift is purely
integrator error and serves as a quality metric.

### 2.2 Angular Momentum

Central force → zero torque → exact conservation:

```
L = μ (r × ṙ)_z = μ(x ẏ − y ẋ)    (2D, z-component)
```

The Velocity Verlet scheme computes forces radially at every step, so L
is conserved to **machine precision** — not just timestep order. Any L
drift is floating-point rounding noise.

---

## 3. Orbital Invariants

### 3.1 The Orbit Equation

Substitution u = 1/r with L = μr²θ̇ gives:

```
r(θ) = p / (1 + e cos θ)

p = L²/(μk)                         semi-latus rectum, k = G m₁m₂
e = √(1 + 2EL²/(μk²))               eccentricity
```

This is the algebraic backbone of the entire two-body analysis.

### 3.2 Orbit Classification

| E | e | Type |
|---|---|------|
| E < 0 | 0 ≤ e < 1 | Bound: circle or ellipse |
| E = 0 | e = 1 | Marginally bound: parabola |
| E > 0 | e > 1 | Unbound: hyperbola |

### 3.3 Geometric Quantities

```
a     = p/(1−e²)      semi-major axis (bound only)
r_min = p/(1+e)       periapsis
r_max = p/(1−e)       apoapsis (bound only; ∞ for e ≥ 1)
T     = 2π a^(3/2) / √(G(m₁+m₂))   period (bound only)
```

### 3.4 Normalised (Specific) Quantities

Dividing E and L by μ removes explicit mass dependence:

```
ε = E/μ    (specific energy)
h = L/μ    (specific angular momentum)
```

With M = m₁+m₂ = μ(1+q)²/q recoverable from (μ, q):

```
e = √(1 + 2ε h² / M²)
```

This is the exact formula underlying the Phase 2 feature representation.
The features (ε, h, μ, q) analytically determine e.

---

## 4. Effective Potential

```
U_eff(r) = L²/(2μr²) − k/r
```

Centrifugal barrier (r⁻²) repels at small r; gravitational well (r⁻¹)
dominates at large r. Their sum creates a potential well with minimum at
**r_circ = p** (circular orbit radius). Turning points: U_eff(r) = E → r_min, r_max.

The circular orbit is stable (d²U_eff/dr² > 0 at r_circ): small radial
perturbations produce bounded oscillations. This stability does NOT
generalise to three bodies in the general case.

---

## 5. Velocity Verlet Integration

### 5.1 Algorithm

```
a_n      = F(r_n)/m
r_{n+1}  = r_n + v_n dt + ½ a_n dt²
a_{n+1}  = F(r_{n+1})/m
v_{n+1}  = v_n + ½(a_n + a_{n+1}) dt
```

### 5.2 Why Symplectic

Velocity Verlet exactly conserves a modified Hamiltonian H̃ = H + O(dt²).
Energy error stays **bounded** for all time — no secular drift. Non-symplectic
methods (RK4) produce secular energy growth that mimics physical dissipation,
making it impossible to distinguish numerical error from real orbital decay.

### 5.3 Convergence Order

Globally 2nd order: energy error ∝ O(dt²). Halving dt reduces energy
error by ~4×. Verified by `convergence_study()` in `analysis.py`.

### 5.4 Gravitational Softening

Force function: `r² → r² + ε²` with ε = 10⁻⁶. At separations r > 10⁻³,
the force changes by < 0.01% — physically negligible. Collision detection
(Section 12.3) stops integration before bodies reach r ~ ε. The variational
equations `_da()` use the identical softened law, ensuring MEGNO consistency.

### 5.5 Timestep and Integration Window (Phase 3)

```
τ_peri  = √(min_sep³ / (G·M))    periapsis crossing timescale
dt      = min(τ_peri/20,  0.05)   ≥20 steps per periapsis

T_inner = 2π √(r12³/(G·M12))     inner binary period
t_min   = max(15·T_inner, 50.0)   ≥15 inner periods
n_steps = clip(t_min/dt, 5000, 20000)
```

15 inner periods ensures the outer body has multiple encounters with the
binary, allowing energy exchange to develop. The 20 000 step cap prevents
intractably long runs for loose systems.

---

## 6. Orbit Residual — Two-Body Validation

```
θ_i        = arctan2(y_i, x_i)
r_analytic  = p / (1 + e cos θ_i)
Δr_i        = |r_numerical,i − r_analytic(θ_i)|
```

A small residual (< 0.01% of r_min) simultaneously validates: correct
integration, correct e and p formulas, genuine invariant conservation, and
self-consistent initial conditions. Stronger than energy conservation alone.

Orbit residuals are **not defined for three-body systems** (no analytical
orbit equation exists to compare against).

---

## 7. MEGNO — Chaos Indicator

### 7.1 Definition

Mean Exponential Growth of Nearby Orbits (Cincotta & Simó 2000):

```
⟨Y⟩(T) = (2/T) ∫₀ᵀ  t · d(ln|w(t)|)/dt  dt
```

w(t) = phase-space separation from shadow trajectory displaced by δ₀.

**Convergence values:**
```
⟨Y⟩ → 2.0    regular (quasi-periodic) orbit
⟨Y⟩ ~ λt     chaotic orbit, λ = largest Lyapunov exponent
```

### 7.2 Phase 3 Implementation — Variational Equations

Phase 3 uses the exact variational (tangent map) approach via `_da()`:

```
δä_i = G Σⱼ≠ᵢ mⱼ [(δrⱼ−δrᵢ)/r_ij³  −  3((rⱼ−rᵢ)·(δrⱼ−δrᵢ))(rⱼ−rᵢ)/r_ij⁵]
```

MEGNO accumulation:

```
growth = dot(δ, δ̇) / |δ|²
W     += growth · t · dt
⟨Y⟩   = 2W / T_total
```

More accurate than the finite-difference shadow approach (Phase 2);
avoids renormalisation artefacts that caused negative MEGNO in high-e cases.
The deviation vector is renormalised when |δ| drifts outside [δ₀×10⁻³, δ₀×10³].

### 7.3 MEGNO_clean

```
MEGNO_clean = clip(MEGNO, 0, 10)
```

Raw MEGNO can be slightly negative for very short integrations before
the accumulator stabilises. Values above 10 are all strongly chaotic;
clipping removes outliers without losing information.

---

## 8. Phase 2 Feature Representation

### 8.1 Generalisation Experiment

Train on asymmetric mass ratios (q ≤ 0.15, star-planet regime).
Test out-of-distribution on equal-mass systems (q ≥ 0.35, binary-star regime).
Split is fixed and never changed.

### 8.2 Feature Groups

```
PHYSICS_NORM = [ε, h, μ, q, r₀]           normalised, mass-invariant (novel)
PHYSICS_RAW  = [E₀, L₀, μ, r₀]            raw, scale-dependent (ablation)
DYNAMICS     = [dE_max, dE_slope, dL_max,  dynamical diagnostics
                MEGNO_clean, e_inst_std]
ALL_PHYSICS  = PHYSICS_NORM + DYNAMICS
ORBITAL_ELEM = [e, p, a, r_min, r_ratio]   Pinheiro 2025 baseline
```

### 8.3 Key Results

| Model | In-dist R² | OOD R² |
|-------|-----------|-------|
| A1 Ridge PHYSICS_NORM | 0.283 | −1.003 |
| A2 XGBoost PHYSICS_NORM | 0.886 | 0.812 |
| A3 XGBoost PHYSICS_RAW | 0.762 | 0.689 |
| A4 XGBoost ALL_PHYSICS | 0.996 | 0.995 |

**Main finding (A2 vs A3):** Normalised features generalise 0.123 R² better
on OOD mass ratios. A4 is valid (residual leakage removed); supports that
dynamical features carry real signal. A1 failure confirms the nonlinear
structure of e = √(1 + 2ε h²/M²) cannot be captured linearly.

---

## 9. The Three-Body Problem

### 9.1 No Closed-Form Solution

Bruns (1887): no additional algebraic first integrals beyond the ten classical
ones (E, L, momentum, CoM) exist. Poincaré (1890): transcendental integrals
generically fail to exist. Consequence: three-body orbits are generically
chaotic — sensitive to initial conditions, not analytically predictable.

### 9.2 Equations of Motion

```
mᵢ r̈ᵢ = G Σⱼ≠ᵢ mᵢmⱼ(rⱼ−rᵢ)/|rⱼ−rᵢ|³      i = 1,2,3
```

**Only two global invariants:**

```
E_total = ½Σᵢ mᵢ|vᵢ|² − G Σᵢ<ⱼ mᵢmⱼ/rᵢⱼ

L_total = Σᵢ mᵢ(rᵢ×vᵢ)_z
```

Per-body energies are NOT conserved. Energy flows between bodies
through gravitational encounters.

### 9.3 What Changes from Two-Body

| Quantity | Two-body | Three-body |
|---------|---------|-----------|
| Orbit equation | Exact analytical conic | Does not exist |
| Eccentricity e | Global constant | Time-varying diagnostic |
| Period T | Constant | Undefined |
| Orbit residuals | Computable | Undefined |
| MEGNO | Validation only (always →2) | Primary chaos signal |
| Prediction | Fully deterministic | Chaotic / ejection / stable |

---

## 10. Phase 3 Initial Condition Construction

### 10.1 Placement in CoM Frame

**Inner binary (bodies 1,2)** at separation r₁₂ on x-axis:

```
r₁ = [ r₁₂·m₂/M₁₂,  0]
r₂ = [−r₁₂·m₁/M₁₂,  0]     M₁₂ = m₁+m₂
```

r₁₂ is treated as the **periapsis distance** of the inner binary.

**Inner binary velocity at periapsis** (exact Kepler formula):

```
v_peri = v_circ(r₁₂) · √(1 + e_inner)
v_circ  = √(G M₁₂ / r₁₂)
```

Derivation: at periapsis, L = μ₁₂·r₁₂·v_peri and p = r₁₂·(1+e),
so L = μ₁₂·√(G·M₁₂·r₁₂·(1+e)) → v_peri = √(G·M₁₂·(1+e)/r₁₂). QED.

Verified numerically:

| e_inner | v/v_circ | e reconstructed |
|--------|---------|----------------|
| 0.0 | 1.0000 | 0.0000 |
| 0.3 | 1.1402 | 0.3000 |
| 0.6 | 1.2649 | 0.6000 |
| 0.9 | 1.3784 | 0.9000 |

The inner binary is always bound: binary energy < 0 for e_inner < 1.

**Outer body (body 3):**
```
r₃ = [r₃·cos(α),  r₃·sin(α)]          α = v3_angle ∈ [0, 2π)
v₃ = v3_mag · [−sin(α),  cos(α)]      tangential velocity
```

**Three-body CoM shift:**
```
r_com = (m₁r₁+m₂r₂+m₃r₃)/M    subtracted from all positions
v_com = (m₁v₁+m₂v₂+m₃v₃)/M    subtracted from all velocities
```

After shift: Σmᵢrᵢ = 0, Σmᵢvᵢ = 0. Total momentum = 0.

### 10.2 Escape Velocity Constraint

All regimes enforce **v3_frac < √2 ≈ 1.414**, where:

```
v3_frac = v3_mag / v_circ(r₃)
v_esc   = √2 · v_circ(r₃)
```

A body with v > v_esc escapes without gravitational interaction — a trivial
flyby, not a three-body problem. Enforcing v3_frac < √2 guarantees that
ejections only arise from gravitational energy exchange during close encounters.
This is the physical process the model needs to learn.

---

## 11. Phase 3 Sampling Regimes

Four regimes cover distinct regions of three-body phase space.
Regime fractions: hierarchical 25%, asymmetric 30%, compact_equal 25%, scatter 20%.

### 11.1 Hierarchical (25%)

```
q₁₂ ∈ [0.1, 1.0]     inner binary mass ratio
m₃  ∈ [0.1%, 5%] of M
r₃/r₁₂ ∈ [3, 15]     outer body far from binary
v3_frac ∈ [0.4, 1.1]  well below escape
e_inner ∈ [0.0, 0.5]
```

**Physics:** Hill stability regime. Most systems satisfy Hill's criterion
(r₃ > 2.4·r₁₂·(M₁₂/m₃)^(1/3)) and remain stable. The outer body orbits
the binary CoM with slow Kozai-Lidov perturbations.

**Expected outcomes:** ~80% stable, ~15% chaotic, ~5% ejection.

**Purpose:** Provides the stable class. Without hierarchical systems, the
model never sees genuinely long-lived bound configurations.

### 11.2 Asymmetric (30%)

```
q₁₂ ∈ [0.01, 0.3]    one dominant mass
m₃  ∈ [0.5%, 10%] of M
r₃/r₁₂ ∈ [1.5, 6.0]
v3_frac ∈ [0.5, 1.3]
e_inner ∈ [0.0, 0.6]
```

**Physics:** One body (m₂) dominates gravitationally. Resembles a
star-planet-moon configuration. Mass asymmetry creates secular resonances
that drive slow chaos. Wide r₃/r₁₂ range spans the stability boundary.

**Expected outcomes:** ~55% stable, ~35% chaotic, ~10% ejection.

**Purpose:** Mass-ratio diversity. Tests whether the classifier generalises
across asymmetric configurations, connecting to Phase 2 results.

### 11.3 Compact Equal (25%)

```
q₁₂ ∈ [0.5, 1.0]    near-equal inner pair
m₃ ≈ M − m₁ − m₂    comparable to m₁, m₂
r₃/r₁₂ ∈ [0.5, 2.0]  COMPACT: outer body close to binary
v3_frac ∈ [0.6, 1.3]
e_inner ∈ [0.0, 0.6]
```

**Physics:** All three bodies at comparable separations. r₃/r₁₂ < 2 means
the outer body is within the strong-coupling zone — every inner binary orbit
brings it into a close encounter. This is the classic chaotic three-body
problem (Poincaré, Szebehely). "Democratic decay" dominates: all three bodies
exchange energy until one acquires enough to escape via the slingshot reaction:

```
(m₁−m₂ binary) + m₃  →  (tighter m₁−m₂ binary) + m₃ ejected
```

The ejected body takes kinetic energy; the remaining binary tightens
(Heggie's theorem: hard binaries get harder).

**Expected outcomes:** ~5% stable, ~50% ejection, ~45% chaotic.

**Purpose:** Primary source of physically meaningful ejection labels.

### 11.4 Scatter (20%)

```
m₁ ≈ m₂ ≈ m₃         equal masses
r₃/r₁₂ ∈ [0.8, 2.5]  compact geometry
v3_frac ∈ [0.8, 1.35]  fast but bound (max = 95% of escape velocity)
e_inner ∈ [0.0, 0.5]
```

**Physics:** Equal masses maximise energy exchange per encounter (equal-mass
scattering transfers maximum momentum). Compact geometry ensures frequent
encounters. v3_frac near 0.95·v_esc means one moderately strong slingshot
is sufficient to push body 3 over the escape threshold. This samples the
stability edge — the transition between chaotic-bound and ejection — which
is the most informative region for ML classification.

**Expected outcomes:** ~10% stable, ~45% ejection, ~45% chaotic.

**Purpose:** Provides the unstable-boundary region and high-ejection
configurations from genuine gravitational dynamics.

---

## 12. Phase 3 Outcome Labels

Labels are assigned in **priority order**. Earlier checks take precedence.
This prevents chaotic near-miss systems from being incorrectly classified
as collisions (the most impactful historical bug, Section 14.3).

### 12.1 Priority 1 — Ejection (outcome_class = 1)

**Check (a): Global energy positive**

```
E₀ > 0
```

Positive total energy → system is globally unbound and must eventually
eject regardless of current trajectory. Caught immediately at label time.

**Check (b): Body has left the system**

```
max(d₁₂, d₁₃, d₂₃) > 10 × initial_scale
initial_scale = max(|r₁₂|₀, |r₁₃|₀, |r₂₃|₀)
```

A body 10 system-widths away has negligible gravitational influence on
the remaining pair. 10× is conservative — any body this far out will not
return on any relevant timescale.

**Why ejection is checked first:** A system in the process of ejecting
first passes through a chaotic phase as MEGNO grows. If chaotic were checked
first, pre-ejection trajectories would be mislabelled as chaotic.

### 12.2 Priority 2 — Chaotic (outcome_class = 2)

If not ejection:

```
MEGNO_final > 3.0
```

MEGNO = 2 for regular orbits. Threshold 3.0 is 50% above the regular value,
standard in the literature (Goździewski 2001). Systems with MEGNO ∈ (2, 3)
are labelled stable — conservative but avoids ambiguous boundary cases.

**Why chaotic before collision:** Chaotic systems frequently have near-misses
— bodies pass close without merging. These are symptoms of chaos, not mergers.
Checking collision first classified 405/441 "collisions" as chaotic incorrectly
(all had MEGNO_clean > 3.0). See Section 14.3.

### 12.3 Priority 3 — Collision (outcome_class = 1)

If not ejection or chaotic:

```
r_collision = max(0.001,  initial_scale × 0.005)
```

Any pairwise separation below r_collision → collision. The threshold is
scale-relative: bodies must approach within 0.5% of the system's own scale.
This avoids triggering on tight binary periapsis passages (a binary at r₁₂ = 0.3
can naturally reach periapsis during its orbit — this is not a merger).

The integrator independently detects collisions and truncates the trajectory.
MEGNO therefore reflects the dynamics up to the moment of merger.

### 12.4 Priority 4 — Stable (outcome_class = 0)

Default. System completed the integration window with MEGNO ≤ 3.0 and no
disruption event.

### 12.5 Three-Class Training Mapping

```
outcome_class = 0  →  stable      long-lived bound system
outcome_class = 1  →  unstable    ejection OR collision (both = disruption)
outcome_class = 2  →  chaotic     bound chaos within window
```

Collision and ejection merge into "unstable" because genuine mergers are rare
(< 1% of samples) — too sparse to train as an independent class. The raw
four-class labels are preserved in `outcome` (string) and `outcome_class4` (int).

---

## 13. Phase 3 Feature Extraction

All dynamical features use the **early window only** (first 20% of trajectory).
MEGNO_final is the only exception — computed by the integrator over all n_steps.

The 20% constraint is the scientific core of Phase 3. With the full trajectory
available, outcome detection is trivial. The challenge is reading early
dynamical signatures to predict eventual fate — the early-warning scenario.

### 13.1 Conservation Diagnostics

```
dE_max   = max|ΔE / E_scale|
           E_scale = |E₀| if |E₀| > 1e-10·KE₀ else KE₀

dE_slope = linear slope of ΔE(t)  [secular drift vs bounded oscillation]

dL_max   = max|ΔL / |L₀||
```

**Quality gate:** `dL_max ≤ 0.02` rejects catastrophic integration failures.
Looser than Phase 2 (10⁻⁶) because three-body dynamics produce more complex
coupling between degrees of freedom, but still rejects NaN and extreme drift.

### 13.2 Chaos Indicator

```
MEGNO_clean = clip(MEGNO_final, 0, 10)
```

The most powerful single feature for stable vs chaotic classification.
MEGNO_clean ≈ 2 for stable; grows above 3 for chaotic. NaN (from short
trajectories) replaced by 2.0 (best estimate for unknown/regular systems).

### 13.3 Pair Eccentricity Variation

```
e_ij_std = std(e_ij(t))    over early window, pairs (1-2), (1-3), (2-3)
```

At each of up to 50 sampled steps, instantaneous eccentricity is computed
from finite-difference velocity estimates:

```
e_ij = √(1 + 2 E_ij L_ij² / (μᵢⱼ kᵢⱼ²))
```

For stable hierarchical: e₁₂_std ≈ 0 (binary eccentricity conserved),
e₁₃_std and e₂₃_std bounded. For chaotic: all three grow rapidly as
encounters perturb every pair. Large e_ij_std in the early window is a
strong ejection predictor. `_inst_e_safe()` returns 0.0 when r < 10⁻¹⁰
or masses ≤ 0 (division-by-zero guard).

### 13.4 Closest Approach Distances

```
r_min_ij = minimum pairwise separation of pair (i,j) in early window
```

A small r_min_12 in the early window indicates a close encounter within
the inner binary — strong disruption signal. A small r_min_13 or r_min_23
means body 3 has already had a slingshot encounter with a binary component.

### 13.5 Initial Condition Features

```
epsilon_total = E₀ / M_total      specific total energy
h_total       = |L₀| / M_total    specific total angular momentum
q12           = m₁/m₂             inner binary mass ratio
q13           = m₁/m₃             cross mass ratio
M_total       = m₁+m₂+m₃
r12_init      = initial inner binary separation
r3_sep        = initial outer body distance
v3_frac       = v₃ / v_circ(r₃)   velocity fraction (< √2 = bound)
e_inner       = inner binary eccentricity
```

`epsilon_total` and `h_total` are the three-body analogues of Phase 2's
ε and h. They are system-level averages rather than exact invariants, but
carry the same intuition: deeply negative epsilon_total → tightly bound →
unlikely to eject; epsilon_total near zero → marginal stability.

---

## 14. Bugs Fixed During Development

### 14.1 Phase 2 Quality Filter Destroyed Parabolic Class

**Bug:** `dE_max > 0.05` rejected 97% of parabolic samples. E₀ ≈ 0 for
parabolic orbits makes ΔE/|E₀| diverge even when the absolute error is tiny.
**Fix:** `dL_max ≤ 10⁻⁶`. Angular momentum is conserved to machine precision
for all orbit types by Velocity Verlet — it's a reliable quality metric that
doesn't blow up near the parabolic boundary.

### 14.2 Phase 2 Experiment A — Bound-Only Regression

**Bug:** Regression ran on bound orbits only (e < 1), dropping 655 unbound
samples before training.
**Fix:** Regression runs on full e ∈ [0, ∞). Hyperbolic orbits and the e ≈ 1
boundary are precisely the region relevant to three-body stability prediction.

### 14.3 Phase 3 Labeller Priority Order (Critical)

**Bug:** Collision was checked first. 405 of 441 "collisions" had
MEGNO_clean > 3.0 — they were chaotic systems with close approaches.
**Fix:** Priority order: ejection → chaotic → collision → stable.
Close approaches are a symptom of chaos, not physical mergers. Checking
collision first caused a 9-fold overcount of the collision class.

### 14.4 Phase 3 Collision Threshold — Absolute vs Scale-Relative

**Bug:** Fixed `R_COLL = 0.05`. For r₁₂ = 0.3, this is 17% of binary
separation — triggered constantly on periapsis passages of a normal orbit.
**Fix:** `r_collision = max(0.001, initial_scale × 0.005)`. Collision
requires bodies to approach within 0.5% of the system's own initial scale.

### 14.5 Division by Zero in Labeller

**Bug (a):** `_escape_check()` called `norm(ri − ra)` which is zero when
bodies overlap. Function removed entirely.
**Bug (b):** `inst_e_pair()` computed `k/r` with r = 0.
**Fix:** `_inst_e_safe()` returns 0.0 when r < 10⁻¹⁰, masses ≤ 0, or k < 10⁻³⁰.

### 14.6 Broken Seed Reproducibility

**Bug:** `np.random.uniform` called inside `_com_3body` instead of the seeded
`rng` argument. Every run drew from an uncontrolled global random state.
**Fix:** `rng` passed into `_com_3body` and used throughout all samplers.

### 14.7 Wrong Inner Binary Eccentricity Formula

**Bug:** `v_factor = sqrt((1+e)/(1-e))`. This is the **apoapsis-to-periapsis
velocity ratio** of an ellipse, not the periapsis speed. At e = 0.8 it gives
v_inner 2.2× too high, producing violently unstable inner binaries.
**Fix:** `v_inner = v_circ × sqrt(1 + e_inner)`. Exact Kepler formula,
verified numerically to reconstruct the input eccentricity exactly.

### 14.8 Unbound Initial Conditions (v3_frac)

**Bug:** Scatter regime: `v3_frac ∈ [1.2, 3.0]` → 88% of samples were
globally unbound at t = 0 (trivial ejections, no gravitational physics).
Near-equal regime similarly had 64% unbound.
**Fix:** All regimes enforce v3_frac < √2 = 1.414. Scatter: [0.8, 1.35].
Near_equal replaced by compact_equal [0.6, 1.30] with compact geometry.

---

## 15. References

**Integration:**
Verlet, L. (1967). Computer "experiments" on classical fluids. *Phys. Rev.* 159, 98.

**MEGNO:**
Cincotta, P.M., Simó, C. (2000). Simple tools to study global dynamics in
Hamiltonian systems. *A&AS* 147, 205.
Goździewski, K. et al. (2001). Global dynamics of planetary systems. *A&A* 378, 569.

**Figure-8 orbit:**
Moore, C. (1993). Braids in classical dynamics. *Phys. Rev. Lett.* 70, 3675.
Chenciner, A., Montgomery, R. (2000). A remarkable periodic solution of the
three-body problem. *Ann. Math.* 152, 881.

**Three-body stability:**
Hill, G.W. (1878). Researches in the lunar theory. *Am. J. Math.* 1, 5.
Heggie, D.C. (1975). Binary evolution in stellar dynamics. *MNRAS* 173, 729.
Valtonen, M., Karttunen, H. (2006). *The Three-Body Problem*. Cambridge UP.
Mardling, R.A., Aarseth, S.J. (2001). Tidal interactions in star cluster
simulations. *MNRAS* 321, 398.

**ML comparison:**
Tamayo, D. et al. (2020). Predicting the long-term stability of compact
multiplanet systems. *PNAS* 117, 18194. (SPOCK)
Pinheiro, F. et al. (2025). *A&A*. (Orbital element classification baseline)
Karniadakis, G.E. et al. (2021). Physics-informed machine learning.
*Nature Reviews Physics* 3, 422.

**N-body methods:**
Aarseth, S.J. (2003). *Gravitational N-Body Simulations*. Cambridge UP.