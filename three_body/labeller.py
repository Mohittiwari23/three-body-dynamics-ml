"""
three_body/labeller.py
=======================
Assigns stability outcomes to completed three-body simulations.

Outcome labels
--------------
  0 = stable    : bound, non-chaotic, no merger detected
  1 = ejection  : one body escapes the system
  2 = collision : two bodies physically merge (< 0.5% of system scale)
  3 = chaotic   : MEGNO > 3.0, bound, no merger

Detection priority
------------------
The order matters. Earlier checks take precedence.

  1. Ejection first
       (a) Total energy E0 > 0 — system is globally unbound, MUST eject.
       (b) Any pairwise separation exceeds 10× initial system scale.
     Ejection is checked first because a chaotic system that is also
     escaping should be labelled ejection, not chaotic.

  2. Chaotic second
       MEGNO > threshold (3.0).
     Checked before collision because chaotic systems frequently
     experience close approaches. Those near-misses are a symptom
     of chaos, not physical mergers. Checking collision first (the
     previous bug) caused 405/441 "collisions" to be mis-labelled.

  3. Collision third
       Any pair reaches separation < r_collision.
       r_collision = max(0.001, initial_scale × 0.005).
     Scale-relative threshold avoids triggering on tight-binary
     periapsis passages. Only bound, non-chaotic systems reach here.

  4. Stable — default if none of the above triggered.

Previous bugs fixed in this version
------------------------------------
  BUG 1 (critical): Collision was checked first. 405 of 441 "collisions"
    had MEGNO_clean > 3 — they were chaotic systems, not mergers.
    Fixed: priority order is now ejection → chaotic → collision → stable.

  BUG 2: R_COLL = 0.05 was an absolute threshold. For r12 = 0.3 this
    fires at 17% of binary separation — triggered on normal periapsis.
    Fixed: r_collision = max(0.001, initial_scale × 0.005).

  BUG 3 (division by zero): _escape_check() divided by pair separation
    which is zero when bodies overlap. Removed entirely — the E0 > 0
    check and 10× separation threshold cover ejection correctly.

  BUG 4 (division by zero): inst_e_pair() divided by r when r = 0.
    Fixed: early-exit with return 0.0 when separation < 1e-10.

Feature extraction
------------------
All dynamical features use the EARLY WINDOW only
(first `window_fraction` of total steps). This simulates the
early-warning scenario: predict outcome from just the start of
a trajectory — exactly what is needed for 3-body ML.
"""

from __future__ import annotations
import numpy as np
from .integrator3 import ThreeBodyResult, inst_e_pair as _inst_e_raw

OUTCOME_NAMES   = {0: "stable", 1: "ejection", 2: "collision", 3: "chaotic"}
OUTCOME_CLASSES = {"stable": 0, "ejection": 1, "collision": 2, "chaotic": 3}
MEGNO_CHAOTIC   = 3.0


def _inst_e_safe(ra, rb, va, vb, ma, mb, G):
    """
    Instantaneous eccentricity of a two-body sub-pair.
    Returns 0.0 safely when bodies are coincident or masses are zero.
    """
    r = np.linalg.norm(ra - rb)
    if r < 1e-10 or ma <= 0 or mb <= 0:
        return 0.0
    mu = ma * mb / (ma + mb)
    k  = G * ma * mb
    if k < 1e-30:
        return 0.0
    vv = va - vb
    vr = ra - rb
    E  = 0.5 * mu * np.dot(vv, vv) - k / r
    L  = mu * (vr[0] * vv[1] - vr[1] * vv[0])
    d  = 1.0 + 2.0 * E * L**2 / (mu * k**2)
    return float(np.sqrt(max(0.0, d)))


def label_result(
    result: ThreeBodyResult,
    window_fraction: float = 0.20,
    megno_threshold: float = MEGNO_CHAOTIC,
) -> ThreeBodyResult:
    """
    Label a simulation outcome and extract early-window features.

    Modifies result in-place and returns it.
    """
    traj   = result.traj      # (N, 3, 2)
    E_hist = result.E_hist
    L_hist = result.L_hist
    time_  = result.time
    sys_   = result.system
    N      = traj.shape[0]
    G      = sys_.G
    m1, m2, m3 = sys_.m1, sys_.m2, sys_.m3
    dt     = sys_.dt

    # ── System scale (used for both ejection and collision thresholds) ──────
    r12_0 = np.linalg.norm(traj[0, 0] - traj[0, 1])
    r13_0 = np.linalg.norm(traj[0, 0] - traj[0, 2])
    r23_0 = np.linalg.norm(traj[0, 1] - traj[0, 2])
    initial_scale = max(r12_0, r13_0, r23_0, 1e-6)

    r_ejection  = initial_scale * 10.0                   # 10× scale → escaped
    r_collision = max(0.001, initial_scale * 0.005)      # 0.5% scale → merged

    outcome       = "stable"
    ejection_step = None
    collision_step = None

    # ── STEP 1: EJECTION ────────────────────────────────────────────────────
    # (a) Global energy check: E0 > 0 means system is unbound. Must eject.
    if result.E0 > 0:
        outcome       = "ejection"
        ejection_step = 0

    # (b) Separation-based scan of full trajectory
    if outcome == "stable":
        for i in range(N):
            r1, r2, r3 = traj[i, 0], traj[i, 1], traj[i, 2]
            d12 = np.linalg.norm(r1 - r2)
            d13 = np.linalg.norm(r1 - r3)
            d23 = np.linalg.norm(r2 - r3)
            if max(d12, d13, d23) > r_ejection:
                outcome       = "ejection"
                ejection_step = i
                break

    # ── STEP 2: CHAOTIC ─────────────────────────────────────────────────────
    # MEGNO > threshold means bound chaotic motion.
    # Must run before collision — chaotic close approaches are not mergers.
    if outcome == "stable":
        meg = result.MEGNO_final
        if not np.isnan(meg) and meg > megno_threshold:
            outcome = "chaotic"

    # ── STEP 3: COLLISION ───────────────────────────────────────────────────
    # Only reached by bound, non-chaotic systems.
    # Scale-relative threshold prevents false triggers on periapsis.
    if outcome == "stable":
        for i in range(N):
            r1, r2, r3 = traj[i, 0], traj[i, 1], traj[i, 2]
            d12 = np.linalg.norm(r1 - r2)
            d13 = np.linalg.norm(r1 - r3)
            d23 = np.linalg.norm(r2 - r3)
            if d12 < r_collision or d13 < r_collision or d23 < r_collision:
                outcome        = "collision"
                collision_step = i
                break

    # ── STEP 4: STABLE — default ────────────────────────────────────────────

    result.outcome        = outcome
    result.outcome_class  = OUTCOME_CLASSES[outcome]
    result.ejection_step  = ejection_step
    result.collision_step = collision_step

    # ── Early-window feature extraction ─────────────────────────────────────
    K      = max(2, int(N * window_fraction))
    traj_w = traj[:K]
    E_w    = E_hist[:K]
    L_w    = L_hist[:K]
    t_w    = time_[:K]

    # Energy scale (avoid dividing by near-zero E0 for near-parabolic systems)
    KE0   = (0.5*m1*np.dot(sys_.v1_0, sys_.v1_0) +
             0.5*m2*np.dot(sys_.v2_0, sys_.v2_0) +
             0.5*m3*np.dot(sys_.v3_0, sys_.v3_0))
    E_sc  = abs(result.E0) if abs(result.E0) > 1e-10 * KE0 else KE0
    L_sc  = max(abs(result.L0), 1e-10)

    dE = (E_w - result.E0) / E_sc
    result.dE_max   = float(np.max(np.abs(dE)))
    result.dL_max   = float(np.max(np.abs(L_w - result.L0)) / L_sc)
    result.dE_slope = float(np.polyfit(t_w, dE, 1)[0]) if len(t_w) > 2 else 0.0

    # Pair separations in early window
    r12_w = np.linalg.norm(traj_w[:, 0] - traj_w[:, 1], axis=1)
    r13_w = np.linalg.norm(traj_w[:, 0] - traj_w[:, 2], axis=1)
    r23_w = np.linalg.norm(traj_w[:, 1] - traj_w[:, 2], axis=1)
    result.r_min_12 = float(r12_w.min())
    result.r_min_13 = float(r13_w.min())
    result.r_min_23 = float(r23_w.min())

    # Instantaneous eccentricity variation — finite-difference velocities
    # _inst_e_safe guards against zero separation and zero mass
    step_ = max(1, K // 50)
    e12_arr, e13_arr, e23_arr = [], [], []
    for i in range(0, K - 2, step_):
        r1 = traj_w[i, 0]; r2 = traj_w[i, 1]; r3 = traj_w[i, 2]
        if i > 0:
            v1 = (traj_w[i+1, 0] - traj_w[i-1, 0]) / (2 * dt)
            v2 = (traj_w[i+1, 1] - traj_w[i-1, 1]) / (2 * dt)
            v3 = (traj_w[i+1, 2] - traj_w[i-1, 2]) / (2 * dt)
        else:
            v1 = (traj_w[1, 0] - r1) / dt
            v2 = (traj_w[1, 1] - r2) / dt
            v3 = (traj_w[1, 2] - r3) / dt
        e12_arr.append(_inst_e_safe(r1, r2, v1, v2, m1, m2, G))
        e13_arr.append(_inst_e_safe(r1, r3, v1, v3, m1, m3, G))
        e23_arr.append(_inst_e_safe(r2, r3, v2, v3, m2, m3, G))

    result.e12_std = float(np.std(e12_arr)) if e12_arr else 0.0
    result.e13_std = float(np.std(e13_arr)) if e13_arr else 0.0
    result.e23_std = float(np.std(e23_arr)) if e23_arr else 0.0

    return result