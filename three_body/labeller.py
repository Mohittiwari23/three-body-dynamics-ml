"""
three_body/labeller.py
=======================
Assigns stability outcomes to completed three-body simulations.

Outcome labels
--------------
  0 = stable      : bound, non-chaotic, no merger in window
  1 = ejection    : one body escapes the system
  2 = collision   : two bodies actually merge (< 0.5% of system scale)
  3 = chaotic     : MEGNO > 3.0, bound, no merger

Detection priority (order matters)
-----------------------------------
  1. Ejection first  — body escaping is the strongest, clearest signal.
                       Detected by: total energy > 0 (globally unbound),
                       OR max separation > 10x initial scale,
                       OR any body's escape velocity exceeded.
  2. Chaotic second  — MEGNO captures bound chaotic motion.
                       Must run before collision because chaotic systems
                       frequently have close approaches that are symptoms
                       of chaos, NOT physical mergers.
  3. Collision third — only reached by bound, non-chaotic systems.
                       Threshold is scale-relative (0.5% of initial scale)
                       to avoid triggering on normal tight-binary periapsis.
  4. Stable default  — everything else.

Previous bugs fixed
-------------------
  BUG 1: Collision was checked first. 405/441 "collisions" had MEGNO > 3
          — they were chaotic systems with close approaches, not mergers.
          Fixed: collision check moved after ejection and MEGNO checks.

  BUG 2: R_COLL = 0.05 was absolute. For r12_init=0.3, that's 17% of
          the binary separation — triggered constantly on periapsis.
          Fixed: r_collision = max(0.001, initial_scale * 0.005).

  BUG 3: Ejection used 20x separation threshold. 642 "stable" systems had
          v3_frac > 1.4, meaning they were almost certainly ejecting but
          the body hadn't traveled 20x yet. Also: total energy > 0 means
          the system IS unbound regardless of current position.
          Fixed: added epsilon_total > 0 check, reduced threshold to 10x,
          added per-body escape velocity check.

  BUG 4: 71 systems labeled stable/chaotic had epsilon_total > 0 (globally
          unbound). An unbound system must eventually eject.
          Fixed: epsilon_total > 0 → ejection (if no collision).

Feature extraction
------------------
All dynamical features are computed from the EARLY WINDOW only
(first `window_fraction` of the total trajectory). This is the
early-warning prediction scenario: given only the start of a
trajectory, predict the final outcome.
"""

from __future__ import annotations
import numpy as np
from .integrator3 import ThreeBodyResult, inst_e_pair

# Outcome class map
OUTCOME_NAMES   = {0: "stable", 1: "ejection", 2: "collision", 3: "chaotic"}
OUTCOME_CLASSES = {"stable": 0, "ejection": 1, "collision": 2, "chaotic": 3}

# MEGNO threshold for chaotic classification
MEGNO_CHAOTIC = 3.0


def _escape_check(r1, r2, r3, v1, v2, v3, m1, m2, m3, G):
    """
    Check whether any single body has enough kinetic energy to escape
    the gravitational pull of the other two combined.

    This is a conservative (generous) escape criterion: we check if
    a body's specific KE exceeds |PE| from the other two.
    Returns (bool, which_body) where which_body is 1, 2, or 3.
    """
    bodies = [(r1,v1,m1, r2,r3,m2,m3),
              (r2,v2,m2, r1,r3,m1,m3),
              (r3,v3,m3, r1,r2,m1,m2)]
    for i, (ri, vi, mi, ra, rb, ma, mb) in enumerate(bodies):
        KE_i  = 0.5 * mi * np.dot(vi, vi)
        PE_ia = -G * mi * ma / np.linalg.norm(ri - ra)
        PE_ib = -G * mi * mb / np.linalg.norm(ri - rb)
        # Body i is escaping if KE_i > |PE_ia| + |PE_ib|
        if KE_i > abs(PE_ia) + abs(PE_ib):
            return True, i+1
    return False, None


def label_result(
    result: ThreeBodyResult,
    window_fraction: float = 0.20,
    megno_threshold: float = MEGNO_CHAOTIC,
) -> ThreeBodyResult:
    """
    Label a completed simulation and compute early-window features.

    Detection runs in priority order:
      ejection → chaotic → collision → stable

    Parameters
    ----------
    result          : ThreeBodyResult from run_simulation_3body
    window_fraction : fraction of trajectory for feature extraction
    megno_threshold : MEGNO above which system is classified chaotic
    """
    traj   = result.traj      # (N, 3, 2)
    E_hist = result.E_hist
    L_hist = result.L_hist
    time_  = result.time
    sys_   = result.system
    N      = traj.shape[0]
    G      = sys_.G
    m1, m2, m3 = sys_.m1, sys_.m2, sys_.m3
    dt = sys_.dt

    # ── Initial system scale ────────────────────────────────────────────────
    r12_0 = np.linalg.norm(traj[0,0] - traj[0,1])
    r13_0 = np.linalg.norm(traj[0,0] - traj[0,2])
    r23_0 = np.linalg.norm(traj[0,1] - traj[0,2])
    initial_scale = max(r12_0, r13_0, r23_0, 1e-6)

    # Scale-relative collision threshold: 0.5% of initial system scale
    # This is a genuine merger — not a periapsis passage
    r_collision = max(0.001, initial_scale * 0.005)

    # Ejection separation threshold: 10x initial scale
    r_ejection = initial_scale * 10.0

    outcome       = "stable"
    ejection_step = None
    collision_step = None

    # ── STEP 1: Ejection detection ──────────────────────────────────────────
    # Three criteria — any one is sufficient:
    #   (a) Total system energy > 0 (globally unbound — MUST eventually eject)
    #   (b) Any body's KE exceeds gravitational binding energy from both others
    #   (c) Max pairwise separation exceeds 10x initial scale

    # (a) Globally unbound — check initial energy sign
    if result.E0 > 0:
        outcome = "ejection"
        ejection_step = 0

    # (b) + (c) Scan trajectory for escape events
    if outcome == "stable":
        for i in range(1, N):
            r1, r2, r3 = traj[i,0], traj[i,1], traj[i,2]
            r12 = np.linalg.norm(r1-r2)
            r13 = np.linalg.norm(r1-r3)
            r23 = np.linalg.norm(r2-r3)
            max_sep = max(r12, r13, r23)

            # (c) Separation-based ejection
            if max_sep > r_ejection:
                outcome = "ejection"
                ejection_step = i
                break

            # (b) Velocity-based ejection (check every 50 steps to stay fast)
            if i % 50 == 0:
                v1 = (traj[i,0] - traj[i-1,0]) / dt
                v2 = (traj[i,1] - traj[i-1,1]) / dt
                v3 = (traj[i,2] - traj[i-1,2]) / dt
                escaped, _ = _escape_check(r1,r2,r3, v1,v2,v3, m1,m2,m3, G)
                if escaped:
                    outcome = "ejection"
                    ejection_step = i
                    break

    # ── STEP 2: Chaotic detection ───────────────────────────────────────────
    # Run BEFORE collision. Close approaches are a symptom of chaos —
    # a chaotic system that has a near-miss should be labelled "chaotic",
    # not "collision".
    if outcome == "stable":
        meg = result.MEGNO_final
        if not np.isnan(meg) and meg > megno_threshold:
            outcome = "chaotic"

    # ── STEP 3: Collision detection ─────────────────────────────────────────
    # Only reached by bound, non-chaotic systems.
    # Uses scale-relative threshold to avoid triggering on periapsis.
    if outcome == "stable":
        for i in range(N):
            r1, r2, r3 = traj[i,0], traj[i,1], traj[i,2]
            d12 = np.linalg.norm(r1-r2)
            d13 = np.linalg.norm(r1-r3)
            d23 = np.linalg.norm(r2-r3)
            if d12 < r_collision or d13 < r_collision or d23 < r_collision:
                outcome = "collision"
                collision_step = i
                break

    # ── Commit label ────────────────────────────────────────────────────────
    result.outcome       = outcome
    result.outcome_class = OUTCOME_CLASSES[outcome]
    result.ejection_step  = ejection_step
    result.collision_step = collision_step

    # ── Early window feature extraction ─────────────────────────────────────
    K      = max(2, int(N * window_fraction))
    traj_w = traj[:K]
    E_w    = E_hist[:K]
    L_w    = L_hist[:K]
    t_w    = time_[:K]

    KE0 = (0.5*m1*np.dot(sys_.v1_0,sys_.v1_0) +
           0.5*m2*np.dot(sys_.v2_0,sys_.v2_0) +
           0.5*m3*np.dot(sys_.v3_0,sys_.v3_0))
    E_sc  = abs(result.E0) if abs(result.E0) > 1e-10*KE0 else KE0
    L_sc  = max(abs(result.L0), 1e-10)

    dE = (E_w - result.E0) / E_sc
    result.dE_max   = float(np.max(np.abs(dE)))
    result.dL_max   = float(np.max(np.abs(L_w - result.L0)) / L_sc)
    result.dE_slope = float(np.polyfit(t_w, dE, 1)[0]) if len(t_w) > 2 else 0.0

    # Pair separations in early window
    r12_w = np.linalg.norm(traj_w[:,0]-traj_w[:,1], axis=1)
    r13_w = np.linalg.norm(traj_w[:,0]-traj_w[:,2], axis=1)
    r23_w = np.linalg.norm(traj_w[:,1]-traj_w[:,2], axis=1)
    result.r_min_12 = float(r12_w.min())
    result.r_min_13 = float(r13_w.min())
    result.r_min_23 = float(r23_w.min())

    # Instantaneous eccentricity variation (finite-difference velocities)
    step_ = max(1, K // 50)
    e12_arr, e13_arr, e23_arr = [], [], []
    for i in range(0, K-2, step_):
        r1 = traj_w[i,0]; r2 = traj_w[i,1]; r3 = traj_w[i,2]
        if i > 0:
            v1 = (traj_w[i+1,0] - traj_w[i-1,0]) / (2*dt)
            v2 = (traj_w[i+1,1] - traj_w[i-1,1]) / (2*dt)
            v3 = (traj_w[i+1,2] - traj_w[i-1,2]) / (2*dt)
        else:
            v1 = (traj_w[1,0] - r1) / dt
            v2 = (traj_w[1,1] - r2) / dt
            v3 = (traj_w[1,2] - r3) / dt
        e12_arr.append(inst_e_pair(r1,r2, v1,v2, m1,m2, G))
        e13_arr.append(inst_e_pair(r1,r3, v1,v3, m1,m3, G))
        e23_arr.append(inst_e_pair(r2,r3, v2,v3, m2,m3, G))

    result.e12_std = float(np.std(e12_arr)) if e12_arr else 0.0
    result.e13_std = float(np.std(e13_arr)) if e13_arr else 0.0
    result.e23_std = float(np.std(e23_arr)) if e23_arr else 0.0

    return result