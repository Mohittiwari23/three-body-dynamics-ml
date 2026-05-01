"""
three_body/labeller.py
"""

from __future__ import annotations
import numpy as np
from .integrator3 import ThreeBodyResult

OUTCOME_NAMES   = {0: "stable", 1: "ejection", 2: "collision", 3: "chaotic"}
OUTCOME_CLASSES = {"stable": 0, "ejection": 1, "collision": 2, "chaotic": 3}
MEGNO_CHAOTIC   = 3.0
E_PAIR_CAP      = 10.0   # cap on instantaneous eccentricity per sample
WINDOWS = [5, 10, 15, 20, 25, 30]

def _inst_e_safe(ra, rb, va, vb, ma, mb, G, e_cap=E_PAIR_CAP):
    """
    Instantaneous eccentricity of a two-body sub-pair.
    Capped at e_cap to prevent blowup for separating/unbound pairs.
    Returns 0.0 when bodies are coincident or masses are zero.
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
    return float(min(np.sqrt(max(0.0, d)), e_cap))


def _extract_window_features(traj_w, E_w, L_w, t_w, sys_, dt,
                              E0, L0, suffix):
    """
    Extract dynamical features from a trajectory window.

    Returns a dict of features with keys suffixed by `suffix`
    (e.g. '_w5' for the 5% window, '_w20' for 20%).
    """
    m1, m2, m3 = sys_.m1, sys_.m2, sys_.m3
    G = sys_.G

    KE0 = (0.5*m1*np.dot(sys_.v1_0, sys_.v1_0) +
           0.5*m2*np.dot(sys_.v2_0, sys_.v2_0) +
           0.5*m3*np.dot(sys_.v3_0, sys_.v3_0))
    E_sc = abs(E0) if abs(E0) > 1e-10 * KE0 else KE0
    L_sc = max(abs(L0), 1e-10)

    dE = (E_w - E0) / E_sc
    dE_max   = float(np.max(np.abs(dE)))
    dL_max   = float(np.max(np.abs(L_w - L0)) / L_sc)
    dE_slope = float(np.polyfit(t_w, dE, 1)[0]) if len(t_w) > 2 else 0.0

    # Pair separations
    r12_w = np.linalg.norm(traj_w[:,0] - traj_w[:,1], axis=1)
    r13_w = np.linalg.norm(traj_w[:,0] - traj_w[:,2], axis=1)
    r23_w = np.linalg.norm(traj_w[:,1] - traj_w[:,2], axis=1)

    # Instantaneous eccentricity variation (capped at E_PAIR_CAP per sample)
    K     = traj_w.shape[0]
    step_ = max(1, K // 50)
    e12_arr, e13_arr, e23_arr = [], [], []
    for i in range(0, K - 2, step_):
        r1 = traj_w[i,0]; r2 = traj_w[i,1]; r3 = traj_w[i,2]
        if i > 0:
            v1 = (traj_w[i+1,0] - traj_w[i-1,0]) / (2*dt)
            v2 = (traj_w[i+1,1] - traj_w[i-1,1]) / (2*dt)
            v3 = (traj_w[i+1,2] - traj_w[i-1,2]) / (2*dt)
        else:
            v1 = (traj_w[1,0] - r1) / dt
            v2 = (traj_w[1,1] - r2) / dt
            v3 = (traj_w[1,2] - r3) / dt
        e12_arr.append(_inst_e_safe(r1,r2, v1,v2, m1,m2, G))
        e13_arr.append(_inst_e_safe(r1,r3, v1,v3, m1,m3, G))
        e23_arr.append(_inst_e_safe(r2,r3, v2,v3, m2,m3, G))

    s = suffix
    return {
        f"dE_max{s}":   dE_max,
        f"dE_slope{s}": dE_slope,
        f"dL_max{s}":   dL_max,
        f"r_min_12{s}": float(r12_w.min()),
        f"r_min_13{s}": float(r13_w.min()),
        f"r_min_23{s}": float(r23_w.min()),
        f"e12_std{s}":  float(np.std(e12_arr)) if e12_arr else 0.0,
        f"e13_std{s}":  float(np.std(e13_arr)) if e13_arr else 0.0,
        f"e23_std{s}":  float(np.std(e23_arr)) if e23_arr else 0.0,
    }


def label_result(
    result: ThreeBodyResult,
    megno_threshold: float = MEGNO_CHAOTIC,
) -> tuple[ThreeBodyResult, dict]:
    """
    Label a simulation outcome and extract features at two windows.

    Returns
    -------
    result : ThreeBodyResult with outcome fields set
    features : dict of all extracted features (two windows + labels)
    """
    traj   = result.traj
    E_hist = result.E_hist
    L_hist = result.L_hist
    time_  = result.time
    sys_   = result.system
    N      = traj.shape[0]
    dt     = sys_.dt
    E0     = result.E0
    L0     = result.L0

    # System scale
    r12_0 = np.linalg.norm(traj[0,0] - traj[0,1])
    r13_0 = np.linalg.norm(traj[0,0] - traj[0,2])
    r23_0 = np.linalg.norm(traj[0,1] - traj[0,2])
    initial_scale = max(r12_0, r13_0, r23_0, 1e-6)
    r_ejection  = initial_scale * 10.0
    r_collision = max(0.001, initial_scale * 0.005)

    outcome       = "stable"
    ejection_step = None
    collision_step = None

    # ── 1. EJECTION ─────────────────────────────────────────────────────────
    if E0 > 0:
        outcome = "ejection"; ejection_step = 0

    if outcome == "stable":
        for i in range(N):
            r1,r2,r3 = traj[i,0], traj[i,1], traj[i,2]
            if max(np.linalg.norm(r1-r2),
                   np.linalg.norm(r1-r3),
                   np.linalg.norm(r2-r3)) > r_ejection:
                outcome = "ejection"; ejection_step = i; break

    # ── 2. CHAOTIC ──────────────────────────────────────────────────────────
    if outcome == "stable":
        meg = result.MEGNO_final
        if not np.isnan(meg) and meg > megno_threshold:
            outcome = "chaotic"

    # ── 3. COLLISION ────────────────────────────────────────────────────────
    if outcome == "stable":
        for i in range(N):
            r1,r2,r3 = traj[i,0], traj[i,1], traj[i,2]
            d12 = np.linalg.norm(r1-r2)
            d13 = np.linalg.norm(r1-r3)
            d23 = np.linalg.norm(r2-r3)
            if d12 < r_collision or d13 < r_collision or d23 < r_collision:
                outcome = "collision"; collision_step = i; break

    result.outcome       = outcome
    result.outcome_class = OUTCOME_CLASSES[outcome]
    result.ejection_step  = ejection_step
    result.collision_step = collision_step

    # ── Feature extraction at 5% to 30% windows ────────────────────────────
    all_feats = {}

    for w in WINDOWS:
        K = max(2, int(N * (w / 100.0)))

        feats_w = _extract_window_features(
            traj[:K],
            E_hist[:K],
            L_hist[:K],
            time_[:K],
            sys_,
            dt,
            E0,
            L0,
            suffix=f"_w{w}"
        )

        all_feats.update(feats_w)

    # Keep 20% as reference window (important for filters + compatibility)
    w_ref = 20

    result.dE_max = all_feats[f"dE_max_w{w_ref}"]
    result.dL_max = all_feats[f"dL_max_w{w_ref}"]
    result.dE_slope = all_feats[f"dE_slope_w{w_ref}"]

    result.r_min_12 = all_feats[f"r_min_12_w{w_ref}"]
    result.r_min_13 = all_feats[f"r_min_13_w{w_ref}"]
    result.r_min_23 = all_feats[f"r_min_23_w{w_ref}"]

    result.e12_std = all_feats[f"e12_std_w{w_ref}"]
    result.e13_std = all_feats[f"e13_std_w{w_ref}"]
    result.e23_std = all_feats[f"e23_std_w{w_ref}"]

    features = all_feats
    return result, features