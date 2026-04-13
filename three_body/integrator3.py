"""
three_body/integrator3.py
==========================
Velocity Verlet integrator for the planar three-body problem.

Physics
-------
Three point masses m1, m2, m3. All forces pairwise, no approximations.

    F_ij = G * mi * mj * (rj - ri) / |rj - ri|^3

Conserved globally (NOT per-body):
    E_total = KE_total + PE_12 + PE_13 + PE_23
    L_total = Σ mi (ri × vi)_z

Unlike 2-body: NO analytical orbit equation, NO Kepler residuals.
Instantaneous e(t) per pair is a diagnostic, not a constant.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThreeBodySystem:
    G: float
    m1: float; m2: float; m3: float
    r1_0: np.ndarray; r2_0: np.ndarray; r3_0: np.ndarray
    v1_0: np.ndarray; v2_0: np.ndarray; v3_0: np.ndarray
    dt: float; n_steps: int
    label: str = "unnamed"

    @property
    def M_total(self): return self.m1 + self.m2 + self.m3


@dataclass
class ThreeBodyResult:
    traj: np.ndarray       # (n_steps, 3, 2)
    E_hist: np.ndarray     # (n_steps,)
    L_hist: np.ndarray     # (n_steps,)
    time: np.ndarray       # (n_steps,)
    E0: float; L0: float
    system: ThreeBodySystem
    MEGNO_final: float = 2.0
    # Filled by labeller
    dE_max: float = 0.0; dE_slope: float = 0.0; dL_max: float = 0.0
    e12_std: float = 0.0; e13_std: float = 0.0; e23_std: float = 0.0
    r_min_12: float = np.inf; r_min_13: float = np.inf; r_min_23: float = np.inf
    outcome: str = "stable"; outcome_class: int = 0
    ejection_step: Optional[int] = None; collision_step: Optional[int] = None

def _da(r_from, r_to, dr_from, dr_to, G, m_to, eps=1e-6):
    r = r_to - r_from
    dr = dr_to - dr_from

    r2 = np.dot(r, r) + eps**2
    r1 = np.sqrt(r2)
    r3 = r2 * r1
    r5 = r3 * r2

    return G * m_to * (
        dr / r3 - 3.0 * np.dot(r, dr) * r / r5
    )

def _a(r_from, r_to, G, m_to, eps=1e-6):
    dr = r_to - r_from
    r2 = np.dot(dr, dr) + eps**2
    return G * m_to * dr / (r2 * np.sqrt(r2))


def total_energy_3(r1,r2,r3, v1,v2,v3, m1,m2,m3, G):
    KE = 0.5*(m1*np.dot(v1,v1) + m2*np.dot(v2,v2) + m3*np.dot(v3,v3))
    PE = (-G*m1*m2/np.linalg.norm(r1-r2)
          -G*m1*m3/np.linalg.norm(r1-r3)
          -G*m2*m3/np.linalg.norm(r2-r3))
    return float(KE + PE)


def total_Lz_3(r1,r2,r3, v1,v2,v3, m1,m2,m3):
    def lz(r,v,m): return float(m*(r[0]*v[1]-r[1]*v[0]))
    return lz(r1,v1,m1)+lz(r2,v2,m2)+lz(r3,v3,m3)


def inst_e_pair(ra,rb, va,vb, ma,mb, G):
    """Instantaneous eccentricity of a two-body sub-pair."""
    mu = ma*mb/(ma+mb); k = G*ma*mb
    r  = np.linalg.norm(ra-rb)
    vr = ra-rb; vv = va-vb
    E  = 0.5*mu*np.dot(vv,vv) - k/r
    L  = mu*(vr[0]*vv[1]-vr[1]*vv[0])
    d  = 1.0 + 2.0*E*L**2/(mu*k**2)
    return float(np.sqrt(max(0.0, d)))


def run_simulation_3body(system: ThreeBodySystem,
                          compute_megno: bool = True,
                          delta0: float = 1e-8 ) -> ThreeBodyResult:
    G  = system.G; m1,m2,m3 = system.m1,system.m2,system.m3
    dt = system.dt; N = system.n_steps
    r1=system.r1_0.copy(); r2=system.r2_0.copy(); r3=system.r3_0.copy()
    v1=system.v1_0.copy(); v2=system.v2_0.copy(); v3=system.v3_0.copy()

    traj   = np.empty((N,3,2)); E_hist = np.empty(N); L_hist = np.empty(N)

    E0 = total_energy_3(r1,r2,r3, v1,v2,v3, m1,m2,m3, G)
    L0 = total_Lz_3(r1,r2,r3, v1,v2,v3, m1,m2,m3)
    KE0 = 0.5*(m1*np.dot(v1,v1)+m2*np.dot(v2,v2)+m3*np.dot(v3,v3))
    E_sc = abs(E0) if abs(E0) > 1e-10*KE0 else KE0

    if compute_megno:
        dr1 = np.zeros(2);
        dr1[0] = delta0
        dr2 = np.zeros(2)
        dr3 = np.zeros(2)

        dv1 = np.zeros(2)
        dv2 = np.zeros(2)
        dv3 = np.zeros(2)

        W = 0.0
        t_m = 0.0

    for i in range(N):
        a1 = _a(r1,r2,G,m2)+_a(r1,r3,G,m3)
        a2 = _a(r2,r1,G,m1)+_a(r2,r3,G,m3)
        a3 = _a(r3,r1,G,m1)+_a(r3,r2,G,m2)
        r1n=r1+v1*dt+0.5*a1*dt**2
        r2n=r2+v2*dt+0.5*a2*dt**2
        r3n=r3+v3*dt+0.5*a3*dt**2
        a1n=_a(r1n,r2n,G,m2)+_a(r1n,r3n,G,m3)
        a2n=_a(r2n,r1n,G,m1)+_a(r2n,r3n,G,m3)
        a3n=_a(r3n,r1n,G,m1)+_a(r3n,r2n,G,m2)
        v1+=0.5*(a1+a1n)*dt; v2+=0.5*(a2+a2n)*dt; v3+=0.5*(a3+a3n)*dt
        r1=r1n; r2=r2n; r3=r3n
        traj[i,0]=r1; traj[i,1]=r2; traj[i,2]=r3
        E_hist[i]=total_energy_3(r1,r2,r3,v1,v2,v3,m1,m2,m3,G)
        L_hist[i]=total_Lz_3(r1,r2,r3,v1,v2,v3,m1,m2,m3)

        if compute_megno:
            t_m += dt

            # Variation of acceleration at the current step
            da1 = _da(r1, r2, dr1, dr2, G, m2) + _da(r1, r3, dr1, dr3, G, m3)
            da2 = _da(r2, r1, dr2, dr1, G, m1) + _da(r2, r3, dr2, dr3, G, m3)
            da3 = _da(r3, r1, dr3, dr1, G, m1) + _da(r3, r2, dr3, dr2, G, m2)

            # Verlet update for deviation vectors
            dr1n = dr1 + dv1 * dt + 0.5 * da1 * dt ** 2
            dr2n = dr2 + dv2 * dt + 0.5 * da2 * dt ** 2
            dr3n = dr3 + dv3 * dt + 0.5 * da3 * dt ** 2

            # Use the NEW positions for the next-step variational acceleration
            da1n = _da(r1n, r2n, dr1n, dr2n, G, m2) + _da(r1n, r3n, dr1n, dr3n, G, m3)
            da2n = _da(r2n, r1n, dr2n, dr1n, G, m1) + _da(r2n, r3n, dr2n, dr3n, G, m3)
            da3n = _da(r3n, r1n, dr3n, dr1n, G, m1) + _da(r3n, r2n, dr3n, dr2n, G, m2)

            dv1n = dv1 + 0.5 * (da1 + da1n) * dt
            dv2n = dv2 + 0.5 * (da2 + da2n) * dt
            dv3n = dv3 + 0.5 * (da3 + da3n) * dt

            delta = np.concatenate([dr1n, dr2n, dr3n, dv1n, dv2n, dv3n])
            norm_delta = np.linalg.norm(delta) + 1e-300
            delta_dot = np.concatenate([dv1n, dv2n, dv3n, da1n, da2n, da3n])

            growth = np.dot(delta, delta_dot) / (norm_delta ** 2)
            W += growth * t_m * dt

            # Commit the step
            dr1, dr2, dr3 = dr1n, dr2n, dr3n
            dv1, dv2, dv3 = dv1n, dv2n, dv3n

            # Renormalize if it grows too large or too small
            if norm_delta > delta0 * 1e3 or norm_delta < delta0 * 1e-3:
                scale = delta0 / norm_delta
                dr1 *= scale;
                dr2 *= scale;
                dr3 *= scale
                dv1 *= scale;
                dv2 *= scale;
                dv3 *= scale

    total_time = N * dt
    megno = float(2.0 * W / total_time) if compute_megno else np.nan
    res = ThreeBodyResult(
        traj=traj, E_hist=E_hist, L_hist=L_hist,
        time=np.linspace(0.0, N*dt, N), E0=E0, L0=L0,
        system=system, MEGNO_final=megno
    )
    return res


def figure_eight_system(dt=0.001, n_periods=3) -> ThreeBodySystem:
    """Chenciner-Montgomery figure-8 (equal masses, G=1, T≈6.3259)."""
    T = 6.3259319
    return ThreeBodySystem(
        G=1.0, m1=1.0, m2=1.0, m3=1.0,
        r1_0=np.array([ 0.97000436,-0.24308753]),
        r2_0=np.array([-0.97000436, 0.24308753]),
        r3_0=np.array([ 0.0, 0.0]),
        v1_0=np.array([ 0.93240737/2,  0.86473146/2]),
        v2_0=np.array([ 0.93240737/2,  0.86473146/2]),
        v3_0=np.array([-0.93240737,   -0.86473146]),
        dt=dt, n_steps=int(T*n_periods/dt),
        label="figure_eight",
    )


def validate_figure_eight(dt=0.001, n_periods=10, verbose=True):
    import time as _t
    sys_ = figure_eight_system(dt=dt, n_periods=n_periods)
    t0 = _t.time()
    res = run_simulation_3body(sys_, compute_megno=True)
    elapsed = _t.time()-t0
    KE0 = 0.5*(sys_.m1*np.dot(sys_.v1_0,sys_.v1_0)+
               sys_.m2*np.dot(sys_.v2_0,sys_.v2_0)+
               sys_.m3*np.dot(sys_.v3_0,sys_.v3_0))
    E_sc = abs(res.E0) if abs(res.E0)>1e-10*KE0 else KE0
    dE   = float(np.max(np.abs((res.E_hist-res.E0)/E_sc)))
    L_sc = max(abs(res.L0), 1e-6)
    dL   = float(np.max(np.abs((res.L_hist-res.L0)/L_sc)))
    if verbose:
        print(f"\nFigure-8 validation  ({n_periods} periods, dt={dt})")
        print(f"  Steps:  {sys_.n_steps:,}")
        print(f"  Time:   {elapsed:.1f}s")
        print(f"  E0:     {res.E0:.6f}")
        print(f"  dE_max: {dE:.2e}  (target < 1e-3)")
        print(f"  dL_max: {dL:.2e}  (target < 1e-3)")
        print(f"  MEGNO:  {res.MEGNO_final:.4f}  (figure-8 is regular → expect ~2)")
        ok = dE < 1e-3 and dL < 1e-3
        print(f"  Status: {'PASS ✓' if ok else 'FAIL ✗'}")
    return {"dE_max":dE,"dL_max":dL,"MEGNO":res.MEGNO_final}